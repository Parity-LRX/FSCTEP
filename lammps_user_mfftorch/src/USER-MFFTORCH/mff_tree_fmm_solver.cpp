#include "mff_tree_fmm_solver.h"

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mfftorch {

namespace {

struct TreeNode {
  std::vector<int> indices;
  torch::Tensor center;
  double half_extent = 0.0;
  torch::Tensor index_tensor;
  std::vector<std::unique_ptr<TreeNode>> children;

  bool is_leaf() const { return children.empty(); }

  bool contains(int target_idx) const {
    return std::find(indices.begin(), indices.end(), target_idx) != indices.end();
  }
};

struct RankClusterSummary {
  int rank = 0;
  int leaf_id = -1;
  int64_t natoms = 0;
  double charge = 0.0;
  double half_extent = 0.0;
  std::array<double, 3> center{{0.0, 0.0, 0.0}};
  std::array<double, 3> bbox_min{{0.0, 0.0, 0.0}};
  std::array<double, 3> bbox_max{{0.0, 0.0, 0.0}};
  std::vector<int> indices;

  bool has_atoms() const { return natoms > 0; }
};

struct ImportedAtoms {
  torch::Tensor pos;
  torch::Tensor source;
};

struct ExactContribution {
  torch::Tensor atom_energy;
  torch::Tensor forces;
};

struct RemoteLeafPayload {
  torch::Tensor ranks;
  torch::Tensor leaf_ids;
  torch::Tensor charges;
  torch::Tensor half_extents;
  torch::Tensor centers;

  int64_t size() const { return leaf_ids.defined() ? leaf_ids.size(0) : 0; }
};

struct DeviceRequestPlan {
  torch::Tensor far_indices;
  torch::Tensor request_leaf_ids;
  torch::Tensor request_ranks;
  torch::Tensor request_counts_per_rank;
  torch::Tensor request_rank_offsets;

  int64_t size() const { return request_leaf_ids.defined() ? request_leaf_ids.size(0) : 0; }
};

struct DistributedCommStats {
  int64_t total_leafs = 0;
  int64_t charge_leafs = 0;
  int64_t geometry_refreshed_leafs = 0;
  int64_t requested_leafs = 0;
  int64_t imported_atoms = 0;
};

bool read_bool_env(const char* name, bool fallback) {
  if (const char* env = std::getenv(name)) {
    const std::string value(env);
    if (value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON" || value == "yes" ||
        value == "YES") {
      return true;
    }
    if (value == "0" || value == "false" || value == "FALSE" || value == "off" || value == "OFF" || value == "no" ||
        value == "NO") {
      return false;
    }
  }
  return fallback;
}

double read_nonnegative_env(const char* name, double fallback) {
  if (const char* env = std::getenv(name)) {
    const double parsed = std::atof(env);
    if (parsed >= 0.0) return parsed;
  }
  return fallback;
}

bool env_enabled(const char* name) {
  if (const char* env = std::getenv(name)) {
    const std::string value(env);
    return value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON" || value == "yes" ||
           value == "YES";
  }
  return false;
}

bool distributed_profile_enabled() { return env_enabled("MFF_TREE_FMM_PROFILE_DISTRIBUTED"); }

bool gpu_aware_mpi_runtime_enabled(bool requested, const torch::Device& device) {
  if (!requested || !device.is_cuda()) return false;
  if (env_enabled("MFF_TREE_FMM_ASSUME_GPU_AWARE_MPI")) return true;
  return env_enabled("OMPI_MCA_opal_cuda_support") || env_enabled("MPICH_GPU_SUPPORT_ENABLED") ||
         env_enabled("MV2_USE_CUDA") || env_enabled("PSM2_CUDA") || env_enabled("SLURM_MPI_GPU_SUPPORT");
}

torch::Tensor cpu_float_tensor_from_buffer(const std::vector<float>& buf, std::vector<int64_t> shape) {
  int64_t numel = 1;
  for (const auto dim : shape) numel *= dim;
  if (numel == 0) {
    return torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  }
  return torch::from_blob(
             const_cast<float*>(buf.data()),
             shape,
             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
      .clone();
}

torch::Tensor screened_coulomb_kernel(const torch::Tensor& distance, double screening) {
  return torch::exp(-screening * distance) / distance;
}

torch::Tensor screened_coulomb_force_prefactor(const torch::Tensor& distance, double screening) {
  return torch::exp(-screening * distance) * (screening * distance + 1.0) / (distance * distance * distance);
}

std::vector<int> build_displs(const std::vector<int>& counts) {
  std::vector<int> displs(counts.size(), 0);
  for (size_t i = 1; i < counts.size(); ++i) displs[i] = displs[i - 1] + counts[i - 1];
  return displs;
}

torch::Tensor cpu_long_tensor_from_buffer(const std::vector<int64_t>& buf, std::vector<int64_t> shape) {
  int64_t numel = 1;
  for (const auto dim : shape) numel *= dim;
  if (numel == 0) {
    return torch::empty(shape, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  }
  return torch::from_blob(
             const_cast<int64_t*>(buf.data()),
             shape,
             torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
      .clone();
}

std::vector<int> tensor_to_int_vector(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) return {};
  auto cpu = tensor.to(torch::kCPU, torch::kInt32).contiguous().view({-1});
  const int* ptr = cpu.data_ptr<int>();
  return std::vector<int>(ptr, ptr + cpu.numel());
}

RemoteLeafPayload empty_remote_leaf_payload(const torch::Device& device) {
  RemoteLeafPayload payload;
  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  payload.ranks = torch::empty({0}, long_opts);
  payload.leaf_ids = torch::empty({0}, long_opts);
  payload.charges = torch::empty({0}, float_opts);
  payload.half_extents = torch::empty({0}, float_opts);
  payload.centers = torch::empty({0, 3}, float_opts);
  return payload;
}

DeviceRequestPlan empty_device_request_plan(const torch::Device& device, int world_size) {
  DeviceRequestPlan plan;
  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  plan.far_indices = torch::empty({0}, long_opts);
  plan.request_leaf_ids = torch::empty({0}, long_opts);
  plan.request_ranks = torch::empty({0}, long_opts);
  plan.request_counts_per_rank = torch::zeros({std::max(world_size, 0)}, long_opts);
  plan.request_rank_offsets = torch::zeros({std::max(world_size, 0)}, long_opts);
  return plan;
}

bool int64_vectors_equal(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

std::vector<torch::Tensor> split_flat_tensor_by_counts(
    const torch::Tensor& flat,
    const std::vector<int64_t>& counts) {
  std::vector<torch::Tensor> out;
  out.reserve(counts.size());
  int64_t offset = 0;
  for (size_t rank = 0; rank < counts.size(); ++rank) {
    const int64_t count = counts[rank];
    if (count <= 0) {
      if (flat.dim() == 2) {
        out.push_back(torch::empty({0, flat.size(1)}, flat.options()));
      } else {
        out.push_back(torch::empty({0}, flat.options()));
      }
      continue;
    }
    out.push_back(flat.narrow(0, offset, count).contiguous());
    offset += count;
  }
  return out;
}

torch::Tensor build_rank_tensor(const std::vector<int64_t>& leaf_counts, const torch::Device& device) {
  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  std::vector<torch::Tensor> segments;
  segments.reserve(leaf_counts.size());
  for (size_t rank = 0; rank < leaf_counts.size(); ++rank) {
    const int64_t count = leaf_counts[rank];
    if (count <= 0) continue;
    segments.push_back(torch::full({count}, static_cast<int64_t>(rank), long_opts));
  }
  if (segments.empty()) return torch::empty({0}, long_opts);
  return torch::cat(segments, 0).contiguous();
}

RemoteLeafPayload assemble_remote_leaf_payload(
    const MFFTreeFmmSolver::DistributedRemoteGeometryCache& geometry_cache,
    const std::vector<torch::Tensor>& charges_by_rank,
    const torch::Device& device) {
  RemoteLeafPayload payload = empty_remote_leaf_payload(device);
  if (!geometry_cache.valid || geometry_cache.leaf_counts.empty()) return payload;

  std::vector<torch::Tensor> leaf_ids;
  std::vector<torch::Tensor> charges;
  std::vector<torch::Tensor> half_extents;
  std::vector<torch::Tensor> centers;
  leaf_ids.reserve(geometry_cache.leaf_counts.size());
  charges.reserve(geometry_cache.leaf_counts.size());
  half_extents.reserve(geometry_cache.leaf_counts.size());
  centers.reserve(geometry_cache.leaf_counts.size());

  for (size_t rank = 0; rank < geometry_cache.leaf_counts.size(); ++rank) {
    const int64_t count = geometry_cache.leaf_counts[rank];
    if (count <= 0) continue;
    leaf_ids.push_back(geometry_cache.leaf_ids_by_rank[rank]);
    charges.push_back(charges_by_rank[rank]);
    half_extents.push_back(geometry_cache.half_extents_by_rank[rank]);
    centers.push_back(geometry_cache.centers_by_rank[rank]);
  }

  if (leaf_ids.empty()) return payload;
  payload.ranks = build_rank_tensor(geometry_cache.leaf_counts, device);
  payload.leaf_ids = torch::cat(leaf_ids, 0).contiguous();
  payload.charges = torch::cat(charges, 0).contiguous();
  payload.half_extents = torch::cat(half_extents, 0).contiguous();
  payload.centers = torch::cat(centers, 0).contiguous();
  return payload;
}

bool distributed_request_plan_cache_is_valid(
    const MFFTreeFmmSolver::DistributedRequestPlanCache& cache,
    int world_size,
    int world_rank,
    int64_t local_geometry_version,
    const std::vector<int64_t>& remote_geometry_versions,
    const std::vector<int64_t>& remote_leaf_counts) {
  return cache.valid && cache.world_size == world_size && cache.world_rank == world_rank &&
         cache.local_geometry_version == local_geometry_version &&
         int64_vectors_equal(cache.remote_geometry_versions, remote_geometry_versions) &&
         int64_vectors_equal(cache.remote_leaf_counts, remote_leaf_counts);
}

bool linear_tree_cache_is_valid(
    const MFFTreeFmmSolver::LinearTreeCache& cache,
    const torch::Tensor& local_ids,
    const torch::Tensor& pos,
    int leaf_size,
    double reuse_position_tol) {
  if (!cache.valid || reuse_position_tol <= 0.0 || cache.leaf_size != leaf_size) return false;
  if (!cache.local_global_ids.defined() || !cache.local_pos.defined()) return false;
  if (cache.local_global_ids.device() != local_ids.device() || cache.local_pos.device() != pos.device()) return false;
  if (cache.local_global_ids.sizes() != local_ids.sizes() || cache.local_pos.sizes() != pos.sizes()) return false;
  if (local_ids.numel() > 0 && !(cache.local_global_ids == local_ids).all().item<bool>()) return false;
  if (pos.numel() == 0) return true;
  return (cache.local_pos - pos).abs().max().item<double>() <= reuse_position_tol;
}

torch::Tensor compute_spatial_sort_keys(const torch::Tensor& pos) {
  if (pos.numel() == 0) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(pos.device()));
  }
  auto coord_min = std::get<0>(pos.min(0));
  auto coord_max = std::get<0>(pos.max(0));
  auto span = (coord_max - coord_min).clamp_min(1.0e-6);
  auto normalized = ((pos - coord_min) / span).clamp(0.0, 1.0);
  auto grid = torch::round(normalized * 1023.0).to(torch::kInt64);
  auto x = grid.select(1, 0);
  auto y = grid.select(1, 1);
  auto z = grid.select(1, 2);
  return x * 1048576 + y * 1024 + z;
}

void build_or_reuse_linear_tree_cache(
    MFFTreeFmmSolver::LinearTreeCache& cache,
    const torch::Tensor& pos,
    const torch::Tensor& source,
    const torch::Tensor& local_ids,
    int leaf_size,
    double reuse_position_tol) {
  const int64_t natoms = pos.size(0);
  const auto device = pos.device();
  const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);

  if (!linear_tree_cache_is_valid(cache, local_ids, pos, leaf_size, reuse_position_tol)) {
    cache = MFFTreeFmmSolver::LinearTreeCache{};
    cache.local_global_ids = local_ids.clone();
    cache.local_pos = pos.clone();
    cache.leaf_size = leaf_size;
    cache.valid = true;

    auto sort_keys = compute_spatial_sort_keys(pos);
    cache.permutation = std::get<1>(sort_keys.sort());
    cache.inverse_permutation = torch::empty_like(cache.permutation);
    auto sorted_index = torch::arange(natoms, long_opts);
    cache.inverse_permutation.index_copy_(0, cache.permutation, sorted_index);

    if (natoms == 0) {
      cache.sorted_pos = torch::empty({0, 3}, float_opts);
      cache.sorted_source = torch::empty({0}, float_opts);
      cache.sorted_global_ids = torch::empty({0}, long_opts);
      cache.leaf_offsets = torch::zeros({1}, long_opts);
      cache.leaf_counts = torch::empty({0}, long_opts);
      cache.leaf_centers = torch::empty({0, 3}, float_opts);
      cache.leaf_bbox_min = torch::empty({0, 3}, float_opts);
      cache.leaf_bbox_max = torch::empty({0, 3}, float_opts);
      cache.leaf_half_extents = torch::empty({0}, float_opts);
      cache.leaf_charges = torch::empty({0}, float_opts);
      cache.leaf_ids = torch::empty({0}, long_opts);
      return;
    }

    auto starts = torch::arange(0, natoms, leaf_size, long_opts);
    cache.leaf_offsets = torch::cat({starts, torch::tensor({natoms}, long_opts)}, 0);
  } else {
    cache.local_pos = pos.clone();
  }

  cache.sorted_pos = pos.index_select(0, cache.permutation).contiguous();
  cache.sorted_source = source.index_select(0, cache.permutation).contiguous();
  cache.sorted_global_ids = local_ids.index_select(0, cache.permutation).contiguous();

  const int64_t num_leaves = std::max<int64_t>(cache.leaf_offsets.size(0) - 1, 0);
  cache.leaf_counts = (num_leaves > 0)
                          ? (cache.leaf_offsets.slice(0, 1, num_leaves + 1) - cache.leaf_offsets.slice(0, 0, num_leaves))
                          : torch::empty({0}, long_opts);
  cache.leaf_centers = torch::empty({num_leaves, 3}, float_opts);
  cache.leaf_bbox_min = torch::empty({num_leaves, 3}, float_opts);
  cache.leaf_bbox_max = torch::empty({num_leaves, 3}, float_opts);
  cache.leaf_half_extents = torch::empty({num_leaves}, float_opts);
  cache.leaf_charges = torch::empty({num_leaves}, float_opts);
  cache.leaf_ids = torch::arange(num_leaves, long_opts);

  auto offsets_cpu = cache.leaf_offsets.to(torch::kCPU, torch::kInt64);
  const int64_t* offset_ptr = offsets_cpu.data_ptr<int64_t>();
  for (int64_t leaf = 0; leaf < num_leaves; ++leaf) {
    const int64_t begin = offset_ptr[leaf];
    const int64_t end = offset_ptr[leaf + 1];
    const int64_t count = end - begin;
    if (count <= 0) {
      cache.leaf_centers.index_put_({leaf}, torch::zeros({3}, float_opts));
      cache.leaf_bbox_min.index_put_({leaf}, torch::zeros({3}, float_opts));
      cache.leaf_bbox_max.index_put_({leaf}, torch::zeros({3}, float_opts));
      cache.leaf_half_extents.index_put_({leaf}, 0.0f);
      cache.leaf_charges.index_put_({leaf}, 0.0f);
      continue;
    }
    auto leaf_pos = cache.sorted_pos.narrow(0, begin, count);
    auto leaf_source = cache.sorted_source.narrow(0, begin, count);
    auto coord_min = std::get<0>(leaf_pos.min(0));
    auto coord_max = std::get<0>(leaf_pos.max(0));
    auto center = 0.5f * (coord_min + coord_max);
    cache.leaf_centers.index_put_({leaf}, center);
    cache.leaf_bbox_min.index_put_({leaf}, coord_min);
    cache.leaf_bbox_max.index_put_({leaf}, coord_max);
    cache.leaf_half_extents.index_put_({leaf}, 0.5f * (coord_max - coord_min).max());
    cache.leaf_charges.index_put_({leaf}, leaf_source.sum());
  }
}

std::unique_ptr<TreeNode> build_tree(
    const torch::Tensor& pos_detached,
    const std::vector<int>& indices,
    int leaf_size) {
  auto node = std::make_unique<TreeNode>();
  node->indices = indices;
  auto idx = torch::tensor(indices, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  node->index_tensor = idx;
  auto local = pos_detached.index_select(0, idx);
  auto coord_min = std::get<0>(local.min(0));
  auto coord_max = std::get<0>(local.max(0));
  node->center = 0.5f * (coord_min + coord_max);
  node->half_extent = 0.5 * (coord_max - coord_min).max().item<double>();
  if (static_cast<int>(indices.size()) <= leaf_size) return node;

  std::array<std::vector<int>, 8> child_bins;
  for (int atom_idx : indices) {
    auto atom = pos_detached.index({atom_idx});
    const auto cx = node->center.index({0}).item<float>();
    const auto cy = node->center.index({1}).item<float>();
    const auto cz = node->center.index({2}).item<float>();
    const int key = (atom.index({0}).item<float>() >= cx ? 1 : 0) * 4 +
                    (atom.index({1}).item<float>() >= cy ? 1 : 0) * 2 +
                    (atom.index({2}).item<float>() >= cz ? 1 : 0);
    child_bins[static_cast<size_t>(key)].push_back(atom_idx);
  }
  int non_empty_children = 0;
  for (const auto& child : child_bins) {
    if (!child.empty()) ++non_empty_children;
  }
  if (non_empty_children <= 1) return node;
  for (const auto& child : child_bins) {
    if (!child.empty()) node->children.push_back(build_tree(pos_detached, child, leaf_size));
  }
  return node;
}

torch::Tensor approximate_node_potential(
    const TreeNode& node,
    const torch::Tensor& pos,
    const torch::Tensor& source,
    int target_idx,
    const torch::Tensor& target_pos,
    const TreeFmmConfig& config) {
  auto scalar_zero = torch::zeros({}, pos.options());
  if (!node.index_tensor.defined() || node.index_tensor.numel() == 0) return scalar_zero;

  const bool contains_target = node.contains(target_idx);
  auto local_pos = pos.index_select(0, node.index_tensor);
  auto local_source = source.index_select(0, node.index_tensor);

  if (node.is_leaf()) {
    if (contains_target) {
      auto mask = node.index_tensor != target_idx;
      local_pos = local_pos.index({mask});
      local_source = local_source.index({mask});
    }
    if (local_source.numel() == 0) return scalar_zero;
    auto delta = target_pos.unsqueeze(0) - local_pos;
    auto distance = torch::sqrt((delta * delta).sum(-1) + config.softening * config.softening);
    auto kernel = screened_coulomb_kernel(distance, config.screening);
    return (local_source * kernel).sum();
  }

  auto delta_center = target_pos - node.center;
  auto dist_center = torch::sqrt((delta_center * delta_center).sum() + config.softening * config.softening).item<double>();
  const double cell_diameter = std::max(2.0 * node.half_extent, 1.0e-9);
  if (!contains_target && cell_diameter / std::max(dist_center, 1.0e-9) < config.theta) {
    auto cluster_charge = local_source.sum();
    auto cluster_center = local_pos.mean(0);
    auto delta = target_pos - cluster_center;
    auto distance = torch::sqrt((delta * delta).sum() + config.softening * config.softening);
    return cluster_charge * screened_coulomb_kernel(distance, config.screening);
  }

  auto total = scalar_zero;
  for (const auto& child : node.children) {
    total = total + approximate_node_potential(*child, pos, source, target_idx, target_pos, config);
  }
  return total;
}

void collect_leaf_summaries(
    const TreeNode& node,
    const torch::Tensor& pos,
    const torch::Tensor& source,
    int rank,
    int& next_leaf_id,
    std::vector<RankClusterSummary>& out) {
  if (node.is_leaf()) {
    RankClusterSummary summary;
    summary.rank = rank;
    summary.leaf_id = next_leaf_id++;
    summary.natoms = static_cast<int64_t>(node.indices.size());
    summary.indices = node.indices;
    if (summary.natoms > 0) {
      auto local_pos = pos.index_select(0, node.index_tensor);
      auto local_source = source.index_select(0, node.index_tensor);
      auto coord_min = std::get<0>(local_pos.min(0));
      auto coord_max = std::get<0>(local_pos.max(0));
      auto center = 0.5f * (coord_min + coord_max);
      summary.charge = local_source.sum().item<double>();
      summary.half_extent = 0.5 * (coord_max - coord_min).max().item<double>();
      for (int axis = 0; axis < 3; ++axis) {
        summary.center[axis] = center.index({axis}).item<double>();
        summary.bbox_min[axis] = coord_min.index({axis}).item<double>();
        summary.bbox_max[axis] = coord_max.index({axis}).item<double>();
      }
    }
    out.push_back(summary);
    return;
  }
  for (const auto& child : node.children) {
    collect_leaf_summaries(*child, pos, source, rank, next_leaf_id, out);
  }
}

std::unique_ptr<MFFTreeFmmSolver::CachedTreeNode> clone_tree_topology(const TreeNode& node) {
  auto cached = std::make_unique<MFFTreeFmmSolver::CachedTreeNode>();
  cached->indices = node.indices;
  cached->children.reserve(node.children.size());
  for (const auto& child : node.children) {
    cached->children.push_back(clone_tree_topology(*child));
  }
  return cached;
}

std::unique_ptr<TreeNode> rebuild_tree_from_topology(
    const MFFTreeFmmSolver::CachedTreeNode& cached,
    const torch::Tensor& pos_detached) {
  auto node = std::make_unique<TreeNode>();
  node->indices = cached.indices;
  node->index_tensor = torch::tensor(
      cached.indices,
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto local = pos_detached.index_select(0, node->index_tensor);
  auto coord_min = std::get<0>(local.min(0));
  auto coord_max = std::get<0>(local.max(0));
  node->center = 0.5f * (coord_min + coord_max);
  node->half_extent = 0.5 * (coord_max - coord_min).max().item<double>();
  node->children.reserve(cached.children.size());
  for (const auto& child : cached.children) {
    node->children.push_back(rebuild_tree_from_topology(*child, pos_detached));
  }
  return node;
}

bool tree_cache_is_valid(
    const torch::Tensor& cached_ids_cpu,
    const torch::Tensor& cached_pos_cpu,
    const std::unique_ptr<MFFTreeFmmSolver::CachedTreeNode>& cached_tree_topology,
    int cached_leaf_size,
    const torch::Tensor& local_ids,
    const torch::Tensor& pos_detached,
    int leaf_size,
    double reuse_position_tol) {
  if (reuse_position_tol <= 0.0 || !cached_tree_topology || cached_leaf_size != leaf_size) return false;
  if (!cached_ids_cpu.defined() || !cached_pos_cpu.defined()) return false;
  auto local_ids_cpu = local_ids.to(torch::kCPU, torch::kInt64).contiguous();
  if (local_ids_cpu.sizes() != cached_ids_cpu.sizes()) return false;
  if (local_ids_cpu.numel() > 0) {
    if (!(local_ids_cpu == cached_ids_cpu).all().item<bool>()) return false;
  }
  auto pos_cpu = pos_detached.to(torch::kCPU, torch::kFloat32).contiguous();
  if (pos_cpu.sizes() != cached_pos_cpu.sizes()) return false;
  if (pos_cpu.numel() == 0) return true;
  auto max_disp = (pos_cpu - cached_pos_cpu).abs().max().item<double>();
  return max_disp <= reuse_position_tol;
}

bool leaf_requires_exact_exchange(
    const RankClusterSummary& local_leaf,
    const RankClusterSummary& remote_leaf,
    double theta);

torch::Tensor summary_point_potential(
    const torch::Tensor& query_pos,
    const std::vector<RankClusterSummary>& far_summaries,
    const TreeFmmConfig& config);

torch::Tensor exact_point_potential(
    const torch::Tensor& query_pos,
    const torch::Tensor& remote_pos,
    const torch::Tensor& remote_source,
    const TreeFmmConfig& config) {
  if (!remote_pos.defined() || remote_pos.numel() == 0 || !remote_source.defined() || remote_source.numel() == 0) {
    return torch::zeros({query_pos.size(0)}, query_pos.options());
  }
  auto delta = query_pos.unsqueeze(1) - remote_pos.unsqueeze(0);
  auto distance = torch::sqrt((delta * delta).sum(-1) + config.softening * config.softening);
  auto kernel = screened_coulomb_kernel(distance, config.screening);
  return (kernel * remote_source.view({1, -1})).sum(1);
}

ExactContribution exact_point_atom_energy_and_forces(
    const torch::Tensor& query_pos_cpu,
    const torch::Tensor& query_source_cpu,
    const ImportedAtoms& imported,
    const TreeFmmConfig& config,
    const torch::Device& preferred_device) {
  ExactContribution out;
  out.atom_energy = torch::zeros({query_pos_cpu.size(0)}, query_pos_cpu.options());
  out.forces = torch::zeros_like(query_pos_cpu);
  if (query_pos_cpu.numel() == 0 || !imported.pos.defined() || imported.pos.numel() == 0 || !imported.source.defined() ||
      imported.source.numel() == 0) {
    return out;
  }

  const bool use_gpu = preferred_device.is_cuda();
  const auto compute_device = use_gpu ? preferred_device : torch::Device(torch::kCPU);
  auto query_pos = query_pos_cpu.to(compute_device, torch::kFloat32).contiguous().clone();
  auto query_source = query_source_cpu.to(compute_device, torch::kFloat32).contiguous();
  auto remote_pos = imported.pos.to(compute_device, torch::kFloat32).contiguous();
  auto remote_source = imported.source.to(compute_device, torch::kFloat32).contiguous();

  query_pos.set_requires_grad(true);
  auto potential = exact_point_potential(query_pos, remote_pos, remote_source, config);
  auto atom_energy = 0.5f * query_source * potential;
  atom_energy = atom_energy * config.energy_scale;
  auto total_energy = atom_energy.sum();
  auto grads = torch::autograd::grad({total_energy}, {query_pos}, {}, true, false);

  out.atom_energy = atom_energy.detach().to(torch::kCPU, torch::kFloat32).contiguous();
  out.forces = (-grads[0]).detach().to(torch::kCPU, torch::kFloat32).contiguous();
  return out;
}

torch::Tensor leaf_summary_potential(
    const torch::Tensor& query_pos,
    const std::array<double, 3>& center,
    double charge,
    const TreeFmmConfig& config) {
  if (query_pos.numel() == 0 || std::abs(charge) == 0.0) {
    return torch::zeros({query_pos.size(0)}, query_pos.options());
  }
  auto center_tensor = torch::tensor(
      {static_cast<float>(center[0]), static_cast<float>(center[1]), static_cast<float>(center[2])},
      torch::TensorOptions().dtype(torch::kFloat32).device(query_pos.device()));
  auto delta = query_pos - center_tensor.view({1, 3});
  auto distance = torch::sqrt((delta * delta).sum(-1) + config.softening * config.softening);
  return static_cast<float>(charge) * screened_coulomb_kernel(distance, config.screening);
}

torch::Tensor self_leaf_exact_potential(
    const torch::Tensor& leaf_pos,
    const torch::Tensor& leaf_source,
    const TreeFmmConfig& config) {
  if (leaf_pos.size(0) <= 1) return torch::zeros({leaf_pos.size(0)}, leaf_pos.options());
  auto delta = leaf_pos.unsqueeze(1) - leaf_pos.unsqueeze(0);
  auto distance = torch::sqrt((delta * delta).sum(-1) + config.softening * config.softening);
  auto kernel = screened_coulomb_kernel(distance, config.screening);
  auto eye = torch::eye(leaf_pos.size(0), torch::TensorOptions().dtype(torch::kBool).device(leaf_pos.device()));
  kernel.masked_fill_(eye, 0.0);
  return (kernel * leaf_source.view({1, -1})).sum(1);
}

ExactContribution summary_atom_energy_and_forces(
    const torch::Tensor& query_pos_cpu,
    const torch::Tensor& query_source_cpu,
    const std::vector<RankClusterSummary>& far_summaries,
    const TreeFmmConfig& config,
    const torch::Device& preferred_device) {
  ExactContribution out;
  out.atom_energy = torch::zeros({query_pos_cpu.size(0)}, query_pos_cpu.options());
  out.forces = torch::zeros_like(query_pos_cpu);
  if (query_pos_cpu.numel() == 0 || far_summaries.empty()) return out;

  const auto compute_device = preferred_device.is_cuda() ? preferred_device : torch::Device(torch::kCPU);
  auto query_pos = query_pos_cpu.to(compute_device, torch::kFloat32).contiguous().clone();
  auto query_source = query_source_cpu.to(compute_device, torch::kFloat32).contiguous();
  query_pos.set_requires_grad(true);
  auto potential = summary_point_potential(query_pos, far_summaries, config);
  auto atom_energy = 0.5f * query_source * potential;
  atom_energy = atom_energy * config.energy_scale;
  auto grads = torch::autograd::grad({atom_energy.sum()}, {query_pos}, {}, true, false);
  out.atom_energy = atom_energy.detach().to(torch::kCPU, torch::kFloat32).contiguous();
  out.forces = (-grads[0]).detach().to(torch::kCPU, torch::kFloat32).contiguous();
  return out;
}

ExactContribution local_leaf_device_atom_energy_and_forces(
    const torch::Tensor& pos_cpu,
    const torch::Tensor& source_cpu,
    const std::vector<RankClusterSummary>& local_leaf_summaries,
    const TreeFmmConfig& config,
    const torch::Device& preferred_device) {
  ExactContribution out;
  out.atom_energy = torch::zeros({pos_cpu.size(0)}, pos_cpu.options());
  out.forces = torch::zeros_like(pos_cpu);
  if (pos_cpu.numel() == 0 || local_leaf_summaries.empty()) return out;

  const auto compute_device = preferred_device.is_cuda() ? preferred_device : torch::Device(torch::kCPU);
  auto pos = pos_cpu.to(compute_device, torch::kFloat32).contiguous().clone();
  auto source = source_cpu.to(compute_device, torch::kFloat32).contiguous();
  pos.set_requires_grad(true);
  auto potential = torch::zeros({pos.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(compute_device));

  std::vector<torch::Tensor> leaf_indices_dev;
  leaf_indices_dev.reserve(local_leaf_summaries.size());
  for (const auto& leaf : local_leaf_summaries) {
    leaf_indices_dev.push_back(
        torch::tensor(leaf.indices, torch::TensorOptions().dtype(torch::kInt64).device(compute_device)));
  }

  for (size_t i = 0; i < local_leaf_summaries.size(); ++i) {
    const auto& leaf_i = local_leaf_summaries[i];
    const auto& idx_i = leaf_indices_dev[i];
    if (!leaf_i.has_atoms() || idx_i.numel() == 0) continue;
    auto pos_i = pos.index_select(0, idx_i);
    auto source_i = source.index_select(0, idx_i);
    auto pot_i = self_leaf_exact_potential(pos_i, source_i, config);

    for (size_t j = i + 1; j < local_leaf_summaries.size(); ++j) {
      const auto& leaf_j = local_leaf_summaries[j];
      const auto& idx_j = leaf_indices_dev[j];
      if (!leaf_j.has_atoms() || idx_j.numel() == 0) continue;
      auto pos_j = pos.index_select(0, idx_j);
      auto source_j = source.index_select(0, idx_j);
      if (leaf_requires_exact_exchange(leaf_i, leaf_j, config.theta)) {
        pot_i = pot_i + exact_point_potential(pos_i, pos_j, source_j, config);
        auto pot_j = exact_point_potential(pos_j, pos_i, source_i, config);
        potential.index_add_(0, idx_j, pot_j);
      } else {
        pot_i = pot_i + leaf_summary_potential(pos_i, leaf_j.center, leaf_j.charge, config);
        auto pot_j = leaf_summary_potential(pos_j, leaf_i.center, leaf_i.charge, config);
        potential.index_add_(0, idx_j, pot_j);
      }
    }
    potential.index_add_(0, idx_i, pot_i);
  }

  auto atom_energy = 0.5f * source * potential;
  atom_energy = atom_energy * config.energy_scale;
  auto grads = torch::autograd::grad({atom_energy.sum()}, {pos}, {}, true, false);
  out.atom_energy = atom_energy.detach().to(torch::kCPU, torch::kFloat32).contiguous();
  out.forces = (-grads[0]).detach().to(torch::kCPU, torch::kFloat32).contiguous();
  return out;
}

torch::Tensor summary_point_potential(
    const torch::Tensor& query_pos,
    const std::vector<RankClusterSummary>& far_summaries,
    const TreeFmmConfig& config) {
  if (far_summaries.empty()) return torch::zeros({query_pos.size(0)}, query_pos.options());
  std::vector<float> center_buf;
  std::vector<float> charge_buf;
  center_buf.reserve(far_summaries.size() * 3);
  charge_buf.reserve(far_summaries.size());
  for (const auto& summary : far_summaries) {
    center_buf.push_back(static_cast<float>(summary.center[0]));
    center_buf.push_back(static_cast<float>(summary.center[1]));
    center_buf.push_back(static_cast<float>(summary.center[2]));
    charge_buf.push_back(static_cast<float>(summary.charge));
  }
  auto centers = torch::from_blob(
                     center_buf.data(),
                     {static_cast<int64_t>(far_summaries.size()), 3},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                     .clone()
                     .to(query_pos.device());
  auto charges = torch::from_blob(
                     charge_buf.data(),
                     {static_cast<int64_t>(far_summaries.size())},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                     .clone()
                     .to(query_pos.device());
  auto delta = query_pos.unsqueeze(1) - centers.unsqueeze(0);
  auto distance = torch::sqrt((delta * delta).sum(-1) + config.softening * config.softening);
  auto kernel = screened_coulomb_kernel(distance, config.screening);
  return (kernel * charges.view({1, -1})).sum(1);
}

ExactContribution exact_imported_contribution_explicit(
    const torch::Tensor& query_pos,
    const torch::Tensor& query_source,
    const ImportedAtoms& imported,
    const TreeFmmConfig& config) {
  ExactContribution out;
  out.atom_energy = torch::zeros({query_pos.size(0)}, query_source.options());
  out.forces = torch::zeros_like(query_pos);
  if (!imported.pos.defined() || imported.pos.numel() == 0 || !imported.source.defined() || imported.source.numel() == 0 ||
      query_pos.numel() == 0) {
    return out;
  }

  auto remote_pos = imported.pos.to(query_pos.device(), torch::kFloat32).contiguous();
  auto remote_source = imported.source.to(query_pos.device(), torch::kFloat32).contiguous();
  auto delta = query_pos.unsqueeze(1) - remote_pos.unsqueeze(0);
  auto distance = torch::sqrt((delta * delta).sum(-1) + config.softening * config.softening);
  auto kernel = screened_coulomb_kernel(distance, config.screening);
  auto prefactor = screened_coulomb_force_prefactor(distance, config.screening);
  auto weighted_source = remote_source.view({1, -1});
  auto potential = (kernel * weighted_source).sum(1);
  auto force_mat = query_source.view({-1, 1, 1}) * weighted_source.unsqueeze(-1) * prefactor.unsqueeze(-1) * delta;

  out.atom_energy = 0.5f * query_source * potential * config.energy_scale;
  out.forces = force_mat.sum(1) * config.energy_scale;
  return out;
}

ExactContribution summary_contribution_explicit(
    const torch::Tensor& query_pos,
    const torch::Tensor& query_source,
    const RemoteLeafPayload& payload,
    const TreeFmmConfig& config) {
  ExactContribution out;
  out.atom_energy = torch::zeros({query_pos.size(0)}, query_source.options());
  out.forces = torch::zeros_like(query_pos);
  if (query_pos.numel() == 0 || payload.size() == 0) return out;

  auto centers = payload.centers.to(query_pos.device(), torch::kFloat32);
  auto charges = payload.charges.to(query_pos.device(), torch::kFloat32);
  auto delta = query_pos.unsqueeze(1) - centers.unsqueeze(0);
  auto distance = torch::sqrt((delta * delta).sum(-1) + config.softening * config.softening);
  auto kernel = screened_coulomb_kernel(distance, config.screening);
  auto prefactor = screened_coulomb_force_prefactor(distance, config.screening);
  auto potential = (kernel * charges.view({1, -1})).sum(1);
  auto force_mat =
      query_source.view({-1, 1, 1}) * charges.view({1, -1, 1}) * prefactor.unsqueeze(-1) * delta;
  out.atom_energy = 0.5f * query_source * potential * config.energy_scale;
  out.forces = force_mat.sum(1) * config.energy_scale;
  return out;
}

ExactContribution local_linear_tree_contribution_explicit(
    const MFFTreeFmmSolver::LinearTreeCache& cache,
    const TreeFmmConfig& config) {
  const auto device = cache.sorted_pos.device();
  const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  ExactContribution out;
  out.atom_energy = torch::zeros({cache.sorted_pos.size(0)}, float_opts);
  out.forces = torch::zeros_like(cache.sorted_pos);
  if (!cache.valid || cache.sorted_pos.numel() == 0 || !cache.leaf_offsets.defined()) return out;

  auto offsets_cpu = cache.leaf_offsets.to(torch::kCPU, torch::kInt64);
  const int64_t* offset_ptr = offsets_cpu.data_ptr<int64_t>();
  const int64_t num_leaves = std::max<int64_t>(cache.leaf_offsets.size(0) - 1, 0);

  for (int64_t i = 0; i < num_leaves; ++i) {
    const int64_t begin_i = offset_ptr[i];
    const int64_t end_i = offset_ptr[i + 1];
    const int64_t count_i = end_i - begin_i;
    if (count_i <= 0) continue;
    auto pos_i = cache.sorted_pos.narrow(0, begin_i, count_i);
    auto source_i = cache.sorted_source.narrow(0, begin_i, count_i);

    if (count_i > 1) {
      auto delta = pos_i.unsqueeze(1) - pos_i.unsqueeze(0);
      auto distance = torch::sqrt((delta * delta).sum(-1) + config.softening * config.softening);
      auto kernel = screened_coulomb_kernel(distance, config.screening);
      auto prefactor = screened_coulomb_force_prefactor(distance, config.screening);
      auto eye = torch::eye(count_i, torch::TensorOptions().dtype(torch::kBool).device(device));
      kernel.masked_fill_(eye, 0.0);
      prefactor.masked_fill_(eye, 0.0);
      auto potential = (kernel * source_i.view({1, -1})).sum(1);
      auto force_mat = source_i.view({-1, 1, 1}) * source_i.view({1, -1, 1}) * prefactor.unsqueeze(-1) * delta;
      out.atom_energy.narrow(0, begin_i, count_i).add_(0.5f * source_i * potential * config.energy_scale);
      out.forces.narrow(0, begin_i, count_i).add_(force_mat.sum(1) * config.energy_scale);
    }

    for (int64_t j = i + 1; j < num_leaves; ++j) {
      const int64_t begin_j = offset_ptr[j];
      const int64_t end_j = offset_ptr[j + 1];
      const int64_t count_j = end_j - begin_j;
      if (count_j <= 0) continue;
      auto pos_j = cache.sorted_pos.narrow(0, begin_j, count_j);
      auto source_j = cache.sorted_source.narrow(0, begin_j, count_j);
      const double dx = cache.leaf_centers.index({i, 0}).item<double>() - cache.leaf_centers.index({j, 0}).item<double>();
      const double dy = cache.leaf_centers.index({i, 1}).item<double>() - cache.leaf_centers.index({j, 1}).item<double>();
      const double dz = cache.leaf_centers.index({i, 2}).item<double>() - cache.leaf_centers.index({j, 2}).item<double>();
      const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double diameter =
          2.0 * std::max(cache.leaf_half_extents.index({i}).item<double>(), cache.leaf_half_extents.index({j}).item<double>());
      const bool exact = diameter / std::max(dist, 1.0e-9) >= config.theta;

      if (exact) {
        auto delta = pos_i.unsqueeze(1) - pos_j.unsqueeze(0);
        auto distance = torch::sqrt((delta * delta).sum(-1) + config.softening * config.softening);
        auto kernel = screened_coulomb_kernel(distance, config.screening);
        auto prefactor = screened_coulomb_force_prefactor(distance, config.screening);
        auto potential_i = (kernel * source_j.view({1, -1})).sum(1);
        auto potential_j = (kernel * source_i.view({-1, 1})).sum(0);
        auto force_mat = source_i.view({-1, 1, 1}) * source_j.view({1, -1, 1}) * prefactor.unsqueeze(-1) * delta;
        out.atom_energy.narrow(0, begin_i, count_i).add_(0.5f * source_i * potential_i * config.energy_scale);
        out.atom_energy.narrow(0, begin_j, count_j).add_(0.5f * source_j * potential_j * config.energy_scale);
        out.forces.narrow(0, begin_i, count_i).add_(force_mat.sum(1) * config.energy_scale);
        out.forces.narrow(0, begin_j, count_j).add_((-force_mat.sum(0)) * config.energy_scale);
      } else {
        auto center_j = cache.leaf_centers.index({j});
        auto charge_j = cache.leaf_charges.index({j});
        auto delta_i = pos_i - center_j.view({1, 3});
        auto distance_i = torch::sqrt((delta_i * delta_i).sum(-1) + config.softening * config.softening);
        auto kernel_i = screened_coulomb_kernel(distance_i, config.screening);
        auto prefactor_i = screened_coulomb_force_prefactor(distance_i, config.screening);
        out.atom_energy.narrow(0, begin_i, count_i).add_(0.5f * source_i * (charge_j * kernel_i) * config.energy_scale);
        out.forces.narrow(0, begin_i, count_i).add_(
            (source_i.view({-1, 1}) * charge_j.view({1, 1}) * prefactor_i.view({-1, 1}) * delta_i) * config.energy_scale);

        auto center_i = cache.leaf_centers.index({i});
        auto charge_i = cache.leaf_charges.index({i});
        auto delta_j = pos_j - center_i.view({1, 3});
        auto distance_j = torch::sqrt((delta_j * delta_j).sum(-1) + config.softening * config.softening);
        auto kernel_j = screened_coulomb_kernel(distance_j, config.screening);
        auto prefactor_j = screened_coulomb_force_prefactor(distance_j, config.screening);
        out.atom_energy.narrow(0, begin_j, count_j).add_(0.5f * source_j * (charge_i * kernel_j) * config.energy_scale);
        out.forces.narrow(0, begin_j, count_j).add_(
            (source_j.view({-1, 1}) * charge_i.view({1, 1}) * prefactor_j.view({-1, 1}) * delta_j) * config.energy_scale);
      }
    }
  }

  auto atom_energy = torch::zeros_like(out.atom_energy);
  atom_energy.index_copy_(0, cache.permutation, out.atom_energy);
  auto forces = torch::zeros_like(out.forces);
  forces.index_copy_(0, cache.permutation, out.forces);
  out.atom_energy = atom_energy;
  out.forces = forces;
  return out;
}

RemoteLeafPayload build_local_leaf_payload(
    const MFFTreeFmmSolver::LinearTreeCache& cache,
    int world_rank) {
  const int64_t n = cache.leaf_ids.defined() ? cache.leaf_ids.size(0) : 0;
  const auto device = cache.leaf_ids.defined() ? cache.leaf_ids.device() : torch::Device(torch::kCPU);
  RemoteLeafPayload payload = empty_remote_leaf_payload(device);
  if (n == 0) return payload;
  payload.ranks = torch::full({n}, static_cast<int64_t>(world_rank), torch::TensorOptions().dtype(torch::kInt64).device(device));
  payload.leaf_ids = cache.leaf_ids;
  payload.charges = cache.leaf_charges;
  payload.half_extents = cache.leaf_half_extents;
  payload.centers = cache.leaf_centers;
  return payload;
}

std::vector<RankClusterSummary> local_summaries_from_linear_cache(
    const MFFTreeFmmSolver::LinearTreeCache& cache,
    int world_rank) {
  std::vector<RankClusterSummary> out;
  if (!cache.valid || !cache.leaf_offsets.defined()) return out;
  auto offsets_cpu = cache.leaf_offsets.to(torch::kCPU, torch::kInt64);
  auto perm_cpu = cache.permutation.to(torch::kCPU, torch::kInt64);
  const int64_t* offsets = offsets_cpu.data_ptr<int64_t>();
  const int64_t* perm = perm_cpu.data_ptr<int64_t>();
  const int64_t num_leaves = std::max<int64_t>(cache.leaf_offsets.size(0) - 1, 0);
  out.reserve(static_cast<size_t>(num_leaves));
  for (int64_t leaf = 0; leaf < num_leaves; ++leaf) {
    RankClusterSummary summary;
    summary.rank = world_rank;
    summary.leaf_id = static_cast<int>(leaf);
    summary.natoms = cache.leaf_counts.index({leaf}).item<int64_t>();
    summary.charge = cache.leaf_charges.index({leaf}).item<double>();
    summary.half_extent = cache.leaf_half_extents.index({leaf}).item<double>();
    for (int axis = 0; axis < 3; ++axis) {
      summary.center[axis] = cache.leaf_centers.index({leaf, axis}).item<double>();
      summary.bbox_min[axis] = cache.leaf_bbox_min.index({leaf, axis}).item<double>();
      summary.bbox_max[axis] = cache.leaf_bbox_max.index({leaf, axis}).item<double>();
    }
    for (int64_t idx = offsets[leaf]; idx < offsets[leaf + 1]; ++idx) {
      summary.indices.push_back(static_cast<int>(perm[idx]));
    }
    out.push_back(std::move(summary));
  }
  return out;
}

RemoteLeafPayload gather_remote_leaf_payload(
    MPI_Comm world,
    const RemoteLeafPayload& local_payload,
    int world_size,
    int64_t local_geometry_version,
    MFFTreeFmmSolver::DistributedRemoteGeometryCache& geometry_cache,
    const torch::Device& preferred_device,
    bool gpu_aware_mpi,
    DistributedCommStats* stats) {
  int world_rank = 0;
  MPI_Comm_rank(world, &world_rank);

  const int64_t local_leaf_count = local_payload.size();
  std::vector<int64_t> leaf_counts(static_cast<size_t>(world_size), 0);
  std::vector<int64_t> geometry_versions(static_cast<size_t>(world_size), 0);
  MPI_Allgather(&local_leaf_count, 1, MPI_LONG_LONG, leaf_counts.data(), 1, MPI_LONG_LONG, world);
  MPI_Allgather(&local_geometry_version, 1, MPI_LONG_LONG, geometry_versions.data(), 1, MPI_LONG_LONG, world);

  std::vector<int> charge_recvcounts(static_cast<size_t>(world_size), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    charge_recvcounts[static_cast<size_t>(rank)] = static_cast<int>(leaf_counts[static_cast<size_t>(rank)]);
  }
  auto charge_recvdispls = build_displs(charge_recvcounts);
  const int total_leafs =
      std::accumulate(charge_recvcounts.begin(), charge_recvcounts.end(), 0);
  if (stats != nullptr) {
    stats->total_leafs = total_leafs;
    stats->charge_leafs = total_leafs;
  }
  const bool use_gpu = gpu_aware_mpi && preferred_device.is_cuda();

  torch::Tensor gathered_charges;
  if (use_gpu) {
    auto charge_send = local_payload.charges.contiguous();
    auto charge_recv = torch::empty(
        {std::max(total_leafs, 0)},
        torch::TensorOptions().dtype(torch::kFloat32).device(preferred_device));
    MPI_Allgatherv(
        charge_send.numel() > 0 ? charge_send.data_ptr<float>() : nullptr,
        static_cast<int>(local_leaf_count),
        MPI_FLOAT,
        charge_recv.numel() > 0 ? charge_recv.data_ptr<float>() : nullptr,
        charge_recvcounts.data(),
        charge_recvdispls.data(),
        MPI_FLOAT,
        world);
    gathered_charges = charge_recv;
  } else {
    auto charge_send_cpu = local_payload.charges.to(torch::kCPU, torch::kFloat32).contiguous();
    auto charge_recv_cpu =
        torch::empty({std::max(total_leafs, 0)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    MPI_Allgatherv(
        charge_send_cpu.numel() > 0 ? charge_send_cpu.data_ptr<float>() : nullptr,
        static_cast<int>(local_leaf_count),
        MPI_FLOAT,
        charge_recv_cpu.numel() > 0 ? charge_recv_cpu.data_ptr<float>() : nullptr,
        charge_recvcounts.data(),
        charge_recvdispls.data(),
        MPI_FLOAT,
        world);
    gathered_charges = charge_recv_cpu.to(preferred_device);
  }

  const auto charges_by_rank = split_flat_tensor_by_counts(gathered_charges, leaf_counts);

  bool cache_device_mismatch = false;
  for (size_t rank = 0; rank < geometry_cache.leaf_ids_by_rank.size(); ++rank) {
    if (geometry_cache.leaf_ids_by_rank[rank].defined() &&
        geometry_cache.leaf_ids_by_rank[rank].device() != preferred_device) {
      cache_device_mismatch = true;
      break;
    }
  }

  const bool cache_reset =
      !geometry_cache.valid || geometry_cache.world_size != world_size ||
      geometry_cache.leaf_counts.size() != static_cast<size_t>(world_size) ||
      geometry_cache.geometry_versions.size() != static_cast<size_t>(world_size) ||
      geometry_cache.leaf_ids_by_rank.size() != static_cast<size_t>(world_size) ||
      geometry_cache.half_extents_by_rank.size() != static_cast<size_t>(world_size) ||
      geometry_cache.centers_by_rank.size() != static_cast<size_t>(world_size) || cache_device_mismatch;
  if (cache_reset) {
    geometry_cache = MFFTreeFmmSolver::DistributedRemoteGeometryCache{};
    geometry_cache.world_size = world_size;
    geometry_cache.leaf_counts.assign(static_cast<size_t>(world_size), 0);
    geometry_cache.geometry_versions.assign(static_cast<size_t>(world_size), -1);
    geometry_cache.leaf_ids_by_rank.assign(static_cast<size_t>(world_size), torch::Tensor());
    geometry_cache.half_extents_by_rank.assign(static_cast<size_t>(world_size), torch::Tensor());
    geometry_cache.centers_by_rank.assign(static_cast<size_t>(world_size), torch::Tensor());
  }

  std::vector<int> geometry_id_recvcounts(static_cast<size_t>(world_size), 0);
  std::vector<int> geometry_float_recvcounts(static_cast<size_t>(world_size), 0);
  std::vector<char> refresh_geometry(static_cast<size_t>(world_size), 0);
  bool any_geometry_refresh = false;
  for (int rank = 0; rank < world_size; ++rank) {
    const bool needs_refresh =
        cache_reset ||
        geometry_cache.geometry_versions[static_cast<size_t>(rank)] != geometry_versions[static_cast<size_t>(rank)] ||
        geometry_cache.leaf_counts[static_cast<size_t>(rank)] != leaf_counts[static_cast<size_t>(rank)] ||
        !geometry_cache.leaf_ids_by_rank[static_cast<size_t>(rank)].defined();
    refresh_geometry[static_cast<size_t>(rank)] = needs_refresh ? 1 : 0;
    if (!needs_refresh) continue;
    any_geometry_refresh = true;
    geometry_id_recvcounts[static_cast<size_t>(rank)] = static_cast<int>(leaf_counts[static_cast<size_t>(rank)]);
    geometry_float_recvcounts[static_cast<size_t>(rank)] = static_cast<int>(leaf_counts[static_cast<size_t>(rank)] * 4);
  }

  if (any_geometry_refresh) {
    auto geometry_id_recvdispls = build_displs(geometry_id_recvcounts);
    auto geometry_float_recvdispls = build_displs(geometry_float_recvcounts);
    const int total_geometry_leafs =
        std::accumulate(geometry_id_recvcounts.begin(), geometry_id_recvcounts.end(), 0);
    if (stats != nullptr) stats->geometry_refreshed_leafs = total_geometry_leafs;
    const bool local_needs_refresh = refresh_geometry[static_cast<size_t>(world_rank)] != 0;

    torch::Tensor recv_leaf_ids;
    torch::Tensor recv_geometry;
    if (use_gpu) {
      auto leaf_id_send = local_needs_refresh ? local_payload.leaf_ids.contiguous()
                                              : torch::empty(
                                                    {0},
                                                    torch::TensorOptions().dtype(torch::kInt64).device(preferred_device));
      auto geometry_send = local_needs_refresh
                               ? torch::cat({local_payload.half_extents.view({-1, 1}), local_payload.centers}, 1).contiguous()
                               : torch::empty(
                                     {0, 4},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(preferred_device));
      recv_leaf_ids = torch::empty(
          {std::max(total_geometry_leafs, 0)},
          torch::TensorOptions().dtype(torch::kInt64).device(preferred_device));
      recv_geometry = torch::empty(
          {std::max(total_geometry_leafs, 0), 4},
          torch::TensorOptions().dtype(torch::kFloat32).device(preferred_device));
      MPI_Allgatherv(
          leaf_id_send.numel() > 0 ? leaf_id_send.data_ptr<int64_t>() : nullptr,
          local_needs_refresh ? static_cast<int>(local_leaf_count) : 0,
          MPI_LONG_LONG,
          recv_leaf_ids.numel() > 0 ? recv_leaf_ids.data_ptr<int64_t>() : nullptr,
          geometry_id_recvcounts.data(),
          geometry_id_recvdispls.data(),
          MPI_LONG_LONG,
          world);
      MPI_Allgatherv(
          geometry_send.numel() > 0 ? geometry_send.data_ptr<float>() : nullptr,
          local_needs_refresh ? static_cast<int>(local_leaf_count * 4) : 0,
          MPI_FLOAT,
          recv_geometry.numel() > 0 ? recv_geometry.data_ptr<float>() : nullptr,
          geometry_float_recvcounts.data(),
          geometry_float_recvdispls.data(),
          MPI_FLOAT,
          world);
    } else {
      auto leaf_id_send_cpu = local_needs_refresh ? local_payload.leaf_ids.to(torch::kCPU, torch::kInt64).contiguous()
                                                  : torch::empty(
                                                        {0},
                                                        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
      auto geometry_send_cpu =
          local_needs_refresh
              ? torch::cat({local_payload.half_extents.view({-1, 1}), local_payload.centers}, 1).to(torch::kCPU, torch::kFloat32).contiguous()
              : torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      auto recv_leaf_ids_cpu = torch::empty(
          {std::max(total_geometry_leafs, 0)},
          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
      auto recv_geometry_cpu = torch::empty(
          {std::max(total_geometry_leafs, 0), 4},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      MPI_Allgatherv(
          leaf_id_send_cpu.numel() > 0 ? leaf_id_send_cpu.data_ptr<int64_t>() : nullptr,
          local_needs_refresh ? static_cast<int>(local_leaf_count) : 0,
          MPI_LONG_LONG,
          recv_leaf_ids_cpu.numel() > 0 ? recv_leaf_ids_cpu.data_ptr<int64_t>() : nullptr,
          geometry_id_recvcounts.data(),
          geometry_id_recvdispls.data(),
          MPI_LONG_LONG,
          world);
      MPI_Allgatherv(
          geometry_send_cpu.numel() > 0 ? geometry_send_cpu.data_ptr<float>() : nullptr,
          local_needs_refresh ? static_cast<int>(local_leaf_count * 4) : 0,
          MPI_FLOAT,
          recv_geometry_cpu.numel() > 0 ? recv_geometry_cpu.data_ptr<float>() : nullptr,
          geometry_float_recvcounts.data(),
          geometry_float_recvdispls.data(),
          MPI_FLOAT,
          world);
      recv_leaf_ids = recv_leaf_ids_cpu.to(preferred_device);
      recv_geometry = recv_geometry_cpu.to(preferred_device);
    }

    for (int rank = 0; rank < world_size; ++rank) {
      if (!refresh_geometry[static_cast<size_t>(rank)]) continue;
      const int64_t count = leaf_counts[static_cast<size_t>(rank)];
      if (count <= 0) {
        geometry_cache.leaf_ids_by_rank[static_cast<size_t>(rank)] =
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(preferred_device));
        geometry_cache.half_extents_by_rank[static_cast<size_t>(rank)] =
            torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(preferred_device));
        geometry_cache.centers_by_rank[static_cast<size_t>(rank)] =
            torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(preferred_device));
        continue;
      }
      const int row_offset = geometry_id_recvdispls[static_cast<size_t>(rank)];
      geometry_cache.leaf_ids_by_rank[static_cast<size_t>(rank)] =
          recv_leaf_ids.narrow(0, row_offset, count).contiguous();
      auto geometry_rows = recv_geometry.narrow(0, row_offset, count).contiguous();
      geometry_cache.half_extents_by_rank[static_cast<size_t>(rank)] = geometry_rows.select(1, 0).contiguous();
      geometry_cache.centers_by_rank[static_cast<size_t>(rank)] = geometry_rows.slice(1, 1, 4).contiguous();
    }
  }

  geometry_cache.world_size = world_size;
  geometry_cache.valid = true;
  geometry_cache.leaf_counts = leaf_counts;
  geometry_cache.geometry_versions = geometry_versions;
  return assemble_remote_leaf_payload(geometry_cache, charges_by_rank, preferred_device);
}

RemoteLeafPayload payload_from_summaries(
    const std::vector<RankClusterSummary>& summaries,
    const torch::Device& device) {
  RemoteLeafPayload payload = empty_remote_leaf_payload(device);
  const int64_t n = static_cast<int64_t>(summaries.size());
  if (n == 0) return payload;

  std::vector<int64_t> ranks(n), leaf_ids(n);
  std::vector<float> charges(n), half_extents(n), centers(static_cast<size_t>(n) * 3);
  for (int64_t i = 0; i < n; ++i) {
    const auto& s = summaries[static_cast<size_t>(i)];
    ranks[static_cast<size_t>(i)] = s.rank;
    leaf_ids[static_cast<size_t>(i)] = s.leaf_id;
    charges[static_cast<size_t>(i)] = static_cast<float>(s.charge);
    half_extents[static_cast<size_t>(i)] = static_cast<float>(s.half_extent);
    for (int axis = 0; axis < 3; ++axis) {
      centers[static_cast<size_t>(i) * 3 + axis] = static_cast<float>(s.center[axis]);
    }
  }
  payload.ranks = cpu_long_tensor_from_buffer(ranks, {n}).to(device);
  payload.leaf_ids = cpu_long_tensor_from_buffer(leaf_ids, {n}).to(device);
  payload.charges = cpu_float_tensor_from_buffer(charges, {n}).to(device);
  payload.half_extents = cpu_float_tensor_from_buffer(half_extents, {n}).to(device);
  payload.centers = cpu_float_tensor_from_buffer(centers, {n, 3}).to(device);
  return payload;
}

RemoteLeafPayload index_payload(const RemoteLeafPayload& payload, const torch::Tensor& index) {
  RemoteLeafPayload out = empty_remote_leaf_payload(payload.leaf_ids.defined() ? payload.leaf_ids.device() : torch::Device(torch::kCPU));
  out.ranks = payload.ranks.index_select(0, index);
  out.leaf_ids = payload.leaf_ids.index_select(0, index);
  out.charges = payload.charges.index_select(0, index);
  out.half_extents = payload.half_extents.index_select(0, index);
  out.centers = payload.centers.index_select(0, index);
  return out;
}

DeviceRequestPlan classify_remote_payload(
    const MFFTreeFmmSolver::LinearTreeCache& cache,
    const RemoteLeafPayload& remote_payload,
    int world_rank,
    int world_size,
    double theta,
    RemoteLeafPayload& far_payload) {
  const auto device = remote_payload.leaf_ids.defined() ? remote_payload.leaf_ids.device() : cache.sorted_pos.device();
  auto plan = empty_device_request_plan(device, world_size);
  far_payload = empty_remote_leaf_payload(device);
  if (remote_payload.size() == 0) return plan;

  auto remote_mask = remote_payload.ranks.ne(world_rank);
  auto remote_full_idx = torch::nonzero(remote_mask).view({-1});
  if (remote_full_idx.numel() == 0) {
    far_payload = index_payload(remote_payload, remote_full_idx);
    plan.far_indices = remote_full_idx;
    return plan;
  }
  auto remote_only = index_payload(remote_payload, remote_full_idx);
  if (!cache.leaf_centers.defined() || cache.leaf_centers.numel() == 0) {
    far_payload = remote_only;
    plan.far_indices = remote_full_idx;
    return plan;
  }

  auto delta = cache.leaf_centers.unsqueeze(1) - remote_only.centers.unsqueeze(0);
  auto dist = torch::sqrt((delta * delta).sum(-1) + 1.0e-18);
  auto diameter = 2.0f * torch::maximum(
                              cache.leaf_half_extents.view({-1, 1}),
                              remote_only.half_extents.view({1, -1}));
  auto exact = (diameter / dist.clamp_min(1.0e-9)) >= theta;
  auto remote_exact = exact.any(0);
  auto exact_idx = torch::nonzero(remote_exact).view({-1});
  auto far_idx = torch::nonzero(remote_exact.logical_not()).view({-1});
  plan.far_indices = remote_full_idx.index_select(0, far_idx).contiguous();
  far_payload = index_payload(remote_payload, plan.far_indices);

  if (exact_idx.numel() == 0) return plan;

  auto exact_full_idx = remote_full_idx.index_select(0, exact_idx);
  auto exact_ranks = remote_payload.ranks.index_select(0, exact_full_idx).contiguous();
  auto exact_leaf_ids = remote_payload.leaf_ids.index_select(0, exact_full_idx).contiguous();
  auto sort_order = std::get<1>(exact_ranks.sort());
  plan.request_ranks = exact_ranks.index_select(0, sort_order).contiguous();
  plan.request_leaf_ids = exact_leaf_ids.index_select(0, sort_order).contiguous();

  auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
  plan.request_counts_per_rank = torch::zeros({world_size}, long_opts);
  auto ones = torch::ones({plan.request_ranks.numel()}, long_opts);
  plan.request_counts_per_rank.index_add_(0, plan.request_ranks, ones);
  plan.request_rank_offsets = torch::cumsum(plan.request_counts_per_rank, 0) - plan.request_counts_per_rank;
  return plan;
}

std::pair<torch::Tensor, torch::Tensor> pack_requested_leaf_atoms_device(
    const MFFTreeFmmSolver::LinearTreeCache& cache,
    const torch::Tensor& requested_leaf_ids) {
  const auto device = cache.sorted_pos.device();
  if (!requested_leaf_ids.defined() || requested_leaf_ids.numel() == 0) {
    return {
        torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
        torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
    };
  }

  auto leaf_ids = requested_leaf_ids.to(device, torch::kInt64).contiguous();
  auto leaf_begin = cache.leaf_offsets.index_select(0, leaf_ids);
  auto leaf_end = cache.leaf_offsets.index_select(0, leaf_ids + 1);
  auto leaf_counts = (leaf_end - leaf_begin).contiguous();
  const int64_t total_atoms = leaf_counts.numel() > 0 ? leaf_counts.sum().item<int64_t>() : 0;
  if (total_atoms <= 0) {
    return {
        torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
        torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
    };
  }

  auto request_offsets = torch::cumsum(leaf_counts, 0) - leaf_counts;
  auto repeated_requests = torch::repeat_interleave(
      torch::arange(leaf_counts.size(0), torch::TensorOptions().dtype(torch::kInt64).device(device)),
      leaf_counts);
  auto within_leaf_offsets =
      torch::arange(total_atoms, torch::TensorOptions().dtype(torch::kInt64).device(device)) -
      request_offsets.index_select(0, repeated_requests);
  auto gather_idx = leaf_begin.index_select(0, repeated_requests) + within_leaf_offsets;
  return {
      cache.sorted_pos.index_select(0, gather_idx).contiguous(),
      cache.sorted_source.index_select(0, gather_idx).contiguous(),
  };
}

ImportedAtoms exchange_requested_linear_leaf_atoms(
    MPI_Comm world,
    const DeviceRequestPlan& request_plan,
    const MFFTreeFmmSolver::LinearTreeCache& cache,
    const torch::Device& preferred_device,
    bool gpu_aware_mpi) {
  int world_size = 1;
  MPI_Comm_size(world, &world_size);

  auto request_sendcounts = tensor_to_int_vector(request_plan.request_counts_per_rank);
  if (request_sendcounts.size() != static_cast<size_t>(world_size)) {
    request_sendcounts.assign(static_cast<size_t>(world_size), 0);
  }
  std::vector<int> request_recvcounts(static_cast<size_t>(world_size), 0);
  MPI_Alltoall(request_sendcounts.data(), 1, MPI_INT, request_recvcounts.data(), 1, MPI_INT, world);
  auto request_senddispls = tensor_to_int_vector(request_plan.request_rank_offsets);
  if (request_senddispls.size() != static_cast<size_t>(world_size)) {
    request_senddispls = build_displs(request_sendcounts);
  }
  auto request_recvdispls = build_displs(request_recvcounts);
  const int total_request_send =
      request_senddispls.empty() ? 0 : request_senddispls.back() + request_sendcounts.back();
  const int total_request_recv =
      request_recvdispls.empty() ? 0 : request_recvdispls.back() + request_recvcounts.back();

  const bool use_gpu_exchange = gpu_aware_mpi && preferred_device.is_cuda();
  torch::Tensor request_recv_ids_dev;
  if (use_gpu_exchange) {
    auto request_send_ids = request_plan.request_leaf_ids.to(preferred_device, torch::kInt32).contiguous();
    request_recv_ids_dev = torch::empty(
        {std::max(total_request_recv, 0)},
        torch::TensorOptions().dtype(torch::kInt32).device(preferred_device));
    MPI_Alltoallv(
        request_send_ids.numel() > 0 ? request_send_ids.data_ptr<int>() : nullptr,
        request_sendcounts.data(),
        request_senddispls.data(),
        MPI_INT,
        request_recv_ids_dev.numel() > 0 ? request_recv_ids_dev.data_ptr<int>() : nullptr,
        request_recvcounts.data(),
        request_recvdispls.data(),
        MPI_INT,
        world);
  } else {
    auto request_send_ids_cpu = request_plan.request_leaf_ids.to(torch::kCPU, torch::kInt32).contiguous();
    std::vector<int> request_recvbuf(static_cast<size_t>(std::max(total_request_recv, 0)), 0);
    MPI_Alltoallv(
        request_send_ids_cpu.numel() > 0 ? request_send_ids_cpu.data_ptr<int>() : nullptr,
        request_sendcounts.data(),
        request_senddispls.data(),
        MPI_INT,
        request_recvbuf.empty() ? nullptr : request_recvbuf.data(),
        request_recvcounts.data(),
        request_recvdispls.data(),
        MPI_INT,
        world);
    request_recv_ids_dev =
        cpu_long_tensor_from_buffer(
            std::vector<int64_t>(request_recvbuf.begin(), request_recvbuf.end()),
            {static_cast<int64_t>(total_request_recv)})
            .to(preferred_device);
  }

  auto request_recv_ids_long = request_recv_ids_dev.to(preferred_device, torch::kInt64).contiguous();
  auto request_recv_leaf_counts =
      request_recv_ids_long.numel() > 0 ? cache.leaf_counts.index_select(0, request_recv_ids_long).to(torch::kCPU, torch::kInt64).contiguous()
                                        : torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  const int64_t* request_recv_leaf_count_ptr =
      request_recv_leaf_counts.numel() > 0 ? request_recv_leaf_counts.data_ptr<int64_t>() : nullptr;

  std::vector<int> atom_sendcounts(static_cast<size_t>(world_size), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    const int begin = request_recvdispls[static_cast<size_t>(rank)];
    const int end = begin + request_recvcounts[static_cast<size_t>(rank)];
    int atom_count = 0;
    for (int idx = begin; idx < end; ++idx) {
      atom_count += static_cast<int>(request_recv_leaf_count_ptr[idx]);
    }
    atom_sendcounts[static_cast<size_t>(rank)] = atom_count;
  }
  std::vector<int> atom_recvcounts(static_cast<size_t>(world_size), 0);
  MPI_Alltoall(atom_sendcounts.data(), 1, MPI_INT, atom_recvcounts.data(), 1, MPI_INT, world);

  std::vector<int> pos_sendcounts(static_cast<size_t>(world_size), 0);
  std::vector<int> pos_recvcounts(static_cast<size_t>(world_size), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    pos_sendcounts[static_cast<size_t>(rank)] = atom_sendcounts[static_cast<size_t>(rank)] * 3;
    pos_recvcounts[static_cast<size_t>(rank)] = atom_recvcounts[static_cast<size_t>(rank)] * 3;
  }
  auto atom_senddispls = build_displs(atom_sendcounts);
  auto atom_recvdispls = build_displs(atom_recvcounts);
  auto pos_senddispls = build_displs(pos_sendcounts);
  auto pos_recvdispls = build_displs(pos_recvcounts);

  const int total_atom_send = atom_senddispls.empty() ? 0 : atom_senddispls.back() + atom_sendcounts.back();
  const int total_atom_recv = atom_recvdispls.empty() ? 0 : atom_recvdispls.back() + atom_recvcounts.back();
  auto packed = pack_requested_leaf_atoms_device(cache, request_recv_ids_long);
  auto send_pos = packed.first;
  auto send_source = packed.second;

  if (use_gpu_exchange) {
    auto recv_pos = torch::empty(
        {static_cast<int64_t>(total_atom_recv), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(preferred_device));
    auto recv_source = torch::empty(
        {static_cast<int64_t>(total_atom_recv)},
        torch::TensorOptions().dtype(torch::kFloat32).device(preferred_device));
    MPI_Alltoallv(
        send_pos.numel() > 0 ? send_pos.data_ptr<float>() : nullptr,
        pos_sendcounts.data(),
        pos_senddispls.data(),
        MPI_FLOAT,
        recv_pos.numel() > 0 ? recv_pos.data_ptr<float>() : nullptr,
        pos_recvcounts.data(),
        pos_recvdispls.data(),
        MPI_FLOAT,
        world);
    MPI_Alltoallv(
        send_source.numel() > 0 ? send_source.data_ptr<float>() : nullptr,
        atom_sendcounts.data(),
        atom_senddispls.data(),
        MPI_FLOAT,
        recv_source.numel() > 0 ? recv_source.data_ptr<float>() : nullptr,
        atom_recvcounts.data(),
        atom_recvdispls.data(),
        MPI_FLOAT,
        world);
    ImportedAtoms imported;
    imported.pos = recv_pos;
    imported.source = recv_source;
    return imported;
  }

  auto send_pos_cpu = send_pos.to(torch::kCPU, torch::kFloat32).contiguous();
  auto send_source_cpu = send_source.to(torch::kCPU, torch::kFloat32).contiguous();
  std::vector<float> recv_pos_buf(static_cast<size_t>(std::max(total_atom_recv * 3, 0)), 0.0f);
  std::vector<float> recv_source_buf(static_cast<size_t>(std::max(total_atom_recv, 0)), 0.0f);
  MPI_Alltoallv(
      send_pos_cpu.numel() > 0 ? send_pos_cpu.data_ptr<float>() : nullptr,
      pos_sendcounts.data(),
      pos_senddispls.data(),
      MPI_FLOAT,
      recv_pos_buf.empty() ? nullptr : recv_pos_buf.data(),
      pos_recvcounts.data(),
      pos_recvdispls.data(),
      MPI_FLOAT,
      world);
  MPI_Alltoallv(
      send_source_cpu.numel() > 0 ? send_source_cpu.data_ptr<float>() : nullptr,
      atom_sendcounts.data(),
      atom_senddispls.data(),
      MPI_FLOAT,
      recv_source_buf.empty() ? nullptr : recv_source_buf.data(),
      atom_recvcounts.data(),
      atom_recvdispls.data(),
      MPI_FLOAT,
      world);
  ImportedAtoms imported;
  imported.pos = cpu_float_tensor_from_buffer(recv_pos_buf, {static_cast<int64_t>(total_atom_recv), 3}).to(preferred_device);
  imported.source = cpu_float_tensor_from_buffer(recv_source_buf, {static_cast<int64_t>(total_atom_recv)}).to(preferred_device);
  return imported;
}

RankClusterSummary build_local_summary(
    int world_rank,
    const torch::Tensor& pos,
    const torch::Tensor& source) {
  RankClusterSummary summary;
  summary.rank = world_rank;
  summary.natoms = pos.defined() ? pos.size(0) : 0;
  if (summary.natoms <= 0) return summary;
  summary.charge = source.sum().item<double>();
  auto mean_pos = pos.mean(0);
  auto coord_min = std::get<0>(pos.min(0));
  auto coord_max = std::get<0>(pos.max(0));
  for (int axis = 0; axis < 3; ++axis) {
    summary.center[axis] = mean_pos.index({axis}).item<double>();
    summary.bbox_min[axis] = coord_min.index({axis}).item<double>();
    summary.bbox_max[axis] = coord_max.index({axis}).item<double>();
  }
  summary.half_extent = 0.5 * (coord_max - coord_min).max().item<double>();
  return summary;
}

std::vector<RankClusterSummary> gather_rank_summaries(
    MPI_Comm world,
    const RankClusterSummary& local_summary,
    int world_size) {
  std::vector<int64_t> counts(static_cast<size_t>(world_size), 0);
  int64_t local_natoms = local_summary.natoms;
  MPI_Allgather(&local_natoms, 1, MPI_LONG_LONG, counts.data(), 1, MPI_LONG_LONG, world);

  std::array<double, 11> local_payload{
      local_summary.charge,
      local_summary.half_extent,
      local_summary.center[0],
      local_summary.center[1],
      local_summary.center[2],
      local_summary.bbox_min[0],
      local_summary.bbox_min[1],
      local_summary.bbox_min[2],
      local_summary.bbox_max[0],
      local_summary.bbox_max[1],
      local_summary.bbox_max[2],
  };
  std::vector<double> gathered(static_cast<size_t>(world_size) * local_payload.size(), 0.0);
  MPI_Allgather(
      local_payload.data(),
      static_cast<int>(local_payload.size()),
      MPI_DOUBLE,
      gathered.data(),
      static_cast<int>(local_payload.size()),
      MPI_DOUBLE,
      world);

  std::vector<RankClusterSummary> summaries(static_cast<size_t>(world_size));
  for (int rank = 0; rank < world_size; ++rank) {
    RankClusterSummary summary;
    summary.rank = rank;
    summary.natoms = counts[static_cast<size_t>(rank)];
    const size_t base = static_cast<size_t>(rank) * local_payload.size();
    summary.charge = gathered[base + 0];
    summary.half_extent = gathered[base + 1];
    summary.center = {gathered[base + 2], gathered[base + 3], gathered[base + 4]};
    summary.bbox_min = {gathered[base + 5], gathered[base + 6], gathered[base + 7]};
    summary.bbox_max = {gathered[base + 8], gathered[base + 9], gathered[base + 10]};
    summaries[static_cast<size_t>(rank)] = summary;
  }
  return summaries;
}

bool rank_requires_exact_exchange(
    const RankClusterSummary& local_summary,
    const RankClusterSummary& remote_summary,
    double theta) {
  if (!local_summary.has_atoms() || !remote_summary.has_atoms()) return false;
  const double dx = local_summary.center[0] - remote_summary.center[0];
  const double dy = local_summary.center[1] - remote_summary.center[1];
  const double dz = local_summary.center[2] - remote_summary.center[2];
  const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double diameter = 2.0 * std::max(local_summary.half_extent, remote_summary.half_extent);
  return diameter / std::max(dist, 1.0e-9) >= theta;
}

std::vector<RankClusterSummary> gather_leaf_summaries(
    MPI_Comm world,
    const std::vector<RankClusterSummary>& local_summaries,
    int world_size) {
  constexpr int payload_size = 13;
  std::vector<int> leaf_counts(static_cast<size_t>(world_size), 0);
  const int local_leaf_count = static_cast<int>(local_summaries.size());
  MPI_Allgather(&local_leaf_count, 1, MPI_INT, leaf_counts.data(), 1, MPI_INT, world);

  std::vector<int> recv_counts(static_cast<size_t>(world_size), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    recv_counts[static_cast<size_t>(rank)] = leaf_counts[static_cast<size_t>(rank)] * payload_size;
  }
  auto recv_displs = build_displs(recv_counts);
  const int total_recv = recv_displs.empty() ? 0 : recv_displs.back() + recv_counts.back();

  std::vector<double> send_buf(static_cast<size_t>(local_leaf_count * payload_size), 0.0);
  for (int i = 0; i < local_leaf_count; ++i) {
    const auto& leaf = local_summaries[static_cast<size_t>(i)];
    const int base = i * payload_size;
    send_buf[base + 0] = static_cast<double>(leaf.leaf_id);
    send_buf[base + 1] = static_cast<double>(leaf.natoms);
    send_buf[base + 2] = leaf.charge;
    send_buf[base + 3] = leaf.half_extent;
    send_buf[base + 4] = leaf.center[0];
    send_buf[base + 5] = leaf.center[1];
    send_buf[base + 6] = leaf.center[2];
    send_buf[base + 7] = leaf.bbox_min[0];
    send_buf[base + 8] = leaf.bbox_min[1];
    send_buf[base + 9] = leaf.bbox_min[2];
    send_buf[base + 10] = leaf.bbox_max[0];
    send_buf[base + 11] = leaf.bbox_max[1];
    send_buf[base + 12] = leaf.bbox_max[2];
  }

  std::vector<double> recv_buf(static_cast<size_t>(std::max(total_recv, 0)), 0.0);
  MPI_Allgatherv(
      send_buf.empty() ? nullptr : send_buf.data(),
      local_leaf_count * payload_size,
      MPI_DOUBLE,
      recv_buf.empty() ? nullptr : recv_buf.data(),
      recv_counts.data(),
      recv_displs.data(),
      MPI_DOUBLE,
      world);

  std::vector<RankClusterSummary> out;
  out.reserve(static_cast<size_t>(total_recv / payload_size));
  for (int rank = 0; rank < world_size; ++rank) {
    const int leaf_count = leaf_counts[static_cast<size_t>(rank)];
    const int base = recv_displs[static_cast<size_t>(rank)];
    for (int i = 0; i < leaf_count; ++i) {
      const int offset = base + i * payload_size;
      RankClusterSummary leaf;
      leaf.rank = rank;
      leaf.leaf_id = static_cast<int>(std::llround(recv_buf[static_cast<size_t>(offset + 0)]));
      leaf.natoms = static_cast<int64_t>(std::llround(recv_buf[static_cast<size_t>(offset + 1)]));
      leaf.charge = recv_buf[static_cast<size_t>(offset + 2)];
      leaf.half_extent = recv_buf[static_cast<size_t>(offset + 3)];
      leaf.center = {
          recv_buf[static_cast<size_t>(offset + 4)],
          recv_buf[static_cast<size_t>(offset + 5)],
          recv_buf[static_cast<size_t>(offset + 6)],
      };
      leaf.bbox_min = {
          recv_buf[static_cast<size_t>(offset + 7)],
          recv_buf[static_cast<size_t>(offset + 8)],
          recv_buf[static_cast<size_t>(offset + 9)],
      };
      leaf.bbox_max = {
          recv_buf[static_cast<size_t>(offset + 10)],
          recv_buf[static_cast<size_t>(offset + 11)],
          recv_buf[static_cast<size_t>(offset + 12)],
      };
      out.push_back(leaf);
    }
  }
  return out;
}

bool leaf_requires_exact_exchange(
    const RankClusterSummary& local_leaf,
    const RankClusterSummary& remote_leaf,
    double theta) {
  if (!local_leaf.has_atoms() || !remote_leaf.has_atoms()) return false;
  const double dx = local_leaf.center[0] - remote_leaf.center[0];
  const double dy = local_leaf.center[1] - remote_leaf.center[1];
  const double dz = local_leaf.center[2] - remote_leaf.center[2];
  const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double diameter = 2.0 * std::max(local_leaf.half_extent, remote_leaf.half_extent);
  return diameter / std::max(dist, 1.0e-9) >= theta;
}

ImportedAtoms exchange_requested_leaf_atoms(
    MPI_Comm world,
    const std::vector<std::vector<int>>& requested_leaf_ids_by_rank,
    const std::vector<RankClusterSummary>& local_leaf_summaries,
    const torch::Tensor& local_pos,
    const torch::Tensor& local_source,
    const torch::Device& preferred_device,
    bool gpu_aware_mpi) {
  int world_size = 1;
  MPI_Comm_size(world, &world_size);

  std::vector<int> request_sendcounts(static_cast<size_t>(world_size), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    request_sendcounts[static_cast<size_t>(rank)] = static_cast<int>(requested_leaf_ids_by_rank[static_cast<size_t>(rank)].size());
  }
  std::vector<int> request_recvcounts(static_cast<size_t>(world_size), 0);
  MPI_Alltoall(request_sendcounts.data(), 1, MPI_INT, request_recvcounts.data(), 1, MPI_INT, world);
  auto request_senddispls = build_displs(request_sendcounts);
  auto request_recvdispls = build_displs(request_recvcounts);

  const int total_request_send =
      request_senddispls.empty() ? 0 : request_senddispls.back() + request_sendcounts.back();
  const int total_request_recv =
      request_recvdispls.empty() ? 0 : request_recvdispls.back() + request_recvcounts.back();
  std::vector<int> request_sendbuf(static_cast<size_t>(std::max(total_request_send, 0)), 0);
  std::vector<int> request_recvbuf(static_cast<size_t>(std::max(total_request_recv, 0)), 0);

  for (int rank = 0; rank < world_size; ++rank) {
    const auto& ids = requested_leaf_ids_by_rank[static_cast<size_t>(rank)];
    if (ids.empty()) continue;
    std::memcpy(
        request_sendbuf.data() + request_senddispls[static_cast<size_t>(rank)],
        ids.data(),
        static_cast<size_t>(ids.size()) * sizeof(int));
  }

  MPI_Alltoallv(
      request_sendbuf.empty() ? nullptr : request_sendbuf.data(),
      request_sendcounts.data(),
      request_senddispls.data(),
      MPI_INT,
      request_recvbuf.empty() ? nullptr : request_recvbuf.data(),
      request_recvcounts.data(),
      request_recvdispls.data(),
      MPI_INT,
      world);

  std::vector<const RankClusterSummary*> local_leaf_lookup(local_leaf_summaries.size(), nullptr);
  for (const auto& leaf : local_leaf_summaries) {
    if (leaf.leaf_id >= 0 && static_cast<size_t>(leaf.leaf_id) < local_leaf_lookup.size()) {
      local_leaf_lookup[static_cast<size_t>(leaf.leaf_id)] = &leaf;
    }
  }

  std::vector<int> atom_sendcounts(static_cast<size_t>(world_size), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    const int begin = request_recvdispls[static_cast<size_t>(rank)];
    const int end = begin + request_recvcounts[static_cast<size_t>(rank)];
    int atom_count = 0;
    for (int idx = begin; idx < end; ++idx) {
      const int leaf_id = request_recvbuf[static_cast<size_t>(idx)];
      if (leaf_id >= 0 && static_cast<size_t>(leaf_id) < local_leaf_lookup.size() &&
          local_leaf_lookup[static_cast<size_t>(leaf_id)] != nullptr) {
        atom_count += static_cast<int>(local_leaf_lookup[static_cast<size_t>(leaf_id)]->natoms);
      }
    }
    atom_sendcounts[static_cast<size_t>(rank)] = atom_count;
  }

  std::vector<int> atom_recvcounts(static_cast<size_t>(world_size), 0);
  MPI_Alltoall(atom_sendcounts.data(), 1, MPI_INT, atom_recvcounts.data(), 1, MPI_INT, world);

  std::vector<int> pos_sendcounts(static_cast<size_t>(world_size), 0);
  std::vector<int> pos_recvcounts(static_cast<size_t>(world_size), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    pos_sendcounts[static_cast<size_t>(rank)] = atom_sendcounts[static_cast<size_t>(rank)] * 3;
    pos_recvcounts[static_cast<size_t>(rank)] = atom_recvcounts[static_cast<size_t>(rank)] * 3;
  }
  auto atom_senddispls = build_displs(atom_sendcounts);
  auto atom_recvdispls = build_displs(atom_recvcounts);
  auto pos_senddispls = build_displs(pos_sendcounts);
  auto pos_recvdispls = build_displs(pos_recvcounts);

  const int total_atom_send = atom_senddispls.empty() ? 0 : atom_senddispls.back() + atom_sendcounts.back();
  const int total_atom_recv = atom_recvdispls.empty() ? 0 : atom_recvdispls.back() + atom_recvcounts.back();
  const int total_pos_send = pos_senddispls.empty() ? 0 : pos_senddispls.back() + pos_sendcounts.back();
  const int total_pos_recv = pos_recvdispls.empty() ? 0 : pos_recvdispls.back() + pos_recvcounts.back();

  std::vector<float> source_sendbuf(static_cast<size_t>(std::max(total_atom_send, 0)), 0.0f);
  std::vector<float> source_recvbuf(static_cast<size_t>(std::max(total_atom_recv, 0)), 0.0f);
  std::vector<float> pos_sendbuf(static_cast<size_t>(std::max(total_pos_send, 0)), 0.0f);
  std::vector<float> pos_recvbuf(static_cast<size_t>(std::max(total_pos_recv, 0)), 0.0f);

  const float* pos_ptr = local_pos.numel() > 0 ? local_pos.data_ptr<float>() : nullptr;
  const float* source_ptr = local_source.numel() > 0 ? local_source.data_ptr<float>() : nullptr;

  for (int rank = 0; rank < world_size; ++rank) {
    int atom_cursor = atom_senddispls[static_cast<size_t>(rank)];
    int pos_cursor = pos_senddispls[static_cast<size_t>(rank)];
    const int begin = request_recvdispls[static_cast<size_t>(rank)];
    const int end = begin + request_recvcounts[static_cast<size_t>(rank)];
    for (int idx = begin; idx < end; ++idx) {
      const int leaf_id = request_recvbuf[static_cast<size_t>(idx)];
      if (leaf_id < 0 || static_cast<size_t>(leaf_id) >= local_leaf_lookup.size()) continue;
      const auto* leaf = local_leaf_lookup[static_cast<size_t>(leaf_id)];
      if (!leaf) continue;
      for (int atom_idx : leaf->indices) {
        source_sendbuf[static_cast<size_t>(atom_cursor)] = source_ptr[atom_idx];
        std::memcpy(
            pos_sendbuf.data() + pos_cursor,
            pos_ptr + static_cast<size_t>(atom_idx) * 3,
            3 * sizeof(float));
        ++atom_cursor;
        pos_cursor += 3;
      }
    }
  }

  const bool use_gpu_exchange = gpu_aware_mpi && preferred_device.is_cuda();
  if (!use_gpu_exchange) {
    MPI_Alltoallv(
        pos_sendbuf.empty() ? nullptr : pos_sendbuf.data(),
        pos_sendcounts.data(),
        pos_senddispls.data(),
        MPI_FLOAT,
        pos_recvbuf.empty() ? nullptr : pos_recvbuf.data(),
        pos_recvcounts.data(),
        pos_recvdispls.data(),
        MPI_FLOAT,
        world);
    MPI_Alltoallv(
        source_sendbuf.empty() ? nullptr : source_sendbuf.data(),
        atom_sendcounts.data(),
        atom_senddispls.data(),
        MPI_FLOAT,
        source_recvbuf.empty() ? nullptr : source_recvbuf.data(),
        atom_recvcounts.data(),
        atom_recvdispls.data(),
        MPI_FLOAT,
        world);

    ImportedAtoms imported;
    imported.pos = cpu_float_tensor_from_buffer(pos_recvbuf, {static_cast<int64_t>(total_atom_recv), 3});
    imported.source = cpu_float_tensor_from_buffer(source_recvbuf, {static_cast<int64_t>(total_atom_recv)});
    return imported;
  }

  auto pos_send_dev =
      cpu_float_tensor_from_buffer(pos_sendbuf, {static_cast<int64_t>(total_atom_send), 3}).to(preferred_device);
  auto source_send_dev =
      cpu_float_tensor_from_buffer(source_sendbuf, {static_cast<int64_t>(total_atom_send)}).to(preferred_device);
  auto pos_recv_dev = torch::empty(
      {static_cast<int64_t>(total_atom_recv), 3},
      torch::TensorOptions().dtype(torch::kFloat32).device(preferred_device));
  auto source_recv_dev = torch::empty(
      {static_cast<int64_t>(total_atom_recv)},
      torch::TensorOptions().dtype(torch::kFloat32).device(preferred_device));

  MPI_Alltoallv(
      pos_send_dev.numel() > 0 ? pos_send_dev.data_ptr<float>() : nullptr,
      pos_sendcounts.data(),
      pos_senddispls.data(),
      MPI_FLOAT,
      pos_recv_dev.numel() > 0 ? pos_recv_dev.data_ptr<float>() : nullptr,
      pos_recvcounts.data(),
      pos_recvdispls.data(),
      MPI_FLOAT,
      world);
  MPI_Alltoallv(
      source_send_dev.numel() > 0 ? source_send_dev.data_ptr<float>() : nullptr,
      atom_sendcounts.data(),
      atom_senddispls.data(),
      MPI_FLOAT,
      source_recv_dev.numel() > 0 ? source_recv_dev.data_ptr<float>() : nullptr,
      atom_recvcounts.data(),
      atom_recvdispls.data(),
      MPI_FLOAT,
      world);

  ImportedAtoms imported;
  imported.pos = pos_recv_dev;
  imported.source = source_recv_dev;
  return imported;
}

ImportedAtoms exchange_requested_atoms(
    MPI_Comm world,
    const std::vector<int>& request_flags,
    const torch::Tensor& local_pos,
    const torch::Tensor& local_source) {
  int world_size = 1;
  MPI_Comm_size(world, &world_size);
  std::vector<int> requested_by(static_cast<size_t>(world_size), 0);
  MPI_Alltoall(
      request_flags.data(),
      1,
      MPI_INT,
      requested_by.data(),
      1,
      MPI_INT,
      world);

  const int local_n = static_cast<int>(local_pos.size(0));
  std::vector<int> send_atom_counts(static_cast<size_t>(world_size), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    if (requested_by[static_cast<size_t>(rank)] != 0) {
      send_atom_counts[static_cast<size_t>(rank)] = local_n;
    }
  }
  std::vector<int> recv_atom_counts(static_cast<size_t>(world_size), 0);
  MPI_Alltoall(
      send_atom_counts.data(),
      1,
      MPI_INT,
      recv_atom_counts.data(),
      1,
      MPI_INT,
      world);

  std::vector<int> send_pos_counts(static_cast<size_t>(world_size), 0);
  std::vector<int> recv_pos_counts(static_cast<size_t>(world_size), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    send_pos_counts[static_cast<size_t>(rank)] = send_atom_counts[static_cast<size_t>(rank)] * 3;
    recv_pos_counts[static_cast<size_t>(rank)] = recv_atom_counts[static_cast<size_t>(rank)] * 3;
  }
  auto send_pos_displs = build_displs(send_pos_counts);
  auto recv_pos_displs = build_displs(recv_pos_counts);
  auto send_source_displs = build_displs(send_atom_counts);
  auto recv_source_displs = build_displs(recv_atom_counts);

  const float* local_pos_ptr = local_pos.defined() && local_pos.numel() > 0 ? local_pos.data_ptr<float>() : nullptr;
  const float* local_source_ptr =
      local_source.defined() && local_source.numel() > 0 ? local_source.data_ptr<float>() : nullptr;

  const int total_send_pos = send_pos_displs.empty() ? 0 : send_pos_displs.back() + send_pos_counts.back();
  const int total_recv_pos = recv_pos_displs.empty() ? 0 : recv_pos_displs.back() + recv_pos_counts.back();
  const int total_send_source =
      send_source_displs.empty() ? 0 : send_source_displs.back() + send_atom_counts.back();
  const int total_recv_source =
      recv_source_displs.empty() ? 0 : recv_source_displs.back() + recv_atom_counts.back();

  std::vector<float> send_pos_buf(static_cast<size_t>(std::max(total_send_pos, 0)), 0.0f);
  std::vector<float> recv_pos_buf(static_cast<size_t>(std::max(total_recv_pos, 0)), 0.0f);
  std::vector<float> send_source_buf(static_cast<size_t>(std::max(total_send_source, 0)), 0.0f);
  std::vector<float> recv_source_buf(static_cast<size_t>(std::max(total_recv_source, 0)), 0.0f);

  for (int rank = 0; rank < world_size; ++rank) {
    if (send_atom_counts[static_cast<size_t>(rank)] == 0 || local_n == 0) continue;
    std::memcpy(
        send_pos_buf.data() + send_pos_displs[static_cast<size_t>(rank)],
        local_pos_ptr,
        static_cast<size_t>(local_n) * 3 * sizeof(float));
    std::memcpy(
        send_source_buf.data() + send_source_displs[static_cast<size_t>(rank)],
        local_source_ptr,
        static_cast<size_t>(local_n) * sizeof(float));
  }

  MPI_Alltoallv(
      send_pos_buf.data(),
      send_pos_counts.data(),
      send_pos_displs.data(),
      MPI_FLOAT,
      recv_pos_buf.data(),
      recv_pos_counts.data(),
      recv_pos_displs.data(),
      MPI_FLOAT,
      world);
  MPI_Alltoallv(
      send_source_buf.data(),
      send_atom_counts.data(),
      send_source_displs.data(),
      MPI_FLOAT,
      recv_source_buf.data(),
      recv_atom_counts.data(),
      recv_source_displs.data(),
      MPI_FLOAT,
      world);

  ImportedAtoms imported;
  imported.pos = torch::from_blob(
                     recv_pos_buf.data(),
                     {static_cast<int64_t>(total_recv_source), 3},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                     .clone();
  imported.source = torch::from_blob(
                        recv_source_buf.data(),
                        {static_cast<int64_t>(total_recv_source)},
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                        .clone();
  return imported;
}

}  // namespace

MFFTreeFmmSolver::MFFTreeFmmSolver() {
  config_.gpu_aware_mpi = read_bool_env("MFF_TREE_FMM_GPU_AWARE_MPI", config_.gpu_aware_mpi);
  config_.device_local_eval = read_bool_env("MFF_TREE_FMM_DEVICE_LOCAL_EVAL", config_.device_local_eval);
  config_.reuse_position_tol = read_nonnegative_env("MFF_TREE_FMM_REUSE_POSITION_TOL", config_.reuse_position_tol);
}

void MFFTreeFmmSolver::set_config(const TreeFmmConfig& config) {
  config_ = config;
  config_.gpu_aware_mpi = read_bool_env("MFF_TREE_FMM_GPU_AWARE_MPI", config_.gpu_aware_mpi);
  config_.device_local_eval = read_bool_env("MFF_TREE_FMM_DEVICE_LOCAL_EVAL", config_.device_local_eval);
  config_.reuse_position_tol = read_nonnegative_env("MFF_TREE_FMM_REUSE_POSITION_TOL", config_.reuse_position_tol);
  distributed_remote_geometry_cache_ = DistributedRemoteGeometryCache{};
  distributed_request_plan_cache_ = DistributedRequestPlanCache{};
  local_geometry_version_ = 0;
}

ReciprocalOutputs MFFTreeFmmSolver::compute(const ReciprocalInputs& inputs) const {
  if (config_.boundary != "nonperiodic") {
    throw std::runtime_error("tree_fmm runtime currently requires long_range_boundary=nonperiodic");
  }
  if (inputs.local_source.dim() != 2 || inputs.local_source.size(1) != 1) {
    throw std::runtime_error("tree_fmm runtime currently expects a scalar latent_charge source with shape (N,1)");
  }
  if (config_.multipole_order != 0) {
    throw std::runtime_error("tree_fmm runtime currently supports only multipole_order=0");
  }

  const int world_rank = inputs.world_rank;
  const int world_size = inputs.world_size;
  const bool use_gpu_main = inputs.preferred_device.is_cuda();
  const bool gpu_aware_mpi_enabled = gpu_aware_mpi_runtime_enabled(config_.gpu_aware_mpi, inputs.preferred_device);

  if (use_gpu_main) {
    auto pos = inputs.local_pos.to(inputs.preferred_device, torch::kFloat32).contiguous().clone();
    auto source = inputs.local_source.to(inputs.preferred_device, torch::kFloat32).contiguous().view({-1}).clone();
    if (config_.neutralize) {
      double local_sum = source.numel() > 0 ? source.sum().item<double>() : 0.0;
      double global_sum = 0.0;
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, inputs.world);
      double local_n = static_cast<double>(source.numel());
      double global_n = 0.0;
      MPI_Allreduce(&local_n, &global_n, 1, MPI_DOUBLE, MPI_SUM, inputs.world);
      if (global_n > 0.0 && source.numel() > 0) {
        source = source - static_cast<float>(global_sum / global_n);
      }
    }

    const int64_t natoms = pos.size(0);
    auto local_ids = inputs.local_global_ids.defined()
                         ? inputs.local_global_ids.to(inputs.preferred_device, torch::kInt64).contiguous().view({-1})
                         : torch::arange(natoms, torch::TensorOptions().dtype(torch::kInt64).device(inputs.preferred_device));
    const int leaf_size = std::max(config_.leaf_size, 1);
    const bool local_geometry_reused =
        linear_tree_cache_is_valid(linear_tree_cache_, local_ids, pos.detach(), leaf_size, config_.reuse_position_tol);
    if (!local_geometry_reused) {
      ++local_geometry_version_;
      distributed_request_plan_cache_.valid = false;
    }
    build_or_reuse_linear_tree_cache(
        linear_tree_cache_,
        pos.detach(),
        source.detach(),
        local_ids,
        leaf_size,
        config_.reuse_position_tol);

    DistributedCommStats dist_stats;
    const auto local_payload = build_local_leaf_payload(linear_tree_cache_, world_rank);
    const auto remote_payload =
        gather_remote_leaf_payload(
            inputs.world,
            local_payload,
            world_size,
            local_geometry_version_,
            distributed_remote_geometry_cache_,
            inputs.preferred_device,
            gpu_aware_mpi_enabled,
            &dist_stats);
    RemoteLeafPayload far_payload;
    DeviceRequestPlan request_plan;
    if (distributed_request_plan_cache_is_valid(
            distributed_request_plan_cache_,
            world_size,
            world_rank,
            local_geometry_version_,
            distributed_remote_geometry_cache_.geometry_versions,
            distributed_remote_geometry_cache_.leaf_counts) &&
        distributed_request_plan_cache_.far_indices.defined() &&
        distributed_request_plan_cache_.far_indices.device() == remote_payload.leaf_ids.device()) {
      request_plan = empty_device_request_plan(remote_payload.leaf_ids.device(), world_size);
      request_plan.far_indices = distributed_request_plan_cache_.far_indices;
      request_plan.request_leaf_ids = distributed_request_plan_cache_.request_leaf_ids;
      request_plan.request_ranks = distributed_request_plan_cache_.request_ranks;
      request_plan.request_counts_per_rank = distributed_request_plan_cache_.request_counts_per_rank;
      request_plan.request_rank_offsets = distributed_request_plan_cache_.request_rank_offsets;
      far_payload = index_payload(remote_payload, request_plan.far_indices);
    } else {
      request_plan = classify_remote_payload(
          linear_tree_cache_,
          remote_payload,
          world_rank,
          world_size,
          config_.theta,
          far_payload);
      distributed_request_plan_cache_.valid = true;
      distributed_request_plan_cache_.world_size = world_size;
      distributed_request_plan_cache_.world_rank = world_rank;
      distributed_request_plan_cache_.local_geometry_version = local_geometry_version_;
      distributed_request_plan_cache_.remote_geometry_versions = distributed_remote_geometry_cache_.geometry_versions;
      distributed_request_plan_cache_.remote_leaf_counts = distributed_remote_geometry_cache_.leaf_counts;
      distributed_request_plan_cache_.far_indices = request_plan.far_indices;
      distributed_request_plan_cache_.request_leaf_ids = request_plan.request_leaf_ids;
      distributed_request_plan_cache_.request_ranks = request_plan.request_ranks;
      distributed_request_plan_cache_.request_counts_per_rank = request_plan.request_counts_per_rank;
      distributed_request_plan_cache_.request_rank_offsets = request_plan.request_rank_offsets;
    }
    dist_stats.requested_leafs = request_plan.size();

    const auto imported = exchange_requested_linear_leaf_atoms(
        inputs.world,
        request_plan,
        linear_tree_cache_,
        inputs.preferred_device,
        gpu_aware_mpi_enabled);
    dist_stats.imported_atoms = imported.source.defined() ? imported.source.size(0) : 0;
    if (distributed_profile_enabled()) {
      const long long charge_bytes = static_cast<long long>(dist_stats.charge_leafs) * 4LL;
      const long long geometry_bytes = static_cast<long long>(dist_stats.geometry_refreshed_leafs) * (8LL + 4LL * 4LL);
      std::fprintf(
          stderr,
          "[tree_fmm][dist] leafs=%lld charge_bytes=%lld geometry_refresh_leafs=%lld geometry_bytes=%lld requested_leafs=%lld imported_atoms=%lld\n",
          static_cast<long long>(dist_stats.total_leafs),
          charge_bytes,
          static_cast<long long>(dist_stats.geometry_refreshed_leafs),
          geometry_bytes,
          static_cast<long long>(dist_stats.requested_leafs),
          static_cast<long long>(dist_stats.imported_atoms));
    }

    const auto local_contrib = local_linear_tree_contribution_explicit(linear_tree_cache_, config_);
    const auto far_contrib = summary_contribution_explicit(pos, source, far_payload, config_);
    const auto exact_contrib = exact_imported_contribution_explicit(pos, source, imported, config_);
    auto atom_energy = local_contrib.atom_energy + far_contrib.atom_energy + exact_contrib.atom_energy;
    auto forces = local_contrib.forces + far_contrib.forces + exact_contrib.forces;

    const double local_energy_value = atom_energy.sum().item<double>();
    double total_energy_value = 0.0;
    MPI_Allreduce(&local_energy_value, &total_energy_value, 1, MPI_DOUBLE, MPI_SUM, inputs.world);

    double global_atom_count = 0.0;
    const double local_atom_count = static_cast<double>(natoms);
    MPI_Allreduce(&local_atom_count, &global_atom_count, 1, MPI_DOUBLE, MPI_SUM, inputs.world);
    if (config_.energy_partition == "uniform" && global_atom_count > 0.0 && natoms > 0) {
      atom_energy = torch::full_like(atom_energy, static_cast<float>(total_energy_value / global_atom_count));
    }

    ReciprocalOutputs outputs;
    outputs.energy = (inputs.need_energy && world_rank == 0) ? total_energy_value : 0.0;
    outputs.forces_local = forces.contiguous();
    outputs.atom_energy_local = atom_energy.contiguous();
    outputs.boundary_mode = ReciprocalBoundaryMode::Open3D;
    outputs.backend = (world_size > 1) ? ReciprocalBackend::DistributedTreeFmm : ReciprocalBackend::ReplicatedAtoms;
    return outputs;
  }

  auto pos = inputs.local_pos.to(torch::kCPU, torch::kFloat32).contiguous().clone();
  auto source = inputs.local_source.to(torch::kCPU, torch::kFloat32).contiguous().view({-1}).clone();

  if (config_.neutralize) {
    double local_sum = source.numel() > 0 ? source.sum().item<double>() : 0.0;
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, inputs.world);
    double local_n = static_cast<double>(source.numel());
    double global_n = 0.0;
    MPI_Allreduce(&local_n, &global_n, 1, MPI_DOUBLE, MPI_SUM, inputs.world);
    if (global_n > 0.0 && source.numel() > 0) {
      source = source - static_cast<float>(global_sum / global_n);
    }
  }

  const int64_t natoms = pos.size(0);
  pos.set_requires_grad(natoms > 0);
  auto potential = torch::zeros({natoms}, pos.options());
  std::unique_ptr<TreeNode> root;
  std::vector<RankClusterSummary> local_leaf_summaries;
  if (natoms > 0) {
    auto local_ids = inputs.local_global_ids.defined()
                         ? inputs.local_global_ids.to(pos.device(), torch::kInt64).contiguous().view({-1})
                         : torch::arange(natoms, torch::TensorOptions().dtype(torch::kInt64).device(pos.device()));
    std::vector<int> indices(static_cast<size_t>(natoms));
    std::iota(indices.begin(), indices.end(), 0);
    const int leaf_size = std::max(config_.leaf_size, 1);
    if (tree_cache_is_valid(
            cached_local_global_ids_cpu_,
            cached_local_pos_cpu_,
            cached_tree_topology_,
            cached_leaf_size_,
            local_ids,
            pos.detach(),
            leaf_size,
            config_.reuse_position_tol)) {
      root = rebuild_tree_from_topology(*cached_tree_topology_, pos.detach());
      cached_local_pos_cpu_ = pos.detach().to(torch::kCPU, torch::kFloat32).clone();
    } else {
      root = build_tree(pos.detach(), indices, leaf_size);
      cached_tree_topology_ = clone_tree_topology(*root);
      cached_local_global_ids_cpu_ = local_ids.to(torch::kCPU, torch::kInt64).clone();
      cached_local_pos_cpu_ = pos.detach().to(torch::kCPU, torch::kFloat32).clone();
      cached_leaf_size_ = leaf_size;
    }
    int next_leaf_id = 0;
    collect_leaf_summaries(*root, pos.detach(), source.detach(), world_rank, next_leaf_id, local_leaf_summaries);
  }
  const auto remote_leaf_summaries = gather_leaf_summaries(inputs.world, local_leaf_summaries, world_size);
  std::vector<std::vector<int>> requested_leaf_ids_by_rank(static_cast<size_t>(world_size));
  std::vector<RankClusterSummary> far_summaries;
  far_summaries.reserve(remote_leaf_summaries.size());
  for (const auto& remote_leaf : remote_leaf_summaries) {
    if (remote_leaf.rank == world_rank || !remote_leaf.has_atoms()) continue;
    bool exact_required = false;
    for (const auto& local_leaf : local_leaf_summaries) {
      if (leaf_requires_exact_exchange(local_leaf, remote_leaf, config_.theta)) {
        exact_required = true;
        break;
      }
    }
    if (exact_required) {
      requested_leaf_ids_by_rank[static_cast<size_t>(remote_leaf.rank)].push_back(remote_leaf.leaf_id);
    } else {
      far_summaries.push_back(remote_leaf);
    }
  }
  const auto imported =
      exchange_requested_leaf_atoms(
          inputs.world,
          requested_leaf_ids_by_rank,
          local_leaf_summaries,
          pos.detach(),
          source.detach(),
          inputs.preferred_device,
          gpu_aware_mpi_enabled);
  const bool use_device_local_eval = config_.device_local_eval && inputs.preferred_device.is_cuda();
  auto local_far_atom_energy = torch::zeros({natoms}, source.options());
  auto local_far_forces = torch::zeros_like(pos);
  if (natoms > 0) {
    if (use_device_local_eval) {
      const auto local_contrib =
          local_leaf_device_atom_energy_and_forces(pos.detach(), source.detach(), local_leaf_summaries, config_, inputs.preferred_device);
      const auto far_contrib =
          summary_atom_energy_and_forces(pos.detach(), source.detach(), far_summaries, config_, inputs.preferred_device);
      local_far_atom_energy = local_contrib.atom_energy + far_contrib.atom_energy;
      local_far_forces = local_contrib.forces + far_contrib.forces;
    } else {
      for (int64_t i = 0; i < natoms; ++i) {
        potential.index_put_(
            {i},
            approximate_node_potential(*root, pos, source, static_cast<int>(i), pos.index({i}), config_));
      }
      potential = potential + summary_point_potential(pos, far_summaries, config_);
      local_far_atom_energy = 0.5f * source * potential;
      local_far_atom_energy = local_far_atom_energy * config_.energy_scale;
      auto differentiable_total = local_far_atom_energy.sum();
      auto grads = torch::autograd::grad({differentiable_total}, {pos}, {}, true, false);
      local_far_forces = -grads[0].detach();
      local_far_atom_energy = local_far_atom_energy.detach();
    }
  }
  const auto exact_contrib =
      exact_point_atom_energy_and_forces(pos.detach(), source.detach(), imported, config_, inputs.preferred_device);
  auto atom_energy = local_far_atom_energy + exact_contrib.atom_energy;
  auto local_total_energy = atom_energy.sum();

  double local_energy_value = local_total_energy.item<double>();
  double total_energy_value = 0.0;
  MPI_Allreduce(&local_energy_value, &total_energy_value, 1, MPI_DOUBLE, MPI_SUM, inputs.world);

  double global_atom_count = 0.0;
  const double local_atom_count = static_cast<double>(natoms);
  MPI_Allreduce(&local_atom_count, &global_atom_count, 1, MPI_DOUBLE, MPI_SUM, inputs.world);
  if (config_.energy_partition == "uniform" && global_atom_count > 0.0 && natoms > 0) {
    atom_energy = torch::full_like(atom_energy, static_cast<float>(total_energy_value / global_atom_count));
  }

  torch::Tensor forces = local_far_forces + exact_contrib.forces;

  ReciprocalOutputs outputs;
  outputs.energy = (inputs.need_energy && world_rank == 0) ? total_energy_value : 0.0;
  outputs.forces_local = forces.to(inputs.preferred_device, torch::kFloat32).contiguous();
  outputs.atom_energy_local = atom_energy.detach().to(inputs.preferred_device, torch::kFloat32).contiguous();
  outputs.boundary_mode = ReciprocalBoundaryMode::Open3D;
  outputs.backend = (world_size > 1) ? ReciprocalBackend::DistributedTreeFmm : ReciprocalBackend::ReplicatedAtoms;
  return outputs;
}

}  // namespace mfftorch
