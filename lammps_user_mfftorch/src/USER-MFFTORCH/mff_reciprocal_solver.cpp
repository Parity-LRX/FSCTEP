#include "mff_reciprocal_solver.h"

#include <torch/fft.h>

#include <ATen/ops/linalg_det.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mfftorch {

namespace {

torch::Tensor build_corner_offsets(const torch::TensorOptions& options) {
  return torch::tensor(
      {{0, 0, 0},
       {0, 0, 1},
       {0, 1, 0},
       {0, 1, 1},
       {1, 0, 0},
       {1, 0, 1},
       {1, 1, 0},
       {1, 1, 1}},
      options);
}

std::vector<int> build_displs(const std::vector<int>& counts) {
  std::vector<int> displs(counts.size(), 0);
  for (size_t i = 1; i < counts.size(); ++i) displs[i] = displs[i - 1] + counts[i - 1];
  return displs;
}

int read_positive_env(const char* name, int fallback) {
  if (const char* env = std::getenv(name)) {
    const int parsed = std::atoi(env);
    if (parsed > 0) return parsed;
  }
  return fallback;
}

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

ReciprocalBackend parse_backend_env(const char* env) {
  if (!env || env[0] == '\0') return ReciprocalBackend::Auto;
  const std::string value(env);
  if (value == "auto") return ReciprocalBackend::Auto;
  if (value == "replicated" || value == "replicated_atoms") return ReciprocalBackend::ReplicatedAtoms;
  if (value == "mesh_reduce") return ReciprocalBackend::MeshReduce;
  if (value == "distributed" || value == "distributed_slab_fft") return ReciprocalBackend::DistributedSlabFFT;
  if (value == "distributed_tree_fmm") return ReciprocalBackend::DistributedTreeFmm;
  return ReciprocalBackend::Auto;
}

ReciprocalGreenMode parse_green_mode_string(const std::string& value) {
  if (value == "poisson") return ReciprocalGreenMode::Poisson;
  if (value == "learned_poisson") return ReciprocalGreenMode::LearnedPoisson;
  return ReciprocalGreenMode::Poisson;
}

torch::Tensor ensure_contiguous_cpu_float(const torch::Tensor& tensor) {
  return tensor.to(torch::kCPU, torch::kFloat32).contiguous();
}

std::vector<float> flatten_cell_values(const torch::Tensor& effective_cell_cpu) {
  auto flat = effective_cell_cpu.contiguous().view({-1});
  const float* ptr = flat.data_ptr<float>();
  return std::vector<float>(ptr, ptr + flat.numel());
}

int encode_mesh_triplet(int x, int y, int z, int mesh_size) {
  return (x * mesh_size + y) * mesh_size + z;
}

}  // namespace

bool MFFReciprocalSolver::SpectralCacheKey::matches(
    const torch::Tensor& effective_cell_cpu,
    int rank,
    const AxisPartition& y_part) const {
  return world_rank == rank &&
         effective_cell_values == flatten_cell_values(effective_cell_cpu) &&
         y_counts == y_part.counts &&
         y_displs == y_part.displs;
}

MFFReciprocalSolver::MFFReciprocalSolver() {
  mesh_size_ = read_positive_env("MFF_RECIPROCAL_MESH", mesh_size_);
  include_k0_ = false;
  neutralize_ = true;
  gpu_aware_mpi_ = read_bool_env("MFF_RECIPROCAL_GPU_AWARE_MPI", gpu_aware_mpi_);
  k_norm_floor_ = 1.0e-6;
  config_.mesh_size = mesh_size_;
  config_.include_k0 = include_k0_;
  config_.neutralize = neutralize_;
  config_.gpu_aware_mpi = gpu_aware_mpi_;
  config_.k_norm_floor = k_norm_floor_;
  config_.slab_padding_factor = read_positive_env("MFF_RECIPROCAL_SLAB_PADDING", config_.slab_padding_factor);
  config_.backend = parse_backend_env(std::getenv("MFF_RECIPROCAL_BACKEND"));
  if (const char* env = std::getenv("MFF_RECIPROCAL_GREEN_MODE")) {
    config_.green_mode = parse_green_mode_string(std::string(env));
  }
}

const char* MFFReciprocalSolver::backend_name(ReciprocalBackend backend) {
  switch (backend) {
    case ReciprocalBackend::Auto:
      return "auto";
    case ReciprocalBackend::ReplicatedAtoms:
      return "replicated_atoms";
    case ReciprocalBackend::MeshReduce:
      return "mesh_reduce";
    case ReciprocalBackend::DistributedSlabFFT:
      return "distributed_slab_fft";
    case ReciprocalBackend::DistributedTreeFmm:
      return "distributed_tree_fmm";
  }
  return "unknown";
}

ReciprocalBackend MFFReciprocalSolver::resolve_backend(int world_size) const {
  if (config_.backend != ReciprocalBackend::Auto) return config_.backend;
  if (world_size > 1) return ReciprocalBackend::DistributedSlabFFT;
  return ReciprocalBackend::MeshReduce;
}

ReciprocalBoundaryMode MFFReciprocalSolver::resolve_boundary_mode(const std::array<int, 3>& pbc) const {
  if (pbc[0] == 1 && pbc[1] == 1 && pbc[2] == 1) return ReciprocalBoundaryMode::Periodic3D;
  if (pbc[0] == 1 && pbc[1] == 1 && pbc[2] == 0) return ReciprocalBoundaryMode::Slab2D;
  return ReciprocalBoundaryMode::Open3D;
}

MFFReciprocalSolver::AxisPartition MFFReciprocalSolver::build_axis_partition(int world_size) const {
  AxisPartition part;
  part.counts.assign(world_size, 0);
  part.displs.assign(world_size, 0);
  const int base = mesh_size_ / std::max(world_size, 1);
  const int rem = mesh_size_ % std::max(world_size, 1);
  for (int i = 0; i < world_size; ++i) {
    part.counts[i] = base + (i < rem ? 1 : 0);
  }
  part.displs = build_displs(part.counts);
  return part;
}

void MFFReciprocalSolver::ensure_sparse_partition_cache(const AxisPartition& x_part) const {
  if (sparse_partition_cache_key_.matches(x_part) &&
      cached_owner_for_x_.size() == static_cast<size_t>(mesh_size_) &&
      cached_plane_point_displs_.size() == x_part.counts.size()) {
    return;
  }

  cached_owner_for_x_.assign(static_cast<size_t>(mesh_size_), 0);
  for (size_t rank = 0; rank < x_part.counts.size(); ++rank) {
    const int begin = x_part.displs[rank];
    const int end = begin + x_part.counts[rank];
    for (int x = begin; x < end; ++x) {
      cached_owner_for_x_[static_cast<size_t>(x)] = static_cast<int>(rank);
    }
  }

  const int64_t yz_points = static_cast<int64_t>(mesh_size_) * mesh_size_;
  cached_plane_point_displs_.assign(x_part.counts.size(), 0);
  for (size_t rank = 0; rank < x_part.counts.size(); ++rank) {
    cached_plane_point_displs_[rank] = static_cast<int64_t>(x_part.displs[rank]) * yz_points;
  }

  sparse_partition_cache_key_.x_counts = x_part.counts;
  sparse_partition_cache_key_.x_displs = x_part.displs;
}

MFFReciprocalSolver::EffectiveGeometry MFFReciprocalSolver::build_effective_geometry(
    const torch::Tensor& local_pos,
    const torch::Tensor& cell,
    const std::array<int, 3>& pbc,
    const torch::Device& device) const {
  EffectiveGeometry geom;
  geom.boundary_mode = resolve_boundary_mode(pbc);
  geom.effective_cell = (cell.device() == device && cell.dtype() == torch::kFloat32)
                            ? cell.view({3, 3}).contiguous()
                            : cell.to(device, torch::kFloat32).view({3, 3}).contiguous();
  if (geom.boundary_mode != ReciprocalBoundaryMode::Periodic3D) {
    const int pad = std::max(config_.slab_padding_factor, 1);
    for (int axis = 0; axis < 3; ++axis) {
      if (pbc[axis] == 0) {
        auto row = geom.effective_cell.select(0, axis);
        geom.effective_cell.select(0, axis).copy_(row * static_cast<float>(pad));
      }
    }
  }
  geom.inv_cell = torch::linalg_inv(geom.effective_cell);
  auto pos = (local_pos.device() == device && local_pos.dtype() == torch::kFloat32)
                 ? local_pos
                 : local_pos.to(device, torch::kFloat32);
  geom.frac_local = torch::matmul(pos, geom.inv_cell);
  for (int axis = 0; axis < 3; ++axis) {
    if (pbc[axis] == 1) {
      auto frac_axis = geom.frac_local.select(1, axis);
      geom.frac_local.select(1, axis).copy_(frac_axis - torch::floor(frac_axis));
    }
  }
  geom.volume =
      torch::abs(torch::linalg_det(geom.effective_cell)).clamp_min(k_norm_floor_).item<double>();
  return geom;
}

torch::Tensor MFFReciprocalSolver::neutralize_local_source(
    MPI_Comm world,
    const torch::Tensor& local_source,
    const torch::Device& device) const {
  auto source = (local_source.device() == device && local_source.dtype() == torch::kFloat32)
                    ? local_source.contiguous()
                    : local_source.to(device, torch::kFloat32).contiguous();
  if (!neutralize_ || !source.defined() || source.numel() == 0) return source;
  auto local_sum_cpu = source.sum(0).to(torch::kCPU, torch::kFloat64).contiguous();
  std::vector<double> global_sum(static_cast<size_t>(source.size(1)), 0.0);
  MPI_Allreduce(
      local_sum_cpu.data_ptr<double>(),
      global_sum.data(),
      static_cast<int>(source.size(1)),
      MPI_DOUBLE,
      MPI_SUM,
      world);
  const double local_n = static_cast<double>(source.size(0));
  double global_n = 0.0;
  MPI_Allreduce(&local_n, &global_n, 1, MPI_DOUBLE, MPI_SUM, world);
  if (global_n <= 0.0) return source;
  auto mean = torch::from_blob(global_sum.data(), {source.size(1)}, torch::TensorOptions().dtype(torch::kFloat64))
                  .clone()
                  .to(device, torch::kFloat32) /
              global_n;
  return source - mean.unsqueeze(0);
}

torch::Tensor MFFReciprocalSolver::build_integer_frequencies(torch::TensorOptions options) const {
  if (!cached_integer_freq_cpu_.defined() || cached_integer_freq_cpu_.size(0) != mesh_size_) {
    auto cpu_options = torch::TensorOptions().dtype(options.dtype()).device(torch::kCPU);
    auto freq_cpu = torch::arange(mesh_size_, cpu_options);
    const int64_t split = (mesh_size_ + 1) / 2;
    cached_integer_freq_cpu_ = torch::where(freq_cpu < split, freq_cpu, freq_cpu - mesh_size_);
  }
  if (options.device().is_cpu()) return cached_integer_freq_cpu_;
  return cached_integer_freq_cpu_.to(options.device(), options.dtype());
}

torch::Tensor MFFReciprocalSolver::spread_to_mesh_full(
    const torch::Tensor& frac,
    const torch::Tensor& source,
    const std::array<int, 3>& pbc) const {
  const auto channels = source.size(1);
  auto mesh = torch::zeros({mesh_size_, mesh_size_, mesh_size_, channels}, source.options());
  auto flat_mesh = mesh.view({-1, channels});
  auto scaled = frac * static_cast<double>(mesh_size_);
  auto base = torch::floor(scaled).to(torch::kLong);
  auto frac_offset = scaled - base.to(scaled.dtype());
  auto wx0 = 1.0 - frac_offset.select(1, 0);
  auto wy0 = 1.0 - frac_offset.select(1, 1);
  auto wz0 = 1.0 - frac_offset.select(1, 2);
  auto wx1 = frac_offset.select(1, 0);
  auto wy1 = frac_offset.select(1, 1);
  auto wz1 = frac_offset.select(1, 2);
  auto weights = torch::stack(
      {wx0 * wy0 * wz0, wx0 * wy0 * wz1, wx0 * wy1 * wz0, wx0 * wy1 * wz1,
       wx1 * wy0 * wz0, wx1 * wy0 * wz1, wx1 * wy1 * wz0, wx1 * wy1 * wz1},
      1);
  auto offsets = build_corner_offsets(base.options());
  for (int64_t corner = 0; corner < 8; ++corner) {
    auto idx = base + offsets[corner];
    for (int axis = 0; axis < 3; ++axis) {
      auto axis_idx = idx.select(1, axis);
      if (pbc[axis] == 1) {
        axis_idx = torch::remainder(axis_idx, mesh_size_);
      } else {
        axis_idx = axis_idx.clamp(0, mesh_size_ - 1);
      }
      idx.select(1, axis).copy_(axis_idx);
    }
    auto flat_idx = ((idx.select(1, 0) * mesh_size_) + idx.select(1, 1)) * mesh_size_ + idx.select(1, 2);
    flat_mesh.index_add_(0, flat_idx, source * weights.select(1, corner).unsqueeze(1));
  }
  return mesh;
}

torch::Tensor MFFReciprocalSolver::gather_from_mesh_full(
    const torch::Tensor& frac,
    const torch::Tensor& mesh,
    const std::array<int, 3>& pbc) const {
  const auto channels = mesh.size(3);
  auto flat_mesh = mesh.view({-1, channels});
  auto scaled = frac * static_cast<double>(mesh_size_);
  auto base = torch::floor(scaled).to(torch::kLong);
  auto frac_offset = scaled - base.to(scaled.dtype());
  auto wx0 = 1.0 - frac_offset.select(1, 0);
  auto wy0 = 1.0 - frac_offset.select(1, 1);
  auto wz0 = 1.0 - frac_offset.select(1, 2);
  auto wx1 = frac_offset.select(1, 0);
  auto wy1 = frac_offset.select(1, 1);
  auto wz1 = frac_offset.select(1, 2);
  auto weights = torch::stack(
      {wx0 * wy0 * wz0, wx0 * wy0 * wz1, wx0 * wy1 * wz0, wx0 * wy1 * wz1,
       wx1 * wy0 * wz0, wx1 * wy0 * wz1, wx1 * wy1 * wz0, wx1 * wy1 * wz1},
      1);
  auto offsets = build_corner_offsets(base.options());
  auto gathered = torch::zeros({frac.size(0), channels}, mesh.options());
  for (int64_t corner = 0; corner < 8; ++corner) {
    auto idx = base + offsets[corner];
    for (int axis = 0; axis < 3; ++axis) {
      auto axis_idx = idx.select(1, axis);
      if (pbc[axis] == 1) {
        axis_idx = torch::remainder(axis_idx, mesh_size_);
      } else {
        axis_idx = axis_idx.clamp(0, mesh_size_ - 1);
      }
      idx.select(1, axis).copy_(axis_idx);
    }
    auto flat_idx = ((idx.select(1, 0) * mesh_size_) + idx.select(1, 1)) * mesh_size_ + idx.select(1, 2);
    gathered = gathered + flat_mesh.index_select(0, flat_idx) * weights.select(1, corner).unsqueeze(1);
  }
  return gathered;
}

torch::Tensor MFFReciprocalSolver::reduce_scatter_xslab(
    MPI_Comm world,
    const torch::Tensor& full_mesh_local,
    const AxisPartition& x_part,
    const torch::Device& device) const {
  const int world_size = static_cast<int>(x_part.counts.size());
  int world_rank = 0;
  MPI_Comm_rank(world, &world_rank);
  const int64_t channels = full_mesh_local.size(3);
  const int64_t yz = static_cast<int64_t>(mesh_size_) * mesh_size_ * channels;
  if (world_size == 1) return (full_mesh_local.device() == device) ? full_mesh_local : full_mesh_local.to(device);

  auto send_cpu = ensure_contiguous_cpu_float(full_mesh_local).view({-1});
  std::vector<int> recvcounts(world_size, 0);
  for (int i = 0; i < world_size; ++i) {
    recvcounts[i] = x_part.counts[i] * static_cast<int>(yz);
  }
  if (gpu_aware_mpi_ && device.is_cuda() && full_mesh_local.device().is_cuda()) {
    auto send_dev = (full_mesh_local.device() == device && full_mesh_local.dtype() == torch::kFloat32)
                        ? full_mesh_local.contiguous().view({-1})
                        : full_mesh_local.to(device, torch::kFloat32).contiguous().view({-1});
    auto recv_dev = torch::empty(
        {recvcounts[world_rank]},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    MPI_Reduce_scatter(
        send_dev.numel() > 0 ? send_dev.data_ptr<float>() : nullptr,
        recv_dev.numel() > 0 ? recv_dev.data_ptr<float>() : nullptr,
        recvcounts.data(),
        MPI_FLOAT,
        MPI_SUM,
        world);
    return recv_dev.view({x_part.counts[world_rank], mesh_size_, mesh_size_, channels});
  }
  reduce_scatter_recvbuf_.assign(static_cast<size_t>(std::max(recvcounts[world_rank], 0)), 0.0f);
  MPI_Reduce_scatter(
      send_cpu.data_ptr<float>(),
      reduce_scatter_recvbuf_.data(),
      recvcounts.data(),
      MPI_FLOAT,
      MPI_SUM,
      world);
  auto out_cpu =
      torch::from_blob(
          reduce_scatter_recvbuf_.data(),
          {x_part.counts[world_rank], mesh_size_, mesh_size_, channels},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
          .clone();
  return device.is_cpu() ? out_cpu : out_cpu.to(device);
}

torch::Tensor MFFReciprocalSolver::gather_from_xslab_sparse(
    MPI_Comm world,
    const torch::Tensor& frac,
    const torch::Tensor& local_x_mesh,
    const AxisPartition& x_part,
    int channels,
    const std::array<int, 3>& pbc,
    const torch::Device& device) const {
  const int world_size = static_cast<int>(x_part.counts.size());
  if (world_size == 1) return gather_from_mesh_full(frac, local_x_mesh, pbc);

  int world_rank = 0;
  MPI_Comm_rank(world, &world_rank);
  ensure_sparse_partition_cache(x_part);

  if (gpu_aware_mpi_ && device.is_cuda() && frac.device().is_cuda() && local_x_mesh.device().is_cuda()) {
    const int local_n = static_cast<int>(frac.size(0));
    const int field_channels = channels;
    const auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto long_options = torch::TensorOptions().dtype(torch::kLong).device(device);
    const auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);

    auto frac_dev = (frac.dtype() == torch::kFloat32) ? frac.contiguous() : frac.to(device, torch::kFloat32).contiguous();
    auto mesh_dev = (local_x_mesh.dtype() == torch::kFloat32) ? local_x_mesh.contiguous()
                                                              : local_x_mesh.to(device, torch::kFloat32).contiguous();

    auto scaled = frac_dev * static_cast<double>(mesh_size_);
    auto base = torch::floor(scaled).to(torch::kLong);
    auto frac_offset = scaled - base.to(scaled.dtype());
    auto wx0 = 1.0 - frac_offset.select(1, 0);
    auto wy0 = 1.0 - frac_offset.select(1, 1);
    auto wz0 = 1.0 - frac_offset.select(1, 2);
    auto wx1 = frac_offset.select(1, 0);
    auto wy1 = frac_offset.select(1, 1);
    auto wz1 = frac_offset.select(1, 2);
    auto weights = torch::stack(
        {wx0 * wy0 * wz0, wx0 * wy0 * wz1, wx0 * wy1 * wz0, wx0 * wy1 * wz1,
         wx1 * wy0 * wz0, wx1 * wy0 * wz1, wx1 * wy1 * wz0, wx1 * wy1 * wz1},
        1);
    auto offsets = build_corner_offsets(torch::TensorOptions().dtype(torch::kLong).device(device)).view({1, 8, 3});
    auto idx = base.unsqueeze(1) + offsets;
    for (int axis = 0; axis < 3; ++axis) {
      auto axis_idx = idx.select(2, axis);
      if (pbc[axis] == 1) {
        axis_idx = torch::remainder(axis_idx, mesh_size_);
      } else {
        axis_idx = axis_idx.clamp(0, mesh_size_ - 1);
      }
      idx.select(2, axis).copy_(axis_idx);
    }

    auto x = idx.select(2, 0).reshape({-1});
    auto y = idx.select(2, 1).reshape({-1});
    auto z = idx.select(2, 2).reshape({-1});
    auto point_idx_long = ((x * mesh_size_) + y) * mesh_size_ + z;
    auto point_idx = point_idx_long.to(torch::kInt32);
    auto atom_idx = torch::arange(local_n, long_options).unsqueeze(1).expand({local_n, 8}).reshape({-1});
    auto weights_flat = weights.reshape({-1});

    auto owner_cpu = torch::from_blob(
                         cached_owner_for_x_.data(),
                         {mesh_size_},
                         torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
                         .clone();
    auto owner_lut = owner_cpu.to(device, torch::kLong);
    auto owners = owner_lut.index_select(0, x);
    auto owners_cpu = owners.to(torch::kCPU, torch::kInt32).contiguous();
    const int total_requests = static_cast<int>(owners_cpu.numel());
    const int* owners_ptr = owners_cpu.data_ptr<int>();

    std::vector<int> send_point_counts(world_size, 0);
    for (int i = 0; i < total_requests; ++i) {
      const int owner = owners_ptr[i];
      if (owner >= 0 && owner < world_size) send_point_counts[owner] += 1;
    }
    auto request_senddispls = build_displs(send_point_counts);
    std::vector<int> request_offsets = request_senddispls;
    std::vector<int64_t> permutation(static_cast<size_t>(total_requests), 0);
    for (int i = 0; i < total_requests; ++i) {
      const int owner = owners_ptr[i];
      permutation[static_cast<size_t>(request_offsets[owner]++)] = static_cast<int64_t>(i);
    }

    auto perm_cpu = torch::from_blob(
                        permutation.data(),
                        {total_requests},
                        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                        .clone();
    auto perm_dev = perm_cpu.to(device, torch::kLong);
    auto request_send = point_idx.index_select(0, perm_dev).contiguous();
    auto atom_idx_sorted = atom_idx.index_select(0, perm_dev).contiguous();
    auto weights_sorted = weights_flat.index_select(0, perm_dev).contiguous();

    std::vector<int> recv_point_counts(world_size, 0);
    MPI_Alltoall(send_point_counts.data(), 1, MPI_INT, recv_point_counts.data(), 1, MPI_INT, world);
    auto request_recvdispls = build_displs(recv_point_counts);
    const int total_request_recv = request_recvdispls.empty() ? 0 : request_recvdispls.back() + recv_point_counts.back();
    auto request_recv = torch::empty({total_request_recv}, int_options);

    MPI_Alltoallv(
        request_send.numel() > 0 ? request_send.data_ptr<int>() : nullptr,
        send_point_counts.data(),
        request_senddispls.data(),
        MPI_INT,
        request_recv.numel() > 0 ? request_recv.data_ptr<int>() : nullptr,
        recv_point_counts.data(),
        request_recvdispls.data(),
        MPI_INT,
        world);

    std::vector<int> response_sendcounts(world_size, 0), response_recvcounts(world_size, 0);
    for (int rank = 0; rank < world_size; ++rank) {
      response_sendcounts[rank] = recv_point_counts[rank] * field_channels;
      response_recvcounts[rank] = send_point_counts[rank] * field_channels;
    }
    auto response_senddispls = build_displs(response_sendcounts);
    auto response_recvdispls = build_displs(response_recvcounts);
    auto request_recv_long = request_recv.to(torch::kLong);
    auto local_point_idx = request_recv_long - cached_plane_point_displs_[world_rank];
    auto response_send =
        mesh_dev.view({-1, field_channels}).index_select(0, local_point_idx).contiguous();
    const int total_response_recv =
        response_recvdispls.empty() ? 0 : response_recvdispls.back() + response_recvcounts.back();
    auto response_recv = torch::empty({total_response_recv / std::max(field_channels, 1), field_channels}, float_options);

    MPI_Alltoallv(
        response_send.numel() > 0 ? response_send.data_ptr<float>() : nullptr,
        response_sendcounts.data(),
        response_senddispls.data(),
        MPI_FLOAT,
        response_recv.numel() > 0 ? response_recv.data_ptr<float>() : nullptr,
        response_recvcounts.data(),
        response_recvdispls.data(),
        MPI_FLOAT,
        world);

    auto gathered = torch::zeros({local_n, field_channels}, float_options);
    if (response_recv.numel() > 0) {
      auto weighted = response_recv * weights_sorted.unsqueeze(1);
      gathered.index_add_(0, atom_idx_sorted, weighted);
    }
    return gathered;
  }

  auto frac_cpu = ensure_contiguous_cpu_float(frac);
  auto x_mesh_cpu = ensure_contiguous_cpu_float(local_x_mesh);
  const int local_n = static_cast<int>(frac_cpu.size(0));
  const float* frac_ptr = frac_cpu.data_ptr<float>();
  const float* mesh_ptr = x_mesh_cpu.data_ptr<float>();
  const int field_channels = channels;
  const int64_t local_plane_point_displ = cached_plane_point_displs_[world_rank];

  static const int corner_offsets[8][3] = {
      {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
  };

  std::vector<std::vector<int>> request_points_by_rank(world_size);
  std::vector<std::vector<SparseCornerContribution>> contributions_by_rank(world_size);
  std::vector<std::unordered_map<int, int>> unique_request_slots(world_size);
  std::vector<int> send_point_counts(world_size, 0);

  for (int atom = 0; atom < local_n; ++atom) {
    const float fx_scaled = frac_ptr[atom * 3 + 0] * static_cast<float>(mesh_size_);
    const float fy_scaled = frac_ptr[atom * 3 + 1] * static_cast<float>(mesh_size_);
    const float fz_scaled = frac_ptr[atom * 3 + 2] * static_cast<float>(mesh_size_);

    const int base_x = static_cast<int>(std::floor(fx_scaled));
    const int base_y = static_cast<int>(std::floor(fy_scaled));
    const int base_z = static_cast<int>(std::floor(fz_scaled));
    const float off_x = fx_scaled - static_cast<float>(base_x);
    const float off_y = fy_scaled - static_cast<float>(base_y);
    const float off_z = fz_scaled - static_cast<float>(base_z);

    const float wx0 = 1.0f - off_x;
    const float wy0 = 1.0f - off_y;
    const float wz0 = 1.0f - off_z;
    const float wx1 = off_x;
    const float wy1 = off_y;
    const float wz1 = off_z;
    const float weights[8] = {
        wx0 * wy0 * wz0, wx0 * wy0 * wz1, wx0 * wy1 * wz0, wx0 * wy1 * wz1,
        wx1 * wy0 * wz0, wx1 * wy0 * wz1, wx1 * wy1 * wz0, wx1 * wy1 * wz1,
    };

    for (int corner = 0; corner < 8; ++corner) {
      int x = base_x + corner_offsets[corner][0];
      int y = base_y + corner_offsets[corner][1];
      int z = base_z + corner_offsets[corner][2];

      if (pbc[0] == 1) x = (x % mesh_size_ + mesh_size_) % mesh_size_;
      else x = std::min(std::max(x, 0), mesh_size_ - 1);
      if (pbc[1] == 1) y = (y % mesh_size_ + mesh_size_) % mesh_size_;
      else y = std::min(std::max(y, 0), mesh_size_ - 1);
      if (pbc[2] == 1) z = (z % mesh_size_ + mesh_size_) % mesh_size_;
      else z = std::min(std::max(z, 0), mesh_size_ - 1);

      const int owner = cached_owner_for_x_[static_cast<size_t>(x)];
      const int key = encode_mesh_triplet(x, y, z, mesh_size_);
      auto& slot_map = unique_request_slots[owner];
      auto it = slot_map.find(key);
      int response_slot = 0;
      if (it == slot_map.end()) {
        response_slot = send_point_counts[owner];
        slot_map.emplace(key, response_slot);
        request_points_by_rank[owner].push_back(key);
        send_point_counts[owner] += 1;
      } else {
        response_slot = it->second;
      }
      contributions_by_rank[owner].push_back({atom, response_slot, weights[corner]});
    }
  }

  std::vector<int> recv_point_counts(world_size, 0);
  MPI_Alltoall(send_point_counts.data(), 1, MPI_INT, recv_point_counts.data(), 1, MPI_INT, world);

  std::vector<int> request_sendcounts(world_size, 0), request_recvcounts(world_size, 0);
  for (int rank = 0; rank < world_size; ++rank) {
    request_sendcounts[rank] = send_point_counts[rank];
    request_recvcounts[rank] = recv_point_counts[rank];
  }
  auto request_senddispls = build_displs(request_sendcounts);
  auto request_recvdispls = build_displs(request_recvcounts);

  const int total_request_send = request_senddispls.empty() ? 0 : request_senddispls.back() + request_sendcounts.back();
  sparse_request_sendbuf_.assign(static_cast<size_t>(total_request_send), 0);
  for (int rank = 0; rank < world_size; ++rank) {
    if (!request_points_by_rank[rank].empty()) {
      std::memcpy(
          sparse_request_sendbuf_.data() + request_senddispls[rank],
          request_points_by_rank[rank].data(),
          sizeof(int) * static_cast<size_t>(request_sendcounts[rank]));
    }
  }
  const int total_request_recv = request_recvdispls.empty() ? 0 : request_recvdispls.back() + request_recvcounts.back();
  sparse_request_recvbuf_.assign(static_cast<size_t>(total_request_recv), 0);
  MPI_Alltoallv(
      sparse_request_sendbuf_.empty() ? nullptr : sparse_request_sendbuf_.data(),
      request_sendcounts.data(),
      request_senddispls.data(),
      MPI_INT,
      sparse_request_recvbuf_.empty() ? nullptr : sparse_request_recvbuf_.data(),
      request_recvcounts.data(),
      request_recvdispls.data(),
      MPI_INT,
      world);

  std::vector<int> response_sendcounts(world_size, 0), response_recvcounts(world_size, 0);
  for (int rank = 0; rank < world_size; ++rank) {
    response_sendcounts[rank] = recv_point_counts[rank] * field_channels;
    response_recvcounts[rank] = send_point_counts[rank] * field_channels;
  }
  auto response_senddispls = build_displs(response_sendcounts);
  auto response_recvdispls = build_displs(response_recvcounts);
  const int total_response_send = response_senddispls.empty() ? 0 : response_senddispls.back() + response_sendcounts.back();
  sparse_response_sendbuf_.assign(static_cast<size_t>(total_response_send), 0.0f);

  for (int src = 0; src < world_size; ++src) {
    const int point_count = recv_point_counts[src];
    const int request_base = request_recvdispls[src];
    const int response_base = response_senddispls[src];
    for (int point = 0; point < point_count; ++point) {
      const int point_idx = sparse_request_recvbuf_[request_base + point];
      const int64_t local_point_idx = static_cast<int64_t>(point_idx) - local_plane_point_displ;
      const int64_t mesh_offset = local_point_idx * field_channels;
      std::memcpy(
          sparse_response_sendbuf_.data() + response_base + point * field_channels,
          mesh_ptr + mesh_offset,
          sizeof(float) * static_cast<size_t>(field_channels));
    }
  }

  const int total_response_recv = response_recvdispls.empty() ? 0 : response_recvdispls.back() + response_recvcounts.back();
  sparse_response_recvbuf_.assign(static_cast<size_t>(total_response_recv), 0.0f);
  MPI_Alltoallv(
      sparse_response_sendbuf_.empty() ? nullptr : sparse_response_sendbuf_.data(),
      response_sendcounts.data(),
      response_senddispls.data(),
      MPI_FLOAT,
      sparse_response_recvbuf_.empty() ? nullptr : sparse_response_recvbuf_.data(),
      response_recvcounts.data(),
      response_recvdispls.data(),
      MPI_FLOAT,
      world);

  std::vector<float> gathered(static_cast<size_t>(local_n) * static_cast<size_t>(field_channels), 0.0f);
  for (int owner = 0; owner < world_size; ++owner) {
    const int response_base = response_recvdispls[owner];
    for (const auto& contrib : contributions_by_rank[owner]) {
      float* atom_out = gathered.data() + static_cast<size_t>(contrib.atom_idx) * field_channels;
      const float* response_ptr =
          sparse_response_recvbuf_.data() + response_base + contrib.response_slot * field_channels;
      for (int channel = 0; channel < field_channels; ++channel) {
        atom_out[channel] += contrib.weight * response_ptr[channel];
      }
    }
  }

  auto out_cpu =
      torch::from_blob(
          gathered.data(),
          {local_n, field_channels},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
          .clone();
  return device.is_cpu() ? out_cpu : out_cpu.to(device);
}

torch::Tensor MFFReciprocalSolver::transpose_x_to_y_complex(
    MPI_Comm world,
    const torch::Tensor& local_x_complex,
    const AxisPartition& x_part,
    const AxisPartition& y_part,
    int channels,
    const torch::Device& device) const {
  const int world_size = static_cast<int>(x_part.counts.size());
  int world_rank = 0;
  MPI_Comm_rank(world, &world_rank);
  if (world_size == 1) return (local_x_complex.device() == device) ? local_x_complex : local_x_complex.to(device);
  const int nx_local = x_part.counts[world_rank];
  const int ny_local = y_part.counts[world_rank];
  std::vector<int> sendcounts(world_size, 0), recvcounts(world_size, 0);
  std::vector<int> senddispls(world_size, 0), recvdispls(world_size, 0);
  int send_total = 0;
  for (int dst = 0; dst < world_size; ++dst) {
    sendcounts[dst] = nx_local * y_part.counts[dst] * mesh_size_ * channels * 2;
    senddispls[dst] = send_total;
    send_total += sendcounts[dst];
    recvcounts[dst] = x_part.counts[dst] * ny_local * mesh_size_ * channels * 2;
  }
  recvdispls = build_displs(recvcounts);
  if (gpu_aware_mpi_ && device.is_cuda() && local_x_complex.device().is_cuda()) {
    auto local_real =
        torch::view_as_real(
            (local_x_complex.device() == device ? local_x_complex.contiguous() : local_x_complex.to(device).contiguous()))
            .contiguous();
    auto send_dev = torch::empty({send_total}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    for (int dst = 0; dst < world_size; ++dst) {
      auto slice = local_real.narrow(1, y_part.displs[dst], y_part.counts[dst]).contiguous().view({-1});
      send_dev.narrow(0, senddispls[dst], sendcounts[dst]).copy_(slice);
    }
    auto recv_dev = torch::empty(
        {recvdispls.back() + recvcounts.back()},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    MPI_Alltoallv(
        send_dev.numel() > 0 ? send_dev.data_ptr<float>() : nullptr,
        sendcounts.data(),
        senddispls.data(),
        MPI_FLOAT,
        recv_dev.numel() > 0 ? recv_dev.data_ptr<float>() : nullptr,
        recvcounts.data(),
        recvdispls.data(),
        MPI_FLOAT,
        world);
    auto out_real = recv_dev.view({mesh_size_, ny_local, mesh_size_, channels, 2});
    return torch::view_as_complex(out_real);
  }
  auto local_cpu = torch::view_as_real(local_x_complex.to(torch::kCPU)).contiguous();
  transpose_sendbuf_.assign(static_cast<size_t>(send_total), 0.0f);
  for (int dst = 0; dst < world_size; ++dst) {
    auto slice = local_cpu.narrow(1, y_part.displs[dst], y_part.counts[dst]).contiguous();
    std::memcpy(transpose_sendbuf_.data() + senddispls[dst], slice.data_ptr<float>(), sizeof(float) * sendcounts[dst]);
  }
  transpose_recvbuf_.assign(static_cast<size_t>(recvdispls.back() + recvcounts.back()), 0.0f);
  MPI_Alltoallv(
      transpose_sendbuf_.data(),
      sendcounts.data(),
      senddispls.data(),
      MPI_FLOAT,
      transpose_recvbuf_.data(),
      recvcounts.data(),
      recvdispls.data(),
      MPI_FLOAT,
      world);
  auto out_cpu = torch::from_blob(
                     transpose_recvbuf_.data(),
                     {mesh_size_, ny_local, mesh_size_, channels, 2},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                     .clone();
  auto out_complex = torch::view_as_complex(out_cpu);
  return device.is_cpu() ? out_complex : out_complex.to(device);
}

torch::Tensor MFFReciprocalSolver::transpose_y_to_x_complex(
    MPI_Comm world,
    const torch::Tensor& local_y_complex,
    const AxisPartition& x_part,
    const AxisPartition& y_part,
    int channels,
    const torch::Device& device) const {
  const int world_size = static_cast<int>(x_part.counts.size());
  int world_rank = 0;
  MPI_Comm_rank(world, &world_rank);
  if (world_size == 1) return (local_y_complex.device() == device) ? local_y_complex : local_y_complex.to(device);
  const int nx_local = x_part.counts[world_rank];
  const int ny_local = y_part.counts[world_rank];
  std::vector<int> sendcounts(world_size, 0), recvcounts(world_size, 0);
  std::vector<int> senddispls(world_size, 0), recvdispls(world_size, 0);
  int send_total = 0;
  for (int dst = 0; dst < world_size; ++dst) {
    sendcounts[dst] = x_part.counts[dst] * ny_local * mesh_size_ * channels * 2;
    senddispls[dst] = send_total;
    send_total += sendcounts[dst];
    recvcounts[dst] = nx_local * y_part.counts[dst] * mesh_size_ * channels * 2;
  }
  recvdispls = build_displs(recvcounts);
  if (gpu_aware_mpi_ && device.is_cuda() && local_y_complex.device().is_cuda()) {
    auto local_real =
        torch::view_as_real(
            (local_y_complex.device() == device ? local_y_complex.contiguous() : local_y_complex.to(device).contiguous()))
            .contiguous();
    auto send_dev = torch::empty({send_total}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    for (int dst = 0; dst < world_size; ++dst) {
      auto slice = local_real.narrow(0, x_part.displs[dst], x_part.counts[dst]).contiguous().view({-1});
      send_dev.narrow(0, senddispls[dst], sendcounts[dst]).copy_(slice);
    }
    auto recv_dev = torch::empty(
        {recvdispls.back() + recvcounts.back()},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    MPI_Alltoallv(
        send_dev.numel() > 0 ? send_dev.data_ptr<float>() : nullptr,
        sendcounts.data(),
        senddispls.data(),
        MPI_FLOAT,
        recv_dev.numel() > 0 ? recv_dev.data_ptr<float>() : nullptr,
        recvcounts.data(),
        recvdispls.data(),
        MPI_FLOAT,
        world);
    auto out_real = recv_dev.view({nx_local, mesh_size_, mesh_size_, channels, 2});
    return torch::view_as_complex(out_real);
  }
  auto local_cpu = torch::view_as_real(local_y_complex.to(torch::kCPU)).contiguous();
  transpose_sendbuf_.assign(static_cast<size_t>(send_total), 0.0f);
  for (int dst = 0; dst < world_size; ++dst) {
    auto slice = local_cpu.narrow(0, x_part.displs[dst], x_part.counts[dst]).contiguous();
    std::memcpy(transpose_sendbuf_.data() + senddispls[dst], slice.data_ptr<float>(), sizeof(float) * sendcounts[dst]);
  }
  transpose_recvbuf_.assign(static_cast<size_t>(recvdispls.back() + recvcounts.back()), 0.0f);
  MPI_Alltoallv(
      transpose_sendbuf_.data(),
      sendcounts.data(),
      senddispls.data(),
      MPI_FLOAT,
      transpose_recvbuf_.data(),
      recvcounts.data(),
      recvdispls.data(),
      MPI_FLOAT,
      world);
  auto out_cpu = torch::from_blob(
                     transpose_recvbuf_.data(),
                     {nx_local, mesh_size_, mesh_size_, channels, 2},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                     .clone();
  auto out_complex = torch::view_as_complex(out_cpu);
  return device.is_cpu() ? out_complex : out_complex.to(device);
}

torch::Tensor MFFReciprocalSolver::build_local_k_cart(
    const torch::Tensor& effective_cell,
    const AxisPartition& y_part,
    int world_rank,
    torch::TensorOptions options) const {
  auto effective_cell_cpu = effective_cell.to(torch::kCPU, torch::kFloat32).contiguous();
  if (!cached_local_k_cart_cpu_.defined() ||
      !spectral_cache_key_.matches(effective_cell_cpu, world_rank, y_part)) {
    auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto freq = build_integer_frequencies(cpu_options);
    auto freq_y = freq.narrow(0, y_part.displs[world_rank], y_part.counts[world_rank]);
    auto grids = torch::meshgrid({freq, freq_y, freq}, "ij");
    auto integer_k = torch::stack({grids[0], grids[1], grids[2]}, -1).reshape({-1, 3});
    auto inv_cell = torch::linalg_inv(effective_cell_cpu);
    cached_local_k_cart_cpu_ =
        (2.0 * M_PI * torch::matmul(integer_k, inv_cell))
            .reshape({mesh_size_, y_part.counts[world_rank], mesh_size_, 3});
    cached_local_spectral_weights_cpu_ = torch::Tensor();
    spectral_cache_key_.effective_cell_values = flatten_cell_values(effective_cell_cpu);
    spectral_cache_key_.world_rank = world_rank;
    spectral_cache_key_.y_counts = y_part.counts;
    spectral_cache_key_.y_displs = y_part.displs;
  }
  if (options.device().is_cpu()) return cached_local_k_cart_cpu_;
  return cached_local_k_cart_cpu_.to(options.device(), options.dtype());
}

torch::Tensor MFFReciprocalSolver::build_local_spectral_weights(
    const torch::Tensor& effective_cell,
    const AxisPartition& y_part,
    int world_rank,
    torch::TensorOptions options) const {
  auto effective_cell_cpu = effective_cell.to(torch::kCPU, torch::kFloat32).contiguous();
  if (!cached_local_spectral_weights_cpu_.defined() ||
      !spectral_cache_key_.matches(effective_cell_cpu, world_rank, y_part)) {
    if (config_.green_mode != ReciprocalGreenMode::Poisson) {
      throw std::runtime_error(
          "USER-MFFTORCH reciprocal solver currently supports only long_range_green_mode=poisson");
    }
    auto k_cart_cpu =
        build_local_k_cart(effective_cell_cpu, y_part, world_rank, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto k_norm = torch::linalg_vector_norm(k_cart_cpu, 2, -1);
    auto safe = k_norm.clamp_min(k_norm_floor_);
    cached_local_spectral_weights_cpu_ = (4.0 * M_PI) / (safe * safe);
    if (!include_k0_) {
      cached_local_spectral_weights_cpu_ = torch::where(
          k_norm > k_norm_floor_,
          cached_local_spectral_weights_cpu_,
          torch::zeros_like(cached_local_spectral_weights_cpu_));
    }
    spectral_cache_key_.effective_cell_values = flatten_cell_values(effective_cell_cpu);
    spectral_cache_key_.world_rank = world_rank;
    spectral_cache_key_.y_counts = y_part.counts;
    spectral_cache_key_.y_displs = y_part.displs;
  }
  if (options.device().is_cpu()) return cached_local_spectral_weights_cpu_;
  return cached_local_spectral_weights_cpu_.to(options.device(), options.dtype());
}

ReciprocalOutputs MFFReciprocalSolver::compute_replicated_atoms(
    const ReciprocalInputs& inputs,
    const EffectiveGeometry& geom,
    const torch::Tensor& local_source,
    const torch::Device& device,
    ReciprocalBoundaryMode boundary_mode) const {
  ReciprocalOutputs out;
  out.boundary_mode = boundary_mode;
  out.backend = ReciprocalBackend::ReplicatedAtoms;
  const int local_n = static_cast<int>(inputs.local_pos.size(0));
  const int source_channels = static_cast<int>(local_source.size(1));
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  int world_size = 1;
  int world_rank = 0;
  MPI_Comm_size(inputs.world, &world_size);
  MPI_Comm_rank(inputs.world, &world_rank);

  std::vector<int> counts(world_size, 0);
  MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, inputs.world);
  std::vector<int> displs = build_displs(counts);
  int global_n = 0;
  for (int c : counts) global_n += c;
  if (global_n == 0) {
    out.forces_local = torch::zeros({0, 3}, options);
    out.atom_energy_local = torch::zeros({0}, options);
    return out;
  }

  std::vector<int> counts_pos(world_size), displs_pos(world_size), counts_src(world_size), displs_src(world_size);
  for (int i = 0; i < world_size; ++i) {
    counts_pos[i] = counts[i] * 3;
    displs_pos[i] = displs[i] * 3;
    counts_src[i] = counts[i] * source_channels;
    displs_src[i] = displs[i] * source_channels;
  }
  auto pos_local_cpu = ensure_contiguous_cpu_float(inputs.local_pos);
  auto src_local_cpu = ensure_contiguous_cpu_float(local_source);
  std::vector<float> pos_global(static_cast<size_t>(global_n) * 3, 0.0f);
  std::vector<float> src_global(static_cast<size_t>(global_n) * source_channels, 0.0f);
  MPI_Allgatherv(
      local_n > 0 ? pos_local_cpu.data_ptr<float>() : nullptr,
      local_n * 3,
      MPI_FLOAT,
      pos_global.data(),
      counts_pos.data(),
      displs_pos.data(),
      MPI_FLOAT,
      inputs.world);
  MPI_Allgatherv(
      local_n > 0 ? src_local_cpu.data_ptr<float>() : nullptr,
      local_n * source_channels,
      MPI_FLOAT,
      src_global.data(),
      counts_src.data(),
      displs_src.data(),
      MPI_FLOAT,
      inputs.world);

  auto global_pos =
      torch::from_blob(pos_global.data(), {global_n, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
          .clone()
          .to(device)
          .set_requires_grad(true);
  auto global_source =
      torch::from_blob(
          src_global.data(),
          {global_n, source_channels},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
          .clone()
          .to(device);
  auto inv_cell = geom.inv_cell;
  auto frac = torch::matmul(global_pos, inv_cell);
  for (int axis = 0; axis < 3; ++axis) {
    if (inputs.pbc[axis] == 1) {
      auto frac_axis = frac.select(1, axis);
      frac.select(1, axis).copy_(frac_axis - torch::floor(frac_axis));
    }
  }
  auto mesh = spread_to_mesh_full(frac, global_source, inputs.pbc);
  auto mesh_fft = torch::fft::fftn(mesh, {}, {0, 1, 2});
  auto spectral_weights = build_local_spectral_weights(geom.effective_cell, build_axis_partition(1), 0, options);
  auto filtered_mesh_complex = torch::fft::ifftn(mesh_fft * spectral_weights.unsqueeze(-1), {}, {0, 1, 2});
  auto filtered_mesh = torch::view_as_real(filtered_mesh_complex).select(-1, 0);
  auto gathered = gather_from_mesh_full(frac, filtered_mesh, inputs.pbc);
  auto energy = 0.5 * (global_source * gathered).sum() /
                (geom.volume * static_cast<double>(mesh_size_ * mesh_size_ * mesh_size_));
  auto grads = torch::autograd::grad({energy}, {global_pos}, {}, false, false, false);
  auto global_forces = -grads[0].detach();

  out.forces_local = global_forces.narrow(0, displs[world_rank], local_n).clone();
  auto gathered_local = gathered.narrow(0, displs[world_rank], local_n).clone();
  out.atom_energy_local =
      (0.5 * (local_source * gathered_local).sum(-1)) /
      static_cast<double>(geom.volume * mesh_size_ * mesh_size_ * mesh_size_);
  if (!inputs.need_energy) {
    out.atom_energy_local = torch::zeros({local_n}, options);
    return out;
  }
  const double total_energy = energy.detach().to(torch::kCPU).item<double>();
  out.energy = (world_rank == 0) ? total_energy : 0.0;
  return out;
}

ReciprocalOutputs MFFReciprocalSolver::compute_mesh_backend(
    const ReciprocalInputs& inputs,
    const EffectiveGeometry& geom,
    const torch::Tensor& local_source,
    const torch::Device& device,
    ReciprocalBackend backend,
    ReciprocalBoundaryMode boundary_mode) const {
  ReciprocalOutputs out;
  out.boundary_mode = boundary_mode;
  out.backend = backend;
  const int local_n = static_cast<int>(inputs.local_pos.size(0));
  const int channels = static_cast<int>(local_source.size(1));
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  int world_size = 1;
  int world_rank = 0;
  MPI_Comm_size(inputs.world, &world_size);
  MPI_Comm_rank(inputs.world, &world_rank);

  auto x_part = build_axis_partition(world_size);
  auto y_part = build_axis_partition(world_size);
  auto local_mesh_full = spread_to_mesh_full(geom.frac_local, local_source, inputs.pbc);
  auto local_x_mesh = reduce_scatter_xslab(inputs.world, local_mesh_full, x_part, device);

  auto yz_fft = torch::fft::fftn(local_x_mesh, {}, {1, 2});
  auto y_slab = transpose_x_to_y_complex(inputs.world, yz_fft, x_part, y_part, channels, device);
  auto mesh_k = torch::fft::fftn(y_slab, {}, {0});

  auto k_cart = build_local_k_cart(geom.effective_cell, y_part, world_rank, options);
  auto spectral_weights = build_local_spectral_weights(geom.effective_cell, y_part, world_rank, options);
  auto phi_k = mesh_k * spectral_weights.unsqueeze(-1);

  auto imag = torch::complex(
      torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(device)),
      torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat32).device(device)));
  auto grad_k_x = phi_k * (imag * k_cart.select(-1, 0)).unsqueeze(-1);
  auto grad_k_y = phi_k * (imag * k_cart.select(-1, 1)).unsqueeze(-1);
  auto grad_k_z = phi_k * (imag * k_cart.select(-1, 2)).unsqueeze(-1);

  auto stacked_k = torch::cat({phi_k, grad_k_x, grad_k_y, grad_k_z}, -1);
  auto stacked_after_x = torch::fft::ifftn(stacked_k, {}, {0});
  auto stacked_x_complex =
      transpose_y_to_x_complex(inputs.world, stacked_after_x, x_part, y_part, channels * 4, device);
  auto stacked_x_real = torch::view_as_real(torch::fft::ifftn(stacked_x_complex, {}, {1, 2})).select(-1, 0);
  auto local_fields =
      gather_from_xslab_sparse(inputs.world, geom.frac_local, stacked_x_real, x_part, channels * 4, inputs.pbc, device);

  auto local_phi = local_fields.narrow(1, 0, channels);
  auto local_gx = local_fields.narrow(1, channels, channels);
  auto local_gy = local_fields.narrow(1, 2 * channels, channels);
  auto local_gz = local_fields.narrow(1, 3 * channels, channels);

  const double norm = geom.volume * static_cast<double>(mesh_size_ * mesh_size_ * mesh_size_);
  auto force_x = -(local_source * local_gx).sum(-1) / norm;
  auto force_y = -(local_source * local_gy).sum(-1) / norm;
  auto force_z = -(local_source * local_gz).sum(-1) / norm;
  out.forces_local = torch::stack({force_x, force_y, force_z}, 1);

  if (inputs.need_energy) {
    auto local_atom_energy = 0.5 * (local_source * local_phi).sum(-1) / norm;
    out.atom_energy_local = local_atom_energy;
    const double local_energy = local_atom_energy.sum().detach().to(torch::kCPU).item<double>();
    double total_energy = 0.0;
    MPI_Allreduce(&local_energy, &total_energy, 1, MPI_DOUBLE, MPI_SUM, inputs.world);
    out.energy = (world_rank == 0) ? total_energy : 0.0;
  } else {
    out.atom_energy_local = torch::zeros({local_n}, options);
  }
  return out;
}

ReciprocalOutputs MFFReciprocalSolver::compute(const ReciprocalInputs& inputs) const {
  ReciprocalOutputs out;
  const int local_n = static_cast<int>(inputs.local_pos.size(0));
  const int source_channels =
      (inputs.local_source.defined() && inputs.local_source.dim() >= 2) ? static_cast<int>(inputs.local_source.size(1)) : 0;
  torch::Device compute_device = inputs.preferred_device;
  if (inputs.local_source.defined() && inputs.local_source.device().is_cuda() && !compute_device.is_cuda()) {
    compute_device = inputs.local_source.device();
  }
  if (!compute_device.is_cpu() && !compute_device.is_cuda()) compute_device = torch::kCPU;
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
  if (!inputs.local_source.defined() || source_channels <= 0) {
    out.forces_local = torch::zeros({local_n, 3}, options);
    out.atom_energy_local = torch::zeros({local_n}, options);
    out.boundary_mode = resolve_boundary_mode(inputs.pbc);
    out.backend = ReciprocalBackend::Auto;
    return out;
  }

  int world_size = 1;
  MPI_Comm_size(inputs.world, &world_size);
  auto boundary_mode = resolve_boundary_mode(inputs.pbc);
  auto backend = resolve_backend(world_size);
  auto local_source = neutralize_local_source(inputs.world, inputs.local_source, compute_device);
  auto geom = build_effective_geometry(inputs.local_pos, inputs.cell, inputs.pbc, compute_device);

  if (backend == ReciprocalBackend::ReplicatedAtoms) {
    return compute_replicated_atoms(inputs, geom, local_source, compute_device, boundary_mode);
  }
  return compute_mesh_backend(inputs, geom, local_source, compute_device, backend, boundary_mode);
}

ReciprocalOutputs MFFReciprocalSolver::compute(
    MPI_Comm world,
    const torch::Tensor& local_pos,
    const torch::Tensor& local_source,
    const torch::Tensor& cell,
    bool need_energy) const {
  ReciprocalInputs inputs;
  inputs.world = world;
  inputs.local_pos = local_pos;
  inputs.local_source = local_source;
  inputs.cell = cell;
  inputs.need_energy = need_energy;
  inputs.preferred_device = local_source.defined() ? local_source.device() : torch::Device(torch::kCPU);
  return compute(inputs);
}

}  // namespace mfftorch
