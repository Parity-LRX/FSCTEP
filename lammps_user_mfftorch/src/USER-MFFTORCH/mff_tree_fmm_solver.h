#pragma once

#include "mff_reciprocal_solver.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mfftorch {

struct TreeFmmConfig {
  double theta = 0.5;
  int leaf_size = 32;
  int multipole_order = 0;
  bool neutralize = true;
  bool gpu_aware_mpi = false;
  bool device_local_eval = false;
  double screening = 0.0;
  double softening = 1.0e-6;
  double energy_scale = 1.0;
  double reuse_position_tol = 0.0;
  std::string boundary = "nonperiodic";
  std::string energy_partition = "potential";
};

class MFFTreeFmmSolver {
 public:
  struct CachedTreeNode {
    std::vector<int> indices;
    std::vector<std::unique_ptr<CachedTreeNode>> children;
  };

  struct LinearTreeCache {
    torch::Tensor local_global_ids;
    torch::Tensor local_pos;
    torch::Tensor permutation;
    torch::Tensor inverse_permutation;
    torch::Tensor sorted_pos;
    torch::Tensor sorted_source;
    torch::Tensor sorted_global_ids;
    torch::Tensor leaf_offsets;
    torch::Tensor leaf_counts;
    torch::Tensor leaf_centers;
    torch::Tensor leaf_bbox_min;
    torch::Tensor leaf_bbox_max;
    torch::Tensor leaf_half_extents;
    torch::Tensor leaf_charges;
    torch::Tensor leaf_ids;
    int leaf_size = -1;
    bool valid = false;
  };

  struct DistributedRemoteGeometryCache {
    bool valid = false;
    int world_size = 0;
    std::vector<int64_t> leaf_counts;
    std::vector<int64_t> geometry_versions;
    std::vector<torch::Tensor> leaf_ids_by_rank;
    std::vector<torch::Tensor> half_extents_by_rank;
    std::vector<torch::Tensor> centers_by_rank;
  };

  struct DistributedRequestPlanCache {
    bool valid = false;
    int world_size = 0;
    int world_rank = -1;
    int64_t local_geometry_version = -1;
    std::vector<int64_t> remote_geometry_versions;
    std::vector<int64_t> remote_leaf_counts;
    torch::Tensor far_indices;
    torch::Tensor request_leaf_ids;
    torch::Tensor request_ranks;
    torch::Tensor request_counts_per_rank;
    torch::Tensor request_rank_offsets;
  };

  MFFTreeFmmSolver();

  void set_config(const TreeFmmConfig& config);
  const TreeFmmConfig& config() const { return config_; }

  ReciprocalOutputs compute(const ReciprocalInputs& inputs) const;

 private:
  TreeFmmConfig config_;
  mutable torch::Tensor cached_local_global_ids_cpu_;
  mutable torch::Tensor cached_local_pos_cpu_;
  mutable int cached_leaf_size_ = -1;
  mutable std::unique_ptr<CachedTreeNode> cached_tree_topology_;
  mutable LinearTreeCache linear_tree_cache_;
  mutable int64_t local_geometry_version_ = 0;
  mutable DistributedRemoteGeometryCache distributed_remote_geometry_cache_;
  mutable DistributedRequestPlanCache distributed_request_plan_cache_;
};

}  // namespace mfftorch
