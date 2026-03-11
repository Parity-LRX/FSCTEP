#pragma once

#include <array>
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace mfftorch {

enum class ReciprocalBoundaryMode {
  Periodic3D,
  Slab2D,
  Open3D,
};

enum class ReciprocalBackend {
  Auto,
  ReplicatedAtoms,
  MeshReduce,
  DistributedSlabFFT,
  DistributedTreeFmm,
};

enum class ReciprocalGreenMode {
  Poisson,
  LearnedPoisson,
};

struct ReciprocalConfig {
  int mesh_size = 16;
  bool include_k0 = false;
  bool neutralize = true;
  bool gpu_aware_mpi = false;
  double k_norm_floor = 1.0e-6;
  int slab_padding_factor = 2;
  ReciprocalBackend backend = ReciprocalBackend::Auto;
  ReciprocalGreenMode green_mode = ReciprocalGreenMode::Poisson;
};

struct ReciprocalInputs {
  MPI_Comm world = MPI_COMM_WORLD;
  torch::Tensor local_pos;
  torch::Tensor local_source;
  torch::Tensor local_global_ids;
  torch::Tensor cell;
  std::array<int, 3> pbc{{1, 1, 1}};
  bool need_energy = false;
  torch::Device preferred_device{torch::kCPU};
  int world_rank = 0;
  int world_size = 1;
};

struct ReciprocalOutputs {
  double energy = 0.0;
  torch::Tensor forces_local;       // (nlocal, 3) on compute device
  torch::Tensor atom_energy_local;  // (nlocal,) on compute device
  ReciprocalBoundaryMode boundary_mode = ReciprocalBoundaryMode::Periodic3D;
  ReciprocalBackend backend = ReciprocalBackend::Auto;
};

class MFFReciprocalSolver {
 public:
  MFFReciprocalSolver();

  ReciprocalOutputs compute(const ReciprocalInputs& inputs) const;
  ReciprocalOutputs compute(
      MPI_Comm world,
      const torch::Tensor& local_pos,
      const torch::Tensor& local_source,
      const torch::Tensor& cell,
      bool need_energy) const;

  void set_config(const ReciprocalConfig& config) {
    config_ = config;
    mesh_size_ = config_.mesh_size;
    include_k0_ = config_.include_k0;
    neutralize_ = config_.neutralize;
    gpu_aware_mpi_ = config_.gpu_aware_mpi;
    k_norm_floor_ = config_.k_norm_floor;
  }
  const ReciprocalConfig& config() const { return config_; }
  int mesh_size() const { return mesh_size_; }

 private:
  struct AxisPartition {
    std::vector<int> counts;
    std::vector<int> displs;
  };

  struct SpectralCacheKey {
    std::vector<float> effective_cell_values;
    int world_rank = -1;
    std::vector<int> y_counts;
    std::vector<int> y_displs;

    bool matches(
        const torch::Tensor& effective_cell_cpu,
        int rank,
        const AxisPartition& y_part) const;
  };

  struct SparsePartitionCacheKey {
    std::vector<int> x_counts;
    std::vector<int> x_displs;

    bool matches(const AxisPartition& x_part) const {
      return x_counts == x_part.counts && x_displs == x_part.displs;
    }
  };

  struct EffectiveGeometry {
    torch::Tensor effective_cell;
    torch::Tensor frac_local;
    torch::Tensor inv_cell;
    double volume = 0.0;
    ReciprocalBoundaryMode boundary_mode = ReciprocalBoundaryMode::Periodic3D;
  };

  struct SparseCornerContribution {
    int atom_idx = 0;
    int response_slot = 0;
    float weight = 0.0f;
  };

  ReciprocalConfig config_;
  int mesh_size_ = 16;
  bool include_k0_ = false;
  bool neutralize_ = true;
  bool gpu_aware_mpi_ = false;
  double k_norm_floor_ = 1.0e-6;
  mutable torch::Tensor cached_integer_freq_cpu_;
  mutable SpectralCacheKey spectral_cache_key_;
  mutable SparsePartitionCacheKey sparse_partition_cache_key_;
  mutable torch::Tensor cached_local_k_cart_cpu_;
  mutable torch::Tensor cached_local_spectral_weights_cpu_;
  mutable std::vector<int> cached_owner_for_x_;
  mutable std::vector<int64_t> cached_plane_point_displs_;
  mutable std::vector<float> reduce_scatter_recvbuf_;
  mutable std::vector<float> transpose_sendbuf_;
  mutable std::vector<float> transpose_recvbuf_;
  mutable std::vector<int> sparse_request_sendbuf_;
  mutable std::vector<int> sparse_request_recvbuf_;
  mutable std::vector<float> sparse_response_sendbuf_;
  mutable std::vector<float> sparse_response_recvbuf_;

  ReciprocalBackend resolve_backend(int world_size) const;
  ReciprocalBoundaryMode resolve_boundary_mode(const std::array<int, 3>& pbc) const;
  AxisPartition build_axis_partition(int world_size) const;
  void ensure_sparse_partition_cache(const AxisPartition& x_part) const;
  EffectiveGeometry build_effective_geometry(
      const torch::Tensor& local_pos,
      const torch::Tensor& cell,
      const std::array<int, 3>& pbc,
      const torch::Device& device) const;
  torch::Tensor neutralize_local_source(
      MPI_Comm world,
      const torch::Tensor& local_source,
      const torch::Device& device) const;
  torch::Tensor spread_to_mesh_full(
      const torch::Tensor& frac,
      const torch::Tensor& source,
      const std::array<int, 3>& pbc) const;
  torch::Tensor gather_from_mesh_full(
      const torch::Tensor& frac,
      const torch::Tensor& mesh,
      const std::array<int, 3>& pbc) const;
  torch::Tensor reduce_scatter_xslab(
      MPI_Comm world,
      const torch::Tensor& full_mesh_local,
      const AxisPartition& x_part,
      const torch::Device& device) const;
  torch::Tensor gather_from_xslab_sparse(
      MPI_Comm world,
      const torch::Tensor& frac,
      const torch::Tensor& local_x_mesh,
      const AxisPartition& x_part,
      int channels,
      const std::array<int, 3>& pbc,
      const torch::Device& device) const;
  torch::Tensor transpose_x_to_y_complex(
      MPI_Comm world,
      const torch::Tensor& local_x_complex,
      const AxisPartition& x_part,
      const AxisPartition& y_part,
      int channels,
      const torch::Device& device) const;
  torch::Tensor transpose_y_to_x_complex(
      MPI_Comm world,
      const torch::Tensor& local_y_complex,
      const AxisPartition& x_part,
      const AxisPartition& y_part,
      int channels,
      const torch::Device& device) const;
  torch::Tensor build_integer_frequencies(torch::TensorOptions options) const;
  torch::Tensor build_local_k_cart(
      const torch::Tensor& effective_cell,
      const AxisPartition& y_part,
      int world_rank,
      torch::TensorOptions options) const;
  torch::Tensor build_local_spectral_weights(
      const torch::Tensor& effective_cell,
      const AxisPartition& y_part,
      int world_rank,
      torch::TensorOptions options) const;
  ReciprocalOutputs compute_replicated_atoms(
      const ReciprocalInputs& inputs,
      const EffectiveGeometry& geom,
      const torch::Tensor& local_source,
      const torch::Device& device,
      ReciprocalBoundaryMode boundary_mode) const;
  ReciprocalOutputs compute_mesh_backend(
      const ReciprocalInputs& inputs,
      const EffectiveGeometry& geom,
      const torch::Tensor& local_source,
      const torch::Device& device,
      ReciprocalBackend backend,
      ReciprocalBoundaryMode boundary_mode) const;

  static const char* backend_name(ReciprocalBackend backend);
};

}  // namespace mfftorch
