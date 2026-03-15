#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

namespace mfftorch {

struct MFFOutputs {
  double energy = 0.0;
  torch::Tensor atom_energy;   // (ntotal,1) or (ntotal,) on engine device
  torch::Tensor forces;        // (ntotal,3) on engine device
  torch::Tensor atom_virial;   // (ntotal,6) on engine device — Voigt: xx,yy,zz,xy,xz,yz
  torch::Tensor global_phys;   // (n_graphs, 22) on engine device
  torch::Tensor atom_phys;     // (ntotal, 31) on engine device
  torch::Tensor global_phys_mask;  // (5,) on engine device
  torch::Tensor atom_phys_mask;    // (5,) on engine device
  torch::Tensor reciprocal_source; // (ntotal, C_lr) on engine device
};

class MFFTorchEngine {
 public:
  MFFTorchEngine() = default;

  void load_core(const std::string& core_pt_path, const std::string& device_str);

  // Warmup: run one dummy forward+backward to trigger JIT compilation and CUDA caching.
  void warmup(int64_t N = 32, int64_t E = 256);

  const torch::Device& device() const { return device_; }
  bool is_cuda() const { return device_.is_cuda(); }
  bool accepts_external_tensor() const { return core_requires_external_tensor_; }
  const std::string& external_tensor_irrep() const { return external_tensor_irrep_; }
  int64_t external_tensor_total_numel() const { return external_tensor_total_numel_; }
  bool external_tensor_has_field_1o() const { return external_tensor_has_field_1o_; }
  bool external_tensor_has_field_1e() const { return external_tensor_has_field_1e_; }
  bool exports_reciprocal_source() const { return core_exports_reciprocal_source_; }
  bool takes_fidelity_arg() const { return core_takes_fidelity_arg_; }
  bool requires_runtime_fidelity() const { return core_requires_runtime_fidelity_; }
  int64_t num_fidelity_levels() const { return num_fidelity_levels_; }
  int64_t export_fidelity_id() const { return export_fidelity_id_; }
  int64_t reciprocal_source_channels() const { return reciprocal_source_channels_; }
  const std::string& reciprocal_source_boundary() const { return reciprocal_source_boundary_; }
  int64_t reciprocal_source_slab_padding_factor() const { return reciprocal_source_slab_padding_factor_; }
  const std::string& long_range_green_mode() const { return long_range_green_mode_; }
  const std::string& long_range_runtime_backend() const { return long_range_runtime_backend_; }
  const std::string& long_range_source_kind() const { return long_range_source_kind_; }
  int64_t long_range_source_channels() const { return long_range_source_channels_; }
  const std::string& long_range_source_layout() const { return long_range_source_layout_; }
  const std::string& long_range_boundary() const { return long_range_boundary_; }
  const std::string& long_range_energy_partition() const { return long_range_energy_partition_; }
  bool long_range_neutralize() const { return long_range_neutralize_; }
  double long_range_theta() const { return long_range_theta_; }
  int64_t long_range_leaf_size() const { return long_range_leaf_size_; }
  int64_t long_range_multipole_order() const { return long_range_multipole_order_; }
  double long_range_screening() const { return long_range_screening_; }
  double long_range_softening() const { return long_range_softening_; }
  double long_range_energy_scale() const { return long_range_energy_scale_; }

  MFFOutputs compute(int64_t nlocal, int64_t ntotal,
                     const torch::Tensor& pos,
                     const torch::Tensor& A,
                     const torch::Tensor& edge_src,
                     const torch::Tensor& edge_dst,
                     const torch::Tensor& edge_shifts,
                     const torch::Tensor& cell,
                     const torch::Tensor& external_tensor = torch::Tensor(),
                     const torch::Tensor& fidelity_ids = torch::Tensor(),
                     bool need_energy = true,
                     bool need_atom_virial = false);

 private:
  torch::jit::script::Module core_;
  bool loaded_ = false;
  bool core_takes_external_tensor_arg_ = false;
  bool core_requires_external_tensor_ = false;
  bool core_takes_fidelity_arg_ = false;
  bool core_requires_runtime_fidelity_ = false;
  std::string external_tensor_irrep_;
  int64_t external_tensor_total_numel_ = 0;
  int64_t num_fidelity_levels_ = 0;
  int64_t export_fidelity_id_ = -1;
  bool external_tensor_has_field_1o_ = false;
  bool external_tensor_has_field_1e_ = false;
  bool core_exports_reciprocal_source_ = false;
  int64_t reciprocal_source_channels_ = 0;
  std::string reciprocal_source_boundary_ = "periodic";
  int64_t reciprocal_source_slab_padding_factor_ = 2;
  std::string long_range_green_mode_ = "poisson";
  std::string long_range_runtime_backend_ = "none";
  std::string long_range_source_kind_ = "none";
  int64_t long_range_source_channels_ = 0;
  std::string long_range_source_layout_ = "none";
  std::string long_range_boundary_ = "nonperiodic";
  std::string long_range_energy_partition_ = "potential";
  bool long_range_neutralize_ = true;
  double long_range_theta_ = 0.5;
  int64_t long_range_leaf_size_ = 32;
  int64_t long_range_multipole_order_ = 0;
  double long_range_screening_ = 0.0;
  double long_range_softening_ = 1.0e-6;
  double long_range_energy_scale_ = 1.0;

  torch::Device device_{torch::kCPU};

  // Reusable per-step buffers (avoid repeated CUDA malloc).
  int64_t cached_ntotal_ = 0;
  int64_t cached_nedges_ = 0;
  torch::Tensor buf_batch_;
};

}  // namespace mfftorch
