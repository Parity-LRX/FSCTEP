#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#if __has_include(<ATen/cuda/CUDAGraph.h>)
#include <ATen/cuda/CUDAGraph.h>
#define MFF_HAS_CUDA_GRAPH 1
#else
#define MFF_HAS_CUDA_GRAPH 0
#endif

// AOTInductor .pt2 loader (an inference-only Inductor-compiled model). Present in
// torch >= 2.6. Lets pair_style mff/torch load an AOTI .pt2 (with the force traced
// INTO the graph) instead of a TorchScript .pt. The .pt2 path is simpler than the
// TorchScript path: it skips the C++-side edge_vec compute and the C++ autograd,
// calling .run() -> (atom_energy, force) directly.
#if __has_include(<torch/csrc/inductor/aoti_package/model_package_loader.h>)
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#define MFF_HAS_AOTI 1
#else
#define MFF_HAS_AOTI 0
#endif

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
  void prepare_for_shape(int64_t nlocal, int64_t ntotal, int64_t nedges);

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
  int64_t long_range_mesh_size() const { return long_range_mesh_size_; }
  int64_t long_range_max_multipole_l() const { return long_range_max_multipole_l_; }
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
  const std::string& tensor_product_mode() const { return tensor_product_mode_; }
  bool prefers_kokkos_host_staging() const { return tensor_product_mode_ == "spherical-save-cue"; }
  bool is_bundle_manifest() const { return bundle_mode_; }

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
  struct BucketSpec {
    std::string name;
    std::string core_path;
    int64_t max_nodes = 0;
    int64_t max_edges = 0;
    int64_t trace_num_nodes = 0;
    int64_t trace_num_edges = 0;
    std::string dtype;
    std::string jit_mode;
  };

  void load_single_core_file(const std::string& core_pt_path);
  // AOTI inference path: .pt2 returns (atom_energy, force) with force already in-graph,
  // so no C++ edge_vec compute and no C++ autograd are needed (unlike run_forward_backward).
  MFFOutputs run_aoti(const torch::Tensor& pos0, const torch::Tensor& A,
                      const torch::Tensor& edge_src, const torch::Tensor& edge_dst,
                      const torch::Tensor& edge_shifts, const torch::Tensor& cell);
  void ensure_core_for_shape(int64_t nlocal, int64_t ntotal, int64_t nedges, bool warmup_on_switch);

  torch::jit::script::Module core_;
#if MFF_HAS_AOTI
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> aoti_loader_;
#endif
  bool aoti_mode_ = false;   // true when core was loaded from an AOTI .pt2 (force in-graph)
  // AOTI .pt2 bakes the atom count N. Pad ntotal up to aoti_nmax_ each step (dummy atoms = valid
  // species, no edges -> inert) and slice the first ntotal outputs back; when ntotal exceeds
  // aoti_nmax_ (e.g. a ghost-count spike), fall back to the N-flexible TorchScript core_. Read from a
  // sidecar "<core>.pt2.meta" (nmax / pad_z / fallback). aoti_nmax_==0 -> no padding (legacy .pt2).
  int64_t aoti_nmax_ = 0;
  int64_t aoti_pad_z_ = 1;          // atomic number for dummy padding atoms (must be a valid embedding Z)
  bool have_ts_fallback_ = false;   // core_ holds an N-flexible TorchScript core for ntotal > nmax_
  bool aoti_fallback_warned_ = false;
  bool loaded_ = false;
  bool bundle_mode_ = false;
  std::string bundle_manifest_path_;
  std::vector<BucketSpec> bundle_buckets_;
  int current_bucket_index_ = -1;
  bool bundle_warned_oversize_ = false;
  bool warming_up_ = false;
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
  int64_t long_range_mesh_size_ = 16;
  int64_t long_range_max_multipole_l_ = 0;
  std::string tensor_product_mode_;
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
  int64_t trace_num_nodes_ = 0;
  int64_t trace_num_edges_ = 0;

  torch::Device device_{torch::kCPU};

  // Reusable per-step buffers (avoid repeated CUDA malloc).
  int64_t cached_ntotal_ = 0;
  int64_t cached_nedges_ = 0;
  torch::Tensor buf_batch_;

  // Cached intermediate buffers to avoid per-step allocation in compute().
  torch::Tensor buf_edge_shifts_fp32_;

  // CUDA Graph replay support (MFF_CUDA_GRAPH=1 to enable).
  bool use_cuda_graph_ = false;
#if MFF_HAS_CUDA_GRAPH
  struct CUDAGraphCache {
    bool valid = false;
    int64_t ntotal = 0;
    int64_t nedges = 0;
    int64_t nlocal = 0;
    bool need_atom_virial = false;

    // Pre-allocated input buffers whose data is overwritten each step.
    torch::Tensor pos_in;
    torch::Tensor A_in;
    torch::Tensor edge_src_in;
    torch::Tensor edge_dst_in;
    torch::Tensor edge_shifts_in;
    torch::Tensor cell_in;
    torch::Tensor external_tensor_in;
    torch::Tensor fidelity_ids_in;

    // Captured output references (addresses fixed across replays).
    torch::Tensor forces_out;
    torch::Tensor atom_e_out;
    torch::Tensor E_local_out;
    torch::Tensor global_phys_out;
    torch::Tensor atom_phys_out;
    torch::Tensor global_phys_mask_out;
    torch::Tensor atom_phys_mask_out;
    torch::Tensor reciprocal_source_out;
    torch::Tensor atom_vir_out;
    torch::Tensor shift_leaf_out;

    at::cuda::CUDAGraph graph;
    c10::cuda::CUDAStream capture_stream{c10::cuda::getStreamFromPool()};
  };
  CUDAGraphCache cg_cache_;

  MFFOutputs compute_with_cuda_graph(
      int64_t nlocal, int64_t ntotal,
      const torch::Tensor& pos, const torch::Tensor& A,
      const torch::Tensor& edge_src, const torch::Tensor& edge_dst,
      const torch::Tensor& edge_shifts, const torch::Tensor& cell,
      const torch::Tensor& external_tensor, const torch::Tensor& fidelity_ids,
      bool need_energy, bool need_atom_virial);

  void capture_cuda_graph(
      int64_t nlocal, int64_t ntotal, int64_t nedges,
      bool need_atom_virial);
#endif

  MFFOutputs run_forward_backward(
      const torch::Tensor& pos0, const torch::Tensor& A,
      const torch::Tensor& edge_src, const torch::Tensor& edge_dst,
      const torch::Tensor& edge_shifts, const torch::Tensor& cell,
      const torch::Tensor& external_tensor, const torch::Tensor& fidelity_ids,
      int64_t nlocal, int64_t ntotal, bool need_energy, bool need_atom_virial);
};

}  // namespace mfftorch
