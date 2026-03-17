#ifdef PAIR_CLASS
// clang-format off
PairStyle(mff/torch,PairMFFTorch);
// clang-format on
#else

#ifndef LMP_PAIR_MFF_TORCH_H
#define LMP_PAIR_MFF_TORCH_H

#include "pair.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace mfftorch {
class MFFTorchEngine;
class MFFReciprocalSolver;
class MFFTreeFmmSolver;
struct MFFOutputs;
}

namespace LAMMPS_NS {

class PairMFFTorch : public Pair {
 public:
  PairMFFTorch(class LAMMPS *lmp);
  ~PairMFFTorch() override;

  void compute(int eflag, int vflag) override;
  void settings(int narg, char **arg) override;
  void coeff(int narg, char **arg) override;
  double init_one(int i, int j) override;
  void init_style() override;
  void set_physical_cache_requested(bool requested) { physical_cache_requested_ = requested; }
  const torch::Tensor& global_phys() const { return global_phys_cpu_; }
  const torch::Tensor& atom_phys() const { return atom_phys_cpu_; }
  const torch::Tensor& global_phys_mask() const { return global_phys_mask_cpu_; }
  const torch::Tensor& atom_phys_mask() const { return atom_phys_mask_cpu_; }
  int64_t cached_phys_timestep() const { return cached_phys_timestep_; }

 protected:
  void allocate();
  torch::Tensor current_external_tensor(const torch::Device& device);
  void validate_external_field_configuration();
  void cache_physical_outputs(const mfftorch::MFFOutputs& out, int nlocal);
  void reset_physical_outputs();
  torch::Tensor current_fidelity_tensor(const torch::Device& device);

  double cut_global_ = 0.0;
  double cutsq_global_ = 0.0;

  std::string device_str_ = "cuda";
  std::string core_pt_path_;

  std::vector<int64_t> type2Z_;
  bool use_external_field_ = false;
  bool use_electric_field_ = false;
  bool use_magnetic_field_ = false;
  bool use_rank2_external_field_ = false;
  bool external_field_symmetric_rank2_ = false;
  bool use_fidelity_input_ = false;
  bool fidelity_is_variable_ = false;
  std::string fidelity_var_name_;
  int64_t fidelity_constant_ = 0;
  std::vector<std::string> electric_field_var_names_;
  std::vector<std::string> magnetic_field_var_names_;
  std::vector<std::string> rank2_external_field_var_names_;
  std::vector<float> cached_external_field_values_;
  torch::Tensor external_tensor_cache_;
  torch::Tensor global_phys_cpu_;
  torch::Tensor atom_phys_cpu_;
  torch::Tensor global_phys_mask_cpu_;
  torch::Tensor atom_phys_mask_cpu_;
  int64_t cached_phys_timestep_ = -1;
  bool physical_cache_requested_ = false;

  std::unique_ptr<mfftorch::MFFTorchEngine> engine_;
  std::unique_ptr<mfftorch::MFFReciprocalSolver> reciprocal_solver_;
  std::unique_ptr<mfftorch::MFFTreeFmmSolver> tree_fmm_solver_;
  bool engine_loaded_ = false;

  // Persistent per-step CPU buffers (avoid repeated heap allocation).
  std::vector<int64_t> buf_A_cpu_;
  std::vector<float> buf_pos_cpu_;
  std::vector<int64_t> buf_edge_src_cpu_;
  std::vector<int64_t> buf_edge_dst_cpu_;
  std::vector<float> buf_edge_shifts_cpu_;

  // Persistent torch tensors (avoid from_blob().clone() every step).
  int64_t cached_compute_ntotal_ = 0;
  int64_t cached_compute_nedges_ = 0;
  torch::Tensor cached_pos_t_;
  torch::Tensor cached_A_t_;
  torch::Tensor cached_edge_src_t_;
  torch::Tensor cached_edge_dst_t_;
  torch::Tensor cached_edge_shifts_t_;
  torch::Tensor cached_cell_t_;
};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_MFF_TORCH_H
#endif  // PAIR_CLASS
