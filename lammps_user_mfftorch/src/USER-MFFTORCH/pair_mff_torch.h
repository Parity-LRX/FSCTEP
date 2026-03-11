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

  double cut_global_ = 0.0;
  double cutsq_global_ = 0.0;

  std::string device_str_ = "cuda";
  std::string core_pt_path_;

  std::vector<int64_t> type2Z_;
  bool use_external_field_ = false;
  int external_tensor_rank_ = 0;
  bool external_field_symmetric_rank2_ = false;
  std::vector<std::string> external_field_var_names_;
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
};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_MFF_TORCH_H
#endif  // PAIR_CLASS
