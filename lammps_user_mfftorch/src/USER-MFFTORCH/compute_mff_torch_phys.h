#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(mff/torch/phys,ComputeMFFTorchPhys);
// clang-format on
#else

#ifndef LMP_COMPUTE_MFF_TORCH_PHYS_H
#define LMP_COMPUTE_MFF_TORCH_PHYS_H

#include "compute.h"

#include <string>
#include <torch/torch.h>

namespace LAMMPS_NS {

class PairMFFTorch;

class ComputeMFFTorchPhys : public Compute {
 public:
  ComputeMFFTorchPhys(class LAMMPS *lmp, int narg, char **arg);
  ~ComputeMFFTorchPhys() override;

  void init() override;
  double compute_scalar() override;
  void compute_vector() override;
  void compute_peratom() override;

 private:
  enum class Mode {
    GLOBAL_VALUES,
    GLOBAL_MASK,
    ATOM_VALUES,
    ATOM_MASK,
  };

  PairMFFTorch *pair_mfftorch_ = nullptr;
  Mode mode_ = Mode::GLOBAL_VALUES;
  int nmax_atom_ = 0;
  int selection_offset_ = 0;
  int selection_length_ = 0;
  bool use_scalar_output_ = false;
  bool use_peratom_vector_output_ = false;

  void parse_mode(const std::string &mode);
  void parse_selection(int narg, char **arg);
  void require_current_cache() const;
  void copy_global_tensor_to_scalar(const torch::Tensor &src, int total_cols);
  void copy_global_tensor_to_vector(const torch::Tensor &src, int total_cols);
  void copy_atom_tensor_to_vector(const torch::Tensor &src, int total_cols);
  void copy_atom_tensor_to_array(const torch::Tensor &src, int total_cols);
};

}  // namespace LAMMPS_NS

#endif  // LMP_COMPUTE_MFF_TORCH_PHYS_H
#endif  // COMPUTE_CLASS
