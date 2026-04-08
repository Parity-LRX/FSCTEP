#include "pair_mff_torch.h"

#ifdef PAIR_CLASS
#ifdef LMP_KOKKOS
// clang-format off
PairStyle(mff/torch/kk,PairMFFTorchKokkos<LMPDeviceType>);
PairStyle(mff/torch/kk/device,PairMFFTorchKokkos<LMPDeviceType>);
#ifdef LMP_KOKKOS_GPU
PairStyle(mff/torch/kk/host,PairMFFTorchKokkos<LMPHostType>);
#endif
// clang-format on
#endif
#else

#ifndef LMP_PAIR_MFF_TORCH_KOKKOS_H
#define LMP_PAIR_MFF_TORCH_KOKKOS_H

#ifdef LMP_KOKKOS

#include "kokkos_type.h"

#include <array>
#include <torch/torch.h>

namespace LAMMPS_NS {

template <class DeviceType>
class PairMFFTorchKokkos : public PairMFFTorch {
 public:
  PairMFFTorchKokkos(class LAMMPS *lmp);
  ~PairMFFTorchKokkos() override = default;

  void compute(int, int) override;
  void init_style() override;

 private:
  class AtomKokkos *atomKK = nullptr;
  ExecutionSpace execution_space = Host;

  int neighflag = 0;
  int nlocal = 0;
  int nall = 0;

  torch::Tensor type2Z_cuda_;

  // Cached CUDA tensor buffers to avoid per-step allocation.
  int64_t cached_Etotal_ = 0;
  int64_t cached_ntotal_ = 0;
  torch::Tensor buf_edge_src_;
  torch::Tensor buf_edge_dst_;
  torch::Tensor buf_edge_shifts_;
  torch::Tensor buf_pos_;
  torch::Tensor buf_type_idx_;

  // Cached Kokkos views to avoid per-step GPU allocation.
  int cached_inum_ = 0;
  Kokkos::View<int64_t *, DeviceType> cached_d_offsets_;

  // Cached cell tensor and shape preparation state.
  bool cached_cell_valid_ = false;
  std::array<float, 9> cached_cell_values_{};
  torch::Tensor cached_cell_t_;
  int64_t prepared_nlocal_ = -1;
  int64_t prepared_ntotal_ = -1;
  int64_t prepared_nedges_ = -1;
};

}  // namespace LAMMPS_NS

#endif  // LMP_KOKKOS

#endif  // LMP_PAIR_MFF_TORCH_KOKKOS_H
#endif  // PAIR_CLASS
