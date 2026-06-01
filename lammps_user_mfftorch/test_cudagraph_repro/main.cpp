// Standalone reproducer for the mff/torch CUDA-graph path (MFF_CUDA_GRAPH=1),
// independent of LAMMPS. Links only LibTorch + the engine TU. Drives
// MFFTorchEngine::compute() with fixed-shape inputs across several steps so the
// engine attempts capture (first call) then replay (subsequent calls).
//
// Usage: mff_cg_repro <core.pt> [device=cuda] [ntotal=64] [nedges=512] [steps=4]
//   MFF_CUDA_GRAPH=1  -> exercise the capture/replay path
//   (unset)           -> eager path (baseline)
#include "mff_torch_engine.h"

#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace mfftorch;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <core.pt> [device] [ntotal] [nedges] [steps]\n", argv[0]);
    return 2;
  }
  const std::string core_pt = argv[1];
  const std::string device_str = (argc > 2) ? argv[2] : "cuda";
  const int64_t ntotal = (argc > 3) ? std::atoll(argv[3]) : 64;
  const int64_t nlocal = ntotal;
  const int64_t nedges = (argc > 4) ? std::atoll(argv[4]) : 512;
  const int steps = (argc > 5) ? std::atoi(argv[5]) : 4;

  const char* cg = std::getenv("MFF_CUDA_GRAPH");
  std::printf("[repro] core=%s device=%s ntotal=%lld nedges=%lld steps=%d MFF_CUDA_GRAPH=%s\n",
              core_pt.c_str(), device_str.c_str(), (long long)ntotal, (long long)nedges, steps,
              cg ? cg : "(unset)");
  std::fflush(stdout);

  try {
    MFFTorchEngine engine;
    engine.load_core(core_pt, device_str);
    std::printf("[repro] load_core OK; is_cuda=%d tp_mode=%s\n",
                (int)engine.is_cuda(), engine.tensor_product_mode().c_str());
    std::fflush(stdout);

    torch::Device dev(device_str == "cuda" ? torch::kCUDA : torch::kCPU);
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto iopt = torch::TensorOptions().dtype(torch::kInt64).device(dev);

    // Fixed-shape, deterministic inputs (H atoms, species id 1; big box -> no PBC wrap).
    torch::manual_seed(0);
    auto pos = torch::randn({ntotal, 3}, fopt) * 2.5f;
    auto A = torch::ones({ntotal}, iopt);  // all H (Z=1); valid for an H/O dummy core
    auto edge_src = torch::randint(0, ntotal, {nedges}, iopt);
    auto edge_dst = torch::randint(0, ntotal, {nedges}, iopt);
    auto edge_shifts = torch::zeros({nedges, 3}, fopt);
    auto cell = torch::eye(3, fopt).unsqueeze(0) * 100.0f;

    for (int s = 0; s < steps; ++s) {
      // Jitter positions each step so replay isn't trivially identical data.
      auto pos_s = pos + 0.01f * torch::randn({ntotal, 3}, fopt);
      auto out = engine.compute(nlocal, ntotal, pos_s, A, edge_src, edge_dst,
                                edge_shifts, cell, torch::Tensor(), torch::Tensor(),
                                /*need_energy=*/true, /*need_atom_virial=*/false);
      double fnorm = out.forces.defined() ? out.forces.norm().item<double>() : -1.0;
      std::printf("[repro] step %d  energy=%.6f  |F|=%.6f\n", s, out.energy, fnorm);
      std::fflush(stdout);
    }
    std::printf("[repro] DONE\n");
    return 0;
  } catch (const std::exception& e) {
    std::printf("[repro] EXCEPTION: %s\n", e.what());
    std::fflush(stdout);
    return 1;
  }
}
