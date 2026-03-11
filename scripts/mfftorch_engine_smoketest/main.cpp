#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mff_torch_engine.h"

int main(int argc, const char* argv[]) {
  try {
    if (argc < 2) {
      std::cerr << "用法: mfftorch_engine_smoketest /path/to/core.pt [cpu|cuda]\n";
      return 2;
    }
    const std::string core_path = argv[1];
    const std::string device = (argc >= 3) ? argv[2] : "cpu";

    mfftorch::MFFTorchEngine eng;
    eng.load_core(core_path, device);

    const int64_t nlocal = 32;
    const int64_t ntotal = 32;
    const int64_t E = 256;

    auto A = torch::randint(1, 9, {ntotal}, torch::TensorOptions().dtype(torch::kInt64));
    auto edge_src = torch::randint(0, ntotal, {E}, torch::TensorOptions().dtype(torch::kInt64));
    auto edge_dst = torch::randint(0, ntotal, {E}, torch::TensorOptions().dtype(torch::kInt64));
    auto rij = torch::randn({E, 3}, torch::TensorOptions().dtype(torch::kFloat32));

    auto out = eng.compute(nlocal, ntotal, A, edge_src, edge_dst, rij);

    std::cout << "device=" << (eng.is_cuda() ? "cuda" : "cpu")
              << " E_total=" << out.energy
              << " atom_e_shape=(" << out.atom_energy.size(0) << "," << (out.atom_energy.dim() > 1 ? out.atom_energy.size(1) : 1) << ")"
              << " forces_shape=(" << out.forces.size(0) << "," << out.forces.size(1) << ")"
              << "\n";
    return 0;
  } catch (const c10::Error& e) {
    std::cerr << "c10::Error: " << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}

