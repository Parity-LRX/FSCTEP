#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>

namespace {

struct Args {
  std::string model_path;
  std::string device = "cpu"; // "cpu" or "cuda"
  int64_t N = 512;
  int64_t E = 16384;
  int warmup = 5;
  int iters = 20;
  uint64_t seed = 0;
};

[[noreturn]] void die(const std::string& msg) { throw std::runtime_error(msg); }

Args parse_args(int argc, const char* argv[]) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k(argv[i]);
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) die(std::string("缺少参数值: ") + name);
      return std::string(argv[++i]);
    };

    if (k == "--model") a.model_path = need("--model");
    else if (k == "--device") a.device = need("--device");
    else if (k == "--N") a.N = std::stoll(need("--N"));
    else if (k == "--E") a.E = std::stoll(need("--E"));
    else if (k == "--warmup") a.warmup = std::stoi(need("--warmup"));
    else if (k == "--iters") a.iters = std::stoi(need("--iters"));
    else if (k == "--seed") a.seed = static_cast<uint64_t>(std::stoull(need("--seed")));
    else if (k == "--help" || k == "-h") {
      std::cout
          << "用法:\n"
          << "  libtorch_smoketest --model core.pt [--device cpu|cuda] [--N 512] [--E 16384]\n"
          << "                   [--warmup 5] [--iters 20] [--seed 0]\n";
      std::exit(0);
    } else {
      die("未知参数: " + k);
    }
  }
  if (a.model_path.empty()) die("必须提供 --model <core.pt>");
  return a;
}

torch::Device pick_device(const std::string& req) {
  if (req == "cuda") {
    if (!torch::cuda::is_available()) {
      std::cerr << "WARN: CUDA 不可用，改用 CPU\n";
      return torch::Device(torch::kCPU);
    }
    return torch::Device(torch::kCUDA);
  }
  return torch::Device(torch::kCPU);
}

double ms_since(const std::chrono::steady_clock::time_point& t0) {
  using namespace std::chrono;
  return duration_cast<duration<double, std::milli>>(steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, const char* argv[]) {
  try {
    const auto args = parse_args(argc, argv);
    const auto device = pick_device(args.device);

    if (args.seed != 0) {
      torch::manual_seed(static_cast<uint64_t>(args.seed));
      if (device.is_cuda()) torch::cuda::manual_seed_all(static_cast<uint64_t>(args.seed));
    }

    torch::jit::script::Module core = torch::jit::load(args.model_path, device);
    core.eval();

    // 该 TorchScript core 的签名是：
    // (pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec) -> atom_energies
    // 我们在 C++ 里复现 AtomForcesWrapper 的关键逻辑：用 dE/d(pos) 得到 per-atom forces。
    const int64_t N = args.N;
    const int64_t E = args.E;

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto lopt = torch::TensorOptions().dtype(torch::kInt64).device(device);

    // 输入张量
    auto pos = torch::zeros({N, 3}, fopt).set_requires_grad(true);
    auto A = torch::randint(/*low=*/1, /*high=*/9, {N}, lopt); // 原子序数 Z（示例：1..8）
    auto batch = torch::zeros({N}, lopt);
    auto edge_src = torch::randint(/*low=*/0, /*high=*/N, {E}, lopt);
    auto edge_dst = torch::randint(/*low=*/0, /*high=*/N, {E}, lopt);
    auto edge_shifts = torch::zeros({E, 3}, fopt);
    auto cell = (torch::eye(3, fopt).unsqueeze(0) * 100.0f);
    auto rij = torch::randn({E, 3}, fopt);

    auto run_once = [&]() {
      // 注意：edge_vec 必须依赖 pos，才能把梯度累积到 pos 上（per-atom forces）
      auto edge_vec = pos.index_select(0, edge_dst) - pos.index_select(0, edge_src) + rij;
      auto atom_e_iv = core.forward({pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec});
      auto atom_e = atom_e_iv.toTensor();
      auto E_total = atom_e.sum();
      auto grads = torch::autograd::grad({E_total}, {pos});
      auto forces = -grads[0];
      // 返回少量信息用于 sanity check，避免编译器过度优化
      return std::make_pair(E_total, forces);
    };

    // Warmup
    for (int i = 0; i < args.warmup; i++) {
      auto out = run_once();
      if (device.is_cuda()) torch::cuda::synchronize();
      (void)out;
    }

    // Benchmark
    double total_ms = 0.0;
    torch::Tensor last_E;
    torch::Tensor last_F;
    for (int i = 0; i < args.iters; i++) {
      const auto t0 = std::chrono::steady_clock::now();
      auto out = run_once();
      if (device.is_cuda()) torch::cuda::synchronize();
      const auto dt = ms_since(t0);
      total_ms += dt;
      last_E = out.first;
      last_F = out.second;
    }

    // Sanity
    const auto E_cpu = last_E.detach().to(torch::kCPU);
    const auto F_cpu = last_F.detach().to(torch::kCPU);
    const double E_val = E_cpu.item<double>();
    const double F_norm = F_cpu.norm().item<double>();

    std::cout << "device=" << (device.is_cuda() ? "cuda" : "cpu")
              << " N=" << N << " E=" << E
              << " iters=" << args.iters
              << " avg=" << (total_ms / args.iters) << " ms/iter"
              << " E_total=" << E_val
              << " |F|=" << F_norm
              << "\n";

    return 0;
  } catch (const c10::Error& e) {
    std::cerr << "c10::Error: " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}

