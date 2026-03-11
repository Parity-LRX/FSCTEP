#include "mff_torch_engine.h"

#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfftorch {

namespace {

constexpr int64_t kGlobalPhysWidth = 22;
constexpr int64_t kAtomPhysWidth = 22;
constexpr int64_t kPhysMaskWidth = 4;

bool parse_bool_from_metadata(const std::string& content, const std::string& key, bool& value) {
  const auto key_pos = content.find(key);
  if (key_pos == std::string::npos) return false;
  const auto colon_pos = content.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) return false;
  const auto value_pos = content.find_first_not_of(" \t\r\n", colon_pos + 1);
  if (value_pos == std::string::npos) return false;
  if (content.compare(value_pos, 4, "true") == 0) {
    value = true;
    return true;
  }
  if (content.compare(value_pos, 5, "false") == 0) {
    value = false;
    return true;
  }
  return false;
}

bool parse_int64_from_metadata(const std::string& content, const std::string& key, int64_t& value) {
  const auto key_pos = content.find(key);
  if (key_pos == std::string::npos) return false;
  const auto colon_pos = content.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) return false;
  const auto value_pos = content.find_first_not_of(" \t\r\n", colon_pos + 1);
  if (value_pos == std::string::npos) return false;
  char* end = nullptr;
  const auto parsed = std::strtoll(content.c_str() + value_pos, &end, 10);
  if (end == content.c_str() + value_pos) return false;
  value = static_cast<int64_t>(parsed);
  return true;
}

bool parse_double_from_metadata(const std::string& content, const std::string& key, double& value) {
  const auto key_pos = content.find(key);
  if (key_pos == std::string::npos) return false;
  const auto colon_pos = content.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) return false;
  const auto value_pos = content.find_first_not_of(" \t\r\n", colon_pos + 1);
  if (value_pos == std::string::npos) return false;
  if (content.compare(value_pos, 4, "null") == 0) return false;
  char* end = nullptr;
  const auto parsed = std::strtod(content.c_str() + value_pos, &end);
  if (end == content.c_str() + value_pos) return false;
  value = parsed;
  return true;
}

bool parse_string_from_metadata(const std::string& content, const std::string& key, std::string& value) {
  const auto key_pos = content.find(key);
  if (key_pos == std::string::npos) return false;
  const auto colon_pos = content.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) return false;
  const auto quote_pos = content.find('"', colon_pos + 1);
  if (quote_pos == std::string::npos) return false;
  const auto end_quote = content.find('"', quote_pos + 1);
  if (end_quote == std::string::npos) return false;
  value = content.substr(quote_pos + 1, end_quote - quote_pos - 1);
  return true;
}

bool parse_external_tensor_rank_from_metadata(const std::string& meta_path, bool& requires_external_tensor) {
  std::ifstream in(meta_path);
  if (!in) return false;

  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::string key = "\"external_tensor_rank\"";
  const auto key_pos = content.find(key);
  if (key_pos == std::string::npos) return false;
  const auto colon_pos = content.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) return false;
  const auto value_pos = content.find_first_not_of(" \t\r\n", colon_pos + 1);
  if (value_pos == std::string::npos) return false;

  if (content.compare(value_pos, 4, "null") == 0) {
    requires_external_tensor = false;
    return true;
  }
  if (content[value_pos] >= '0' && content[value_pos] <= '9') {
    requires_external_tensor = true;
    return true;
  }
  return false;
}

bool try_forward_with_external_tensor(torch::jit::script::Module& core,
                                      const torch::Device& device,
                                      const torch::Tensor& cell,
                                      const torch::Tensor& external_tensor,
                                      bool pass_external_tensor) {
  constexpr int64_t N = 4;
  constexpr int64_t E = 8;
  auto A = torch::ones({N}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto batch = torch::zeros({N}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto edge_src = torch::zeros({E}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto edge_dst = torch::zeros({E}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto edge_shifts = torch::zeros({E, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto pos = torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto edge_vec = torch::zeros({E, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

  std::vector<torch::jit::IValue> inputs;
  inputs.reserve(pass_external_tensor ? 9 : 8);
  inputs.push_back(pos);
  inputs.push_back(A);
  inputs.push_back(batch);
  inputs.push_back(edge_src);
  inputs.push_back(edge_dst);
  inputs.push_back(edge_shifts);
  inputs.push_back(cell);
  inputs.push_back(edge_vec);
  if (pass_external_tensor) inputs.push_back(external_tensor);

  try {
    torch::NoGradGuard no_grad;
    (void)core.forward(inputs);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

static int detect_local_gpu_index() {
  // Try common MPI environment variables for local rank.
  const char* env_vars[] = {
    "OMPI_COMM_WORLD_LOCAL_RANK",  // OpenMPI
    "MV2_COMM_WORLD_LOCAL_RANK",   // MVAPICH2
    "MPI_LOCALRANKID",             // Intel MPI
    "SLURM_LOCALID",               // SLURM
    "LOCAL_RANK",                   // PyTorch convention
    nullptr
  };
  for (const char** v = env_vars; *v; ++v) {
    const char* val = std::getenv(*v);
    if (val && val[0] != '\0') {
      int rank = std::atoi(val);
      int n_gpus = static_cast<int>(torch::cuda::device_count());
      if (n_gpus > 0) return rank % n_gpus;
    }
  }
  return 0;
}

static torch::Device pick_device(const std::string& device_str) {
  if (device_str.rfind("cuda:", 0) == 0) {
    int idx = std::atoi(device_str.c_str() + 5);
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("requested " + device_str + " but CUDA is not available");
    }
    c10::cuda::set_device(idx);
    return torch::Device(torch::kCUDA, idx);
  }
  if (device_str == "cuda") {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error("requested device=cuda but torch::cuda::is_available() is false");
    }
    int idx = detect_local_gpu_index();
    c10::cuda::set_device(idx);
    return torch::Device(torch::kCUDA, idx);
  }
  return torch::Device(torch::kCPU);
}

static void ensure_libpython() {
  // CPython extension modules expect Python C API symbols (e.g. PyExc_ValueError)
  // to be provided by the loading process. In a pure-C++ process like LAMMPS,
  // we must dlopen libpython first with RTLD_GLOBAL so those symbols are available.
  static bool done = false;
  if (done) return;
  done = true;

  const char* env = std::getenv("MFF_LIBPYTHON");
  if (env && env[0] != '\0') {
    if (dlopen(env, RTLD_LAZY | RTLD_GLOBAL)) return;
  }
  // Auto-detect: try common libpython names (relies on LD_LIBRARY_PATH).
  const char* names[] = {
    "libpython3.12.so", "libpython3.11.so", "libpython3.10.so",
    "libpython3.12.so.1.0", "libpython3.11.so.1.0", "libpython3.10.so.1.0",
    "libpython3.so",
    nullptr
  };
  for (const char** n = names; *n; ++n) {
    if (dlopen(*n, RTLD_LAZY | RTLD_GLOBAL)) return;
  }
}

static void load_custom_op_libs() {
  const char* env = std::getenv("MFF_CUSTOM_OPS_LIB");
  if (!env || env[0] == '\0') return;

  ensure_libpython();

  std::string paths(env);
  std::string::size_type start = 0;
  while (start < paths.size()) {
    auto pos = paths.find(':', start);
    std::string lib = (pos == std::string::npos)
                          ? paths.substr(start)
                          : paths.substr(start, pos - start);
    start = (pos == std::string::npos) ? paths.size() : pos + 1;
    if (lib.empty()) continue;
    void* handle = dlopen(lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
      throw std::runtime_error(
          std::string("Failed to load custom ops library '") + lib + "': " + dlerror() +
          "\nSet MFF_CUSTOM_OPS_LIB to the path of cuequivariance ops .so"
          "\nIf 'undefined symbol: Py*', also set MFF_LIBPYTHON=/path/to/libpython3.XX.so");
    }
  }
}

void MFFTorchEngine::load_core(const std::string& core_pt_path, const std::string& device_str) {
  device_ = pick_device(device_str);
  load_custom_op_libs();
  core_ = torch::jit::load(core_pt_path, device_);
  core_.eval();

  try {
    core_ = torch::jit::freeze(core_);
  } catch (...) {
    // freeze may fail for some models; proceed without it.
  }

  loaded_ = true;
  cached_ntotal_ = 0;
  cached_nedges_ = 0;
  core_takes_external_tensor_arg_ = false;
  core_requires_external_tensor_ = false;
  core_exports_reciprocal_source_ = false;
  reciprocal_source_channels_ = 0;
  reciprocal_source_boundary_ = "periodic";
  reciprocal_source_slab_padding_factor_ = 2;
  long_range_green_mode_ = "poisson";
  long_range_runtime_backend_ = "none";
  long_range_source_kind_ = "none";
  long_range_source_channels_ = 0;
  long_range_source_layout_ = "none";
  long_range_boundary_ = "nonperiodic";
  long_range_energy_partition_ = "potential";
  long_range_neutralize_ = true;
  long_range_theta_ = 0.5;
  long_range_leaf_size_ = 32;
  long_range_multipole_order_ = 0;
  long_range_screening_ = 0.0;
  long_range_softening_ = 1.0e-6;
  long_range_energy_scale_ = 1.0;

  try {
    auto schema = core_.get_method("forward").function().getSchema();
    size_t nargs = schema.arguments().size();
    if (nargs > 0 && schema.arguments()[0].name() == "self") nargs -= 1;
    core_takes_external_tensor_arg_ = (nargs >= 9);
  } catch (...) {
    // Keep compatibility with older LibTorch builds that may not expose schema details cleanly.
    core_takes_external_tensor_arg_ = false;
  }

  auto probe_cell = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(device_)).unsqueeze(0) * 100.0f;
  {
    std::ifstream in(core_pt_path + ".json");
    if (in) {
      std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      (void)parse_bool_from_metadata(content, "\"export_reciprocal_source\"", core_exports_reciprocal_source_);
      (void)parse_int64_from_metadata(content, "\"reciprocal_source_channels\"", reciprocal_source_channels_);
      (void)parse_string_from_metadata(content, "\"reciprocal_source_boundary\"", reciprocal_source_boundary_);
      (void)parse_int64_from_metadata(
          content, "\"reciprocal_source_slab_padding_factor\"", reciprocal_source_slab_padding_factor_);
      (void)parse_string_from_metadata(content, "\"long_range_green_mode\"", long_range_green_mode_);
      (void)parse_string_from_metadata(content, "\"long_range_runtime_backend\"", long_range_runtime_backend_);
      (void)parse_string_from_metadata(content, "\"long_range_source_kind\"", long_range_source_kind_);
      (void)parse_int64_from_metadata(content, "\"long_range_source_channels\"", long_range_source_channels_);
      (void)parse_string_from_metadata(content, "\"long_range_source_layout\"", long_range_source_layout_);
      (void)parse_string_from_metadata(content, "\"long_range_boundary\"", long_range_boundary_);
      (void)parse_string_from_metadata(content, "\"long_range_energy_partition\"", long_range_energy_partition_);
      (void)parse_bool_from_metadata(content, "\"long_range_neutralize\"", long_range_neutralize_);
      (void)parse_double_from_metadata(content, "\"long_range_theta\"", long_range_theta_);
      (void)parse_int64_from_metadata(content, "\"long_range_leaf_size\"", long_range_leaf_size_);
      (void)parse_int64_from_metadata(content, "\"long_range_multipole_order\"", long_range_multipole_order_);
      (void)parse_double_from_metadata(content, "\"long_range_screening\"", long_range_screening_);
      (void)parse_double_from_metadata(content, "\"long_range_softening\"", long_range_softening_);
      (void)parse_double_from_metadata(content, "\"long_range_energy_scale\"", long_range_energy_scale_);
      if (long_range_source_channels_ <= 0) long_range_source_channels_ = reciprocal_source_channels_;
      if (long_range_runtime_backend_ == "none" && core_exports_reciprocal_source_ && reciprocal_source_channels_ > 0) {
        long_range_runtime_backend_ = "mesh_fft";
      }
    }
  }
  if (core_takes_external_tensor_arg_) {
    bool parsed = parse_external_tensor_rank_from_metadata(core_pt_path + ".json", core_requires_external_tensor_);
    if (!parsed) {
      const auto empty_external = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      const auto rank1_external = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      const auto rank2_external = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
      if (try_forward_with_external_tensor(core_, device_, probe_cell, empty_external, true)) {
        core_requires_external_tensor_ = false;
      } else if (try_forward_with_external_tensor(core_, device_, probe_cell, rank1_external, true) ||
                 try_forward_with_external_tensor(core_, device_, probe_cell, rank2_external, true)) {
        core_requires_external_tensor_ = true;
      } else {
        core_requires_external_tensor_ = false;
      }
    }
  }
}

void MFFTorchEngine::warmup(int64_t N, int64_t E) {
  if (!loaded_) return;

  auto pos = torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  auto A = torch::ones({N}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto edge_src = torch::zeros({E}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto edge_dst = torch::zeros({E}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto edge_shifts = torch::zeros({E, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  auto cell = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(device_)).unsqueeze(0) * 100.0f;
  std::vector<torch::Tensor> warmup_external_tensors;
  if (core_takes_external_tensor_arg_) {
    warmup_external_tensors.push_back(
        torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    warmup_external_tensors.push_back(
        torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    warmup_external_tensors.push_back(
        torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
  } else {
    warmup_external_tensors.push_back(
        torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
  }

  bool warmed = false;
  for (const auto& external_tensor : warmup_external_tensors) {
    try {
      for (int i = 0; i < 3; i++) {
        compute(N, N, pos, A, edge_src, edge_dst, edge_shifts, cell, external_tensor, false);
      }
      warmed = true;
      break;
    } catch (...) {
      // Try the next candidate external tensor shape.
    }
  }
  if (!warmed) throw std::runtime_error("MFFTorchEngine warmup failed for all supported external tensor shapes");
  if (device_.is_cuda()) torch::cuda::synchronize();
}

MFFOutputs MFFTorchEngine::compute(int64_t nlocal, int64_t ntotal,
                                  const torch::Tensor& pos_in,
                                  const torch::Tensor& A_in,
                                  const torch::Tensor& edge_src_in,
                                  const torch::Tensor& edge_dst_in,
                                  const torch::Tensor& edge_shifts_in,
                                  const torch::Tensor& cell_in,
                                  const torch::Tensor& external_tensor_in,
                                  bool need_energy,
                                  bool need_atom_virial) {
  if (!loaded_) throw std::runtime_error("MFFTorchEngine not loaded");
  if (nlocal <= 0 || ntotal <= 0) return {};

  const int64_t nedges = edge_src_in.size(0);

  auto pos0 = (pos_in.device() == device_ && pos_in.dtype() == torch::kFloat32)
                ? pos_in : pos_in.to(device_, torch::kFloat32);
  auto A = (A_in.device() == device_ && A_in.dtype() == torch::kInt64)
               ? A_in : A_in.to(device_, torch::kInt64);
  auto edge_src = (edge_src_in.device() == device_ && edge_src_in.dtype() == torch::kInt64)
                      ? edge_src_in : edge_src_in.to(device_, torch::kInt64);
  auto edge_dst = (edge_dst_in.device() == device_ && edge_dst_in.dtype() == torch::kInt64)
                      ? edge_dst_in : edge_dst_in.to(device_, torch::kInt64);
  auto edge_shifts = (edge_shifts_in.device() == device_ && edge_shifts_in.dtype() == torch::kFloat32)
                       ? edge_shifts_in : edge_shifts_in.to(device_, torch::kFloat32);
  auto cell = (cell_in.device() == device_ && cell_in.dtype() == torch::kFloat32)
                ? cell_in : cell_in.to(device_, torch::kFloat32);
  torch::Tensor external_tensor;
  if (core_takes_external_tensor_arg_) {
    if (core_requires_external_tensor_ && (!external_tensor_in.defined() || external_tensor_in.numel() == 0)) {
      throw std::runtime_error(
          "TorchScript core expects external_tensor, but none was provided by pair_style mff/torch");
    }
    if (!external_tensor_in.defined() || external_tensor_in.numel() == 0) {
      external_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    } else if (external_tensor_in.numel() != 3 && external_tensor_in.numel() != 9) {
      throw std::runtime_error(
          "USER-MFFTORCH currently supports rank-1 (3 values) and rank-2 (9 values) external tensors only");
    } else if (external_tensor_in.numel() == 3) {
      external_tensor = (external_tensor_in.device() == device_ && external_tensor_in.dtype() == torch::kFloat32)
                            ? external_tensor_in.reshape({3})
                            : external_tensor_in.to(device_, torch::kFloat32).reshape({3});
    } else {
      external_tensor = (external_tensor_in.device() == device_ && external_tensor_in.dtype() == torch::kFloat32)
                            ? external_tensor_in.reshape({3, 3})
                            : external_tensor_in.to(device_, torch::kFloat32).reshape({3, 3});
    }
  } else {
    if (external_tensor_in.defined() && external_tensor_in.numel() > 0) {
      throw std::runtime_error(
          "pair_style mff/torch received an external field, but core.pt does not accept external_tensor");
    }
  }

  if (cached_ntotal_ != ntotal) {
    buf_batch_ = torch::zeros({ntotal},
                              torch::TensorOptions().dtype(torch::kInt64).device(device_));
    cached_ntotal_ = ntotal;
  }
  cached_nedges_ = nedges;

  auto pos = pos0.clone().detach().requires_grad_(true);
  auto edge_batch = buf_batch_.index_select(0, edge_src);
  auto edge_cells = cell.index_select(0, edge_batch);
  auto shift_vec = torch::einsum("ni,nij->nj", {edge_shifts, edge_cells});

  // When per-atom virial is requested, also differentiate w.r.t. the periodic
  // edge offset. This yields edge forces without losing the real-space position
  // dependence needed by future long-range modules.
  torch::Tensor shift_leaf;
  if (need_atom_virial) {
    shift_leaf = shift_vec.clone().detach().requires_grad_(true);
  } else {
    shift_leaf = shift_vec;
  }

  auto edge_vec = pos.index_select(0, edge_dst) - pos.index_select(0, edge_src) + shift_leaf;

  std::vector<torch::jit::IValue> inputs;
  inputs.reserve(core_takes_external_tensor_arg_ ? 9 : 8);
  inputs.push_back(pos);
  inputs.push_back(A);
  inputs.push_back(buf_batch_);
  inputs.push_back(edge_src);
  inputs.push_back(edge_dst);
  inputs.push_back(edge_shifts);
  inputs.push_back(cell);
  inputs.push_back(edge_vec);
  if (core_takes_external_tensor_arg_) inputs.push_back(external_tensor);

  auto core_out = core_.forward(inputs);
  torch::Tensor atom_e;
  torch::Tensor global_phys;
  torch::Tensor atom_phys;
  torch::Tensor global_phys_mask;
  torch::Tensor atom_phys_mask;
  torch::Tensor reciprocal_source;
  if (core_out.isTensor()) {
    atom_e = core_out.toTensor();
  } else if (core_out.isTuple()) {
    auto tup = core_out.toTuple();
    const auto& elems = tup->elements();
    if (elems.size() < 5) {
      throw std::runtime_error("TorchScript core returned a tuple, but it does not match the expected physical tensor schema");
    }
    atom_e = elems[0].toTensor();
    global_phys = elems[1].toTensor();
    atom_phys = elems[2].toTensor();
    global_phys_mask = elems[3].toTensor();
    atom_phys_mask = elems[4].toTensor();
    if (elems.size() > 5 && elems[5].isTensor()) {
      reciprocal_source = elems[5].toTensor();
    }
    if (global_phys.defined() && global_phys.numel() > 0 && global_phys.size(-1) != kGlobalPhysWidth) {
      throw std::runtime_error("TorchScript core global_phys last dim must be 22");
    }
    if (atom_phys.defined() && atom_phys.numel() > 0 && atom_phys.size(-1) != kAtomPhysWidth) {
      throw std::runtime_error("TorchScript core atom_phys last dim must be 22");
    }
    if (global_phys_mask.defined() && global_phys_mask.numel() > 0 && global_phys_mask.numel() != kPhysMaskWidth) {
      throw std::runtime_error("TorchScript core global_phys_mask dim must be 4");
    }
    if (atom_phys_mask.defined() && atom_phys_mask.numel() > 0 && atom_phys_mask.numel() != kPhysMaskWidth) {
      throw std::runtime_error("TorchScript core atom_phys_mask dim must be 4");
    }
  } else {
    throw std::runtime_error("TorchScript core returned unsupported type (expected Tensor or tuple)");
  }
  auto atom_e_flat = atom_e.view({atom_e.size(0)});
  auto E_local = atom_e_flat.narrow(0, 0, nlocal).sum();

  // Differentiate w.r.t. pos (forces) and optionally shift_leaf (edge forces).
  std::vector<torch::Tensor> grad_inputs = {pos};
  if (need_atom_virial) grad_inputs.push_back(shift_leaf);

  auto grads = torch::autograd::grad({E_local}, grad_inputs, {}, /*retain_graph=*/false,
                                     /*create_graph=*/false, /*allow_unused=*/true);
  auto forces = -grads[0];

  MFFOutputs out;
  out.atom_energy = atom_e;
  out.forces = forces;
  out.global_phys = global_phys;
  out.atom_phys = atom_phys;
  out.global_phys_mask = global_phys_mask;
  out.atom_phys_mask = atom_phys_mask;
  out.reciprocal_source = reciprocal_source;

  if (need_atom_virial && grads.size() > 1 && grads[1].defined()) {
    auto edge_forces = -grads[1];  // [nedges, 3]

    // Per-edge virial in Voigt order (LAMMPS convention):
    //   v = del ⊗ f_on_src,  where del = x[src]-x[dst] = -rij,  f_on_src = -edge_forces
    //   v = (-rij) ⊗ (-edge_forces) = rij ⊗ edge_forces
    auto r0 = edge_vec.select(1, 0);  // [E]
    auto r1 = edge_vec.select(1, 1);
    auto r2 = edge_vec.select(1, 2);
    auto f0 = edge_forces.select(1, 0);
    auto f1 = edge_forces.select(1, 1);
    auto f2 = edge_forces.select(1, 2);

    auto edge_vir = torch::stack({
        r0 * f0,  // xx
        r1 * f1,  // yy
        r2 * f2,  // zz
        r0 * f1,  // xy
        r0 * f2,  // xz
        r1 * f2,  // yz
    }, 1);  // [nedges, 6]

    // Scatter half to each endpoint (gauge-invariant split).
    auto atom_vir = torch::zeros({ntotal, 6}, edge_vir.options());
    auto half_vir = 0.5f * edge_vir;
    auto src_idx = edge_src.unsqueeze(1).expand_as(half_vir);
    auto dst_idx = edge_dst.unsqueeze(1).expand_as(half_vir);
    atom_vir.scatter_add_(0, src_idx, half_vir);
    atom_vir.scatter_add_(0, dst_idx, half_vir);

    out.atom_virial = atom_vir;
  }

  if (need_energy) {
    out.energy = E_local.detach().to(torch::kCPU).item<double>();
  }
  return out;
}

}  // namespace mfftorch
