#include "mff_torch_engine.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfftorch {

namespace {

constexpr int64_t kGlobalPhysWidth = 22;
constexpr int64_t kAtomPhysWidth = 31;
constexpr int64_t kPhysMaskWidth = 5;

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

std::string trim_copy(const std::string& s) {
  const auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return std::string();
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

bool is_directory_path(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_directory(std::filesystem::path(path), ec);
}

std::string read_text_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool extract_json_array_block(const std::string& content, const std::string& key, std::string& array_block) {
  const auto key_pos = content.find(key);
  if (key_pos == std::string::npos) return false;
  const auto colon_pos = content.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) return false;
  const auto bracket_pos = content.find('[', colon_pos + 1);
  if (bracket_pos == std::string::npos) return false;
  int depth = 0;
  for (size_t i = bracket_pos; i < content.size(); ++i) {
    const char ch = content[i];
    if (ch == '[') {
      depth += 1;
    } else if (ch == ']') {
      depth -= 1;
      if (depth == 0) {
        array_block = content.substr(bracket_pos + 1, i - bracket_pos - 1);
        return true;
      }
    }
  }
  return false;
}

std::vector<std::string> split_top_level_object_blocks(const std::string& array_block) {
  std::vector<std::string> out;
  int depth = 0;
  size_t obj_start = std::string::npos;
  for (size_t i = 0; i < array_block.size(); ++i) {
    const char ch = array_block[i];
    if (ch == '{') {
      if (depth == 0) obj_start = i;
      depth += 1;
    } else if (ch == '}') {
      depth -= 1;
      if (depth == 0 && obj_start != std::string::npos) {
        out.push_back(array_block.substr(obj_start, i - obj_start + 1));
        obj_start = std::string::npos;
      }
    }
  }
  return out;
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

bool parse_external_tensor_irrep_from_metadata(const std::string& meta_path, std::string& irrep) {
  std::ifstream in(meta_path);
  if (!in) return false;
  std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return parse_string_from_metadata(content, "\"external_tensor_irrep\"", irrep);
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

bool should_use_flat_external_tensor(
    int64_t external_tensor_total_numel,
    bool external_tensor_has_field_1o,
    bool external_tensor_has_field_1e) {
  if (external_tensor_total_numel <= 0) return false;
  const bool has_rank1_field = external_tensor_has_field_1o || external_tensor_has_field_1e;
  if (!has_rank1_field) return false;
  if (external_tensor_total_numel == 3) return true;
  return true;
}

void maybe_dump_forward_inputs_once(const std::vector<torch::jit::IValue>& inputs) {
  static bool done = false;
  if (done) return;
  const char* prefix = std::getenv("MFF_DUMP_INPUT_PREFIX");
  if (!prefix || prefix[0] == '\0') return;
  if (inputs.size() >= 4 && inputs[0].isTensor() && inputs[3].isTensor()) {
    const auto n = inputs[0].toTensor().size(0);
    const auto e = inputs[3].toTensor().size(0);
    if (n <= 32 && e <= 256) return;
  }
  done = true;

  auto dump_tensor = [&](size_t idx, const char* name) {
    if (idx >= inputs.size() || !inputs[idx].isTensor()) return;
    torch::Tensor t = inputs[idx].toTensor().detach().cpu();
    torch::save(t, std::string(prefix) + "_" + name + ".pt");
  };

  dump_tensor(0, "pos");
  dump_tensor(1, "A");
  dump_tensor(2, "batch");
  dump_tensor(3, "edge_src");
  dump_tensor(4, "edge_dst");
  dump_tensor(5, "edge_shifts");
  dump_tensor(6, "cell");
  dump_tensor(7, "edge_vec");
  dump_tensor(8, "external");
  dump_tensor(9, "fidelity");
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

static void ensure_python_custom_op_registrations() {
  static bool done = false;
  if (done) return;
  done = true;

  const char* env = std::getenv("MFF_CUSTOM_OPS_LIB");
  if (!env || env[0] == '\0') return;

  ensure_libpython();

  using Py_InitializeEx_Fn = void (*)(int);
  using Py_IsInitialized_Fn = int (*)();
  using PyGILState_Ensure_Fn = int (*)();
  using PyGILState_Release_Fn = void (*)(int);
  using PyRun_SimpleString_Fn = int (*)(const char*);

  auto py_is_initialized = reinterpret_cast<Py_IsInitialized_Fn>(dlsym(RTLD_DEFAULT, "Py_IsInitialized"));
  auto py_initialize_ex = reinterpret_cast<Py_InitializeEx_Fn>(dlsym(RTLD_DEFAULT, "Py_InitializeEx"));
  auto py_gil_ensure = reinterpret_cast<PyGILState_Ensure_Fn>(dlsym(RTLD_DEFAULT, "PyGILState_Ensure"));
  auto py_gil_release = reinterpret_cast<PyGILState_Release_Fn>(dlsym(RTLD_DEFAULT, "PyGILState_Release"));
  auto py_run_simple_string =
      reinterpret_cast<PyRun_SimpleString_Fn>(dlsym(RTLD_DEFAULT, "PyRun_SimpleString"));

  if (!py_is_initialized || !py_initialize_ex || !py_gil_ensure || !py_gil_release ||
      !py_run_simple_string) {
    throw std::runtime_error(
        "Failed to resolve CPython runtime symbols needed for cue custom op registration");
  }

  if (!py_is_initialized()) {
    py_initialize_ex(0);
  }

  const int gil_state = py_gil_ensure();
  const char* code = "import cuequivariance_ops_torch.tensor_product_uniform_1d_jit\n";
  const int rc = py_run_simple_string(code);
  py_gil_release(gil_state);

  if (rc != 0) {
    throw std::runtime_error(
        "Failed to import cuequivariance_ops_torch.tensor_product_uniform_1d_jit "
        "while registering Torch custom ops"
        "\nFor native cue ops, also set:"
        "\n  PYTHONHOME=/path/to/python/env"
        "\n  PYTHONPATH=/path/to/python/env/lib/pythonX.Y/site-packages[:... ]"
        "\n  MFF_LIBPYTHON=/path/to/libpythonX.Y.so");
  }
}

struct OptionalGilRelease {
  using Py_IsInitialized_Fn = int (*)();
  using PyGILState_Check_Fn = int (*)();
  using PyEval_SaveThread_Fn = void* (*)();
  using PyEval_RestoreThread_Fn = void (*)(void*);

  void* thread_state = nullptr;
  PyEval_RestoreThread_Fn restore = nullptr;

  OptionalGilRelease() {
    auto py_is_initialized = reinterpret_cast<Py_IsInitialized_Fn>(dlsym(RTLD_DEFAULT, "Py_IsInitialized"));
    auto py_gil_check = reinterpret_cast<PyGILState_Check_Fn>(dlsym(RTLD_DEFAULT, "PyGILState_Check"));
    auto py_eval_save = reinterpret_cast<PyEval_SaveThread_Fn>(dlsym(RTLD_DEFAULT, "PyEval_SaveThread"));
    auto py_eval_restore =
        reinterpret_cast<PyEval_RestoreThread_Fn>(dlsym(RTLD_DEFAULT, "PyEval_RestoreThread"));
    if (!py_is_initialized || !py_gil_check || !py_eval_save || !py_eval_restore) return;
    if (!py_is_initialized()) return;
    if (!py_gil_check()) return;
    thread_state = py_eval_save();
    restore = py_eval_restore;
  }

  ~OptionalGilRelease() {
    if (thread_state && restore) restore(thread_state);
  }
};

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

void MFFTorchEngine::load_single_core_file(const std::string& core_pt_path) {
  load_custom_op_libs();
  ensure_python_custom_op_registrations();

  // AOTInductor .pt2: an Inductor-compiled inference package with the force traced INTO
  // the graph. Load via AOTIModelPackageLoader and take the simpler AOTI compute path
  // (no C++ edge_vec compute, no C++ autograd -- the .pt2 already returns (E, force)).
  // .pt2 is device-specific (compiled for the target GPU/CPU), so no device arg is needed.
  if (core_pt_path.size() >= 4 &&
      core_pt_path.compare(core_pt_path.size() - 4, 4, ".pt2") == 0) {
#if MFF_HAS_AOTI
    aoti_loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        core_pt_path, "model", /*run_single_threaded=*/false);
    aoti_mode_ = true;
    loaded_ = true;
    cached_ntotal_ = 0;
    cached_nedges_ = 0;
    // training-signature .pt2 carries no external/fidelity/phys heads
    core_takes_external_tensor_arg_ = false;
    core_requires_external_tensor_ = false;
    core_takes_fidelity_arg_ = false;
    core_requires_runtime_fidelity_ = false;
    core_exports_reciprocal_source_ = false;

    // Sidecar "<core>.pt2.meta": "nmax <N>" (the baked atom count -> pad ntotal up to it),
    // "pad_z <Z>" (dummy padding species), "fallback <path>" (an N-flexible TorchScript core to use
    // when ntotal > nmax, e.g. a ghost-count spike). Absent meta -> aoti_nmax_=0 (no padding).
    aoti_nmax_ = 0;
    aoti_pad_z_ = 1;
    have_ts_fallback_ = false;
    aoti_fallback_warned_ = false;
    {
      std::ifstream mf(core_pt_path + ".meta");
      std::string key, fb;
      while (mf >> key) {
        if (key == "nmax") mf >> aoti_nmax_;
        else if (key == "pad_z") mf >> aoti_pad_z_;
        else if (key == "fallback") mf >> fb;
        else { std::string rest; std::getline(mf, rest); }
      }
      if (!fb.empty()) {
        std::filesystem::path fbp(fb);
        if (fbp.is_relative())
          fbp = std::filesystem::path(core_pt_path).parent_path() / fbp;
        try {
          core_ = torch::jit::load(fbp.string(), device_);
          core_.eval();
          // The fallback's forward signature differs from the .pt2's -- read it to set the arg flags
          // run_forward_backward uses (external_tensor at arg>=9, fidelity at >=10), like the primary
          // TorchScript load path. Without this run_forward_backward omits external_tensor and the
          // core rejects the call ("missing value for argument 'external_tensor'").
          try {
            auto schema = core_.get_method("forward").function().getSchema();
            size_t nargs = schema.arguments().size();
            if (nargs > 0 && schema.arguments()[0].name() == "self") nargs -= 1;
            core_takes_external_tensor_arg_ = (nargs >= 9);
            core_takes_fidelity_arg_ = (nargs >= 10);
          } catch (...) {
            core_takes_external_tensor_arg_ = false;
            core_takes_fidelity_arg_ = false;
          }
          have_ts_fallback_ = true;
          std::fprintf(stderr, "[mff/torch] AOTI .pt2 N_max=%lld pad_z=%lld + TorchScript fallback %s (ext_arg=%d)\n",
                       (long long)aoti_nmax_, (long long)aoti_pad_z_, fbp.string().c_str(),
                       (int)core_takes_external_tensor_arg_);
        } catch (const std::exception& e) {
          std::fprintf(stderr, "[mff/torch] WARNING: AOTI fallback core %s failed to load: %s\n",
                       fbp.string().c_str(), e.what());
        }
      }
    }
    return;
#else
    throw std::runtime_error(
        "core path ends in .pt2 (AOTInductor package) but this libtorch build lacks "
        "AOTIModelPackageLoader (needs torch >= 2.6).");
#endif
  }

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
  core_takes_fidelity_arg_ = false;
  core_requires_runtime_fidelity_ = false;
  external_tensor_irrep_.clear();
  external_tensor_total_numel_ = 0;
  num_fidelity_levels_ = 0;
  export_fidelity_id_ = -1;
  external_tensor_has_field_1o_ = false;
  external_tensor_has_field_1e_ = false;
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
    core_takes_fidelity_arg_ = (nargs >= 10);
  } catch (...) {
    // Keep compatibility with older LibTorch builds that may not expose schema details cleanly.
    core_takes_external_tensor_arg_ = false;
    core_takes_fidelity_arg_ = false;
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
      (void)parse_int64_from_metadata(content, "\"long_range_mesh_size\"", long_range_mesh_size_);
      (void)parse_int64_from_metadata(content, "\"long_range_max_multipole_l\"", long_range_max_multipole_l_);
      (void)parse_string_from_metadata(content, "\"tensor_product_mode\"", tensor_product_mode_);
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
      (void)parse_int64_from_metadata(content, "\"trace_num_nodes\"", trace_num_nodes_);
      (void)parse_int64_from_metadata(content, "\"trace_num_edges\"", trace_num_edges_);
      (void)parse_string_from_metadata(content, "\"external_tensor_irrep\"", external_tensor_irrep_);
      (void)parse_int64_from_metadata(content, "\"external_tensor_total_numel\"", external_tensor_total_numel_);
      (void)parse_int64_from_metadata(content, "\"num_fidelity_levels\"", num_fidelity_levels_);
      (void)parse_int64_from_metadata(content, "\"export_fidelity_id\"", export_fidelity_id_);
      (void)parse_bool_from_metadata(content, "\"external_tensor_has_field_1o\"", external_tensor_has_field_1o_);
      (void)parse_bool_from_metadata(content, "\"external_tensor_has_field_1e\"", external_tensor_has_field_1e_);
      bool runtime_fidelity_input = false;
      if (parse_bool_from_metadata(content, "\"runtime_fidelity_input\"", runtime_fidelity_input)) {
        core_requires_runtime_fidelity_ = runtime_fidelity_input;
      }
      if (long_range_source_channels_ <= 0) long_range_source_channels_ = reciprocal_source_channels_;
      if (long_range_runtime_backend_ == "none" && core_exports_reciprocal_source_ && reciprocal_source_channels_ > 0) {
        long_range_runtime_backend_ = "mesh_fft";
      }
    }
  }
  if (core_takes_external_tensor_arg_) {
    bool parsed = parse_external_tensor_rank_from_metadata(core_pt_path + ".json", core_requires_external_tensor_);
    if (external_tensor_total_numel_ > 0) {
      core_requires_external_tensor_ = true;
      parsed = true;
    }
    if (external_tensor_irrep_.empty()) {
      (void)parse_external_tensor_irrep_from_metadata(core_pt_path + ".json", external_tensor_irrep_);
    }
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
  if (num_fidelity_levels_ > 0 && export_fidelity_id_ < 0 && core_takes_fidelity_arg_) {
    core_requires_runtime_fidelity_ = true;
  }

  // CUDA Graph replay (opt-in via MFF_CUDA_GRAPH=1).
  use_cuda_graph_ = false;
#if MFF_HAS_CUDA_GRAPH
  if (device_.is_cuda()) {
    const char* env = std::getenv("MFF_CUDA_GRAPH");
    if (env && (std::string(env) == "1" || std::string(env) == "true" || std::string(env) == "yes")) {
      use_cuda_graph_ = true;
    }
  }
#endif
}

void MFFTorchEngine::ensure_core_for_shape(int64_t nlocal, int64_t ntotal, int64_t nedges, bool warmup_on_switch) {
  if (!bundle_mode_) return;
  if (bundle_buckets_.empty()) {
    throw std::runtime_error("Manifest bundle has no buckets: " + bundle_manifest_path_);
  }
  const bool cue_node_first = (tensor_product_mode_ == "spherical-save-cue");
  const int64_t node_metric = cue_node_first ? nlocal : ntotal;
  int selected = -1;
  if (cue_node_first) {
    for (int i = 0; i < static_cast<int>(bundle_buckets_.size()); ++i) {
      const auto& bucket = bundle_buckets_[i];
      if (node_metric <= bucket.max_nodes) {
        selected = i;
        break;
      }
    }
    if (selected > 0) {
      while (selected < static_cast<int>(bundle_buckets_.size()) - 1 &&
             nedges > bundle_buckets_[selected].max_edges) {
        selected += 1;
      }
    }
  } else {
    for (int i = 0; i < static_cast<int>(bundle_buckets_.size()); ++i) {
      const auto& bucket = bundle_buckets_[i];
      if (node_metric <= bucket.max_nodes && nedges <= bucket.max_edges) {
        selected = i;
        break;
      }
    }
  }
  if (selected < 0) {
    selected = static_cast<int>(bundle_buckets_.size()) - 1;
    if (!bundle_warned_oversize_) {
      bundle_warned_oversize_ = true;
      std::fprintf(stderr,
                   "[USER-MFFTORCH] workload node_metric=%lld ntotal=%lld nedges=%lld exceeds all bundle buckets; "
                   "using largest bucket '%s' (%lld,%lld)\n",
                   static_cast<long long>(node_metric),
                   static_cast<long long>(ntotal),
                   static_cast<long long>(nedges),
                   bundle_buckets_[selected].name.c_str(),
                   static_cast<long long>(bundle_buckets_[selected].max_nodes),
                   static_cast<long long>(bundle_buckets_[selected].max_edges));
    }
  }
  if (current_bucket_index_ >= 0) {
    const auto& current = bundle_buckets_[current_bucket_index_];
    if (node_metric <= current.max_nodes && nedges <= current.max_edges) {
      return;
    }
    if (selected <= current_bucket_index_) {
      selected = current_bucket_index_;
    }
  }
  if (selected == current_bucket_index_ && loaded_) return;
  const bool promoted = current_bucket_index_ >= 0 && selected > current_bucket_index_;
  load_single_core_file(bundle_buckets_[selected].core_path);
  current_bucket_index_ = selected;
  std::fprintf(stderr,
               "[USER-MFFTORCH] %s bucket '%s' (%lld,%lld) for node_metric=%lld ntotal=%lld nedges=%lld\n",
               promoted ? "promoted to" : "selected",
               bundle_buckets_[selected].name.c_str(),
               static_cast<long long>(bundle_buckets_[selected].max_nodes),
               static_cast<long long>(bundle_buckets_[selected].max_edges),
               static_cast<long long>(node_metric),
               static_cast<long long>(ntotal),
               static_cast<long long>(nedges));
  if (warmup_on_switch && !warming_up_) {
    warmup(0, 0);
  }
}

void MFFTorchEngine::load_core(const std::string& core_pt_path, const std::string& device_str) {
  device_ = pick_device(device_str);
  const bool debug_bundle = std::getenv("MFF_DEBUG_BUNDLE") != nullptr;
  if (debug_bundle) {
    std::fprintf(stderr, "[USER-MFFTORCH] load_core path=%s device=%s\n", core_pt_path.c_str(), device_str.c_str());
  }
  bundle_mode_ = false;
  bundle_manifest_path_.clear();
  bundle_buckets_.clear();
  current_bucket_index_ = -1;
  bundle_warned_oversize_ = false;

  std::string manifest_path;
  if (is_directory_path(core_pt_path)) {
    manifest_path = (std::filesystem::path(core_pt_path) / "manifest.json").string();
  } else if (core_pt_path.size() >= 5 && core_pt_path.substr(core_pt_path.size() - 5) == ".json") {
    manifest_path = core_pt_path;
  }
  if (!manifest_path.empty()) {
    if (debug_bundle) {
      std::fprintf(stderr, "[USER-MFFTORCH] manifest path=%s\n", manifest_path.c_str());
    }
    const auto manifest_content = read_text_file(manifest_path);
    std::string mode;
    (void)parse_string_from_metadata(manifest_content, "\"tensor_product_mode\"", mode);
    tensor_product_mode_ = mode;
    bundle_mode_ = true;
    bundle_manifest_path_ = manifest_path;
    std::string array_block;
    if (!extract_json_array_block(manifest_content, "\"buckets\"", array_block)) {
      throw std::runtime_error("Failed to parse buckets from manifest: " + manifest_path);
    }
    const auto base_dir = std::filesystem::path(manifest_path).parent_path();
    for (const auto& obj : split_top_level_object_blocks(array_block)) {
      BucketSpec bucket;
      if (!parse_string_from_metadata(obj, "\"name\"", bucket.name)) continue;
      if (!parse_string_from_metadata(obj, "\"core_path\"", bucket.core_path)) continue;
      (void)parse_int64_from_metadata(obj, "\"max_nodes\"", bucket.max_nodes);
      (void)parse_int64_from_metadata(obj, "\"max_edges\"", bucket.max_edges);
      (void)parse_int64_from_metadata(obj, "\"trace_num_nodes\"", bucket.trace_num_nodes);
      (void)parse_int64_from_metadata(obj, "\"trace_num_edges\"", bucket.trace_num_edges);
      (void)parse_string_from_metadata(obj, "\"dtype\"", bucket.dtype);
      (void)parse_string_from_metadata(obj, "\"jit_mode\"", bucket.jit_mode);
      bucket.core_path = (base_dir / bucket.core_path).string();
      bundle_buckets_.push_back(bucket);
    }
    if (bundle_buckets_.empty()) {
      throw std::runtime_error("No valid buckets found in manifest: " + manifest_path);
    }
    if (debug_bundle) {
      std::fprintf(stderr, "[USER-MFFTORCH] parsed %zu buckets for mode=%s\n",
                   bundle_buckets_.size(), tensor_product_mode_.c_str());
    }
    loaded_ = false;
    return;
  }
  load_single_core_file(core_pt_path);
}

void MFFTorchEngine::prepare_for_shape(int64_t nlocal, int64_t ntotal, int64_t nedges) {
  if (!bundle_mode_) return;
  ensure_core_for_shape(nlocal, ntotal, nedges, true);
}

void MFFTorchEngine::warmup(int64_t N, int64_t E) {
  if (bundle_mode_ && current_bucket_index_ < 0) return;
  if (!loaded_) return;
  // An AOTI .pt2 is precompiled and BAKES its atom count N. The TorchScript-style JIT
  // warmup doesn't apply, and running the graph here at the guessed default N (!= baked N)
  // feeds it a wrong-shaped input -> device-side index assert ("index out of bounds").
  // The first real compute() call runs it at the correct N, so skip warmup in AOTI mode.
  if (aoti_mode_) return;
  if (trace_num_nodes_ > 0) N = trace_num_nodes_;
  if (trace_num_edges_ > 0) E = trace_num_edges_;

  // Suspend CUDA Graph during warmup — Kokkos background ops (cudaFreeHost
  // etc.) are incompatible with CUDA stream capture.
  const bool saved_cuda_graph = use_cuda_graph_;
  use_cuda_graph_ = false;

  auto pos = torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  auto A = torch::ones({N}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto edge_src = torch::zeros({E}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto edge_dst = torch::zeros({E}, torch::TensorOptions().dtype(torch::kInt64).device(device_));
  auto edge_shifts = torch::zeros({E, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  auto cell = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(device_)).unsqueeze(0) * 100.0f;
  std::vector<torch::Tensor> warmup_external_tensors;
  std::vector<torch::Tensor> warmup_fidelity_tensors;
  if (core_takes_external_tensor_arg_) {
    warmup_external_tensors.push_back(
        torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    if (external_tensor_total_numel_ > 0) {
      if (should_use_flat_external_tensor(
              external_tensor_total_numel_, external_tensor_has_field_1o_, external_tensor_has_field_1e_)) {
        warmup_external_tensors.push_back(
            torch::zeros({external_tensor_total_numel_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
      } else if (external_tensor_total_numel_ == 9) {
        warmup_external_tensors.push_back(
            torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
      } else {
        warmup_external_tensors.push_back(
            torch::zeros({external_tensor_total_numel_}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
      }
    } else {
      warmup_external_tensors.push_back(
          torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
      warmup_external_tensors.push_back(
          torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    }
  } else {
    warmup_external_tensors.push_back(
        torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
  }
  if (core_takes_fidelity_arg_) {
    warmup_fidelity_tensors.push_back(
        torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    if (num_fidelity_levels_ > 1) {
      warmup_fidelity_tensors.push_back(
          torch::full({1}, 1, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
    }
  } else {
    warmup_fidelity_tensors.push_back(torch::Tensor());
  }

  bool warmed = false;
  std::string last_error;
  warming_up_ = true;
  for (const auto& external_tensor : warmup_external_tensors) {
    for (const auto& fidelity_ids : warmup_fidelity_tensors) {
      try {
        for (int i = 0; i < 3; i++) {
          compute(N, N, pos, A, edge_src, edge_dst, edge_shifts, cell, external_tensor, fidelity_ids, false);
        }
        warmed = true;
        break;
      } catch (const std::exception& e) {
        last_error = e.what();
      } catch (...) {
        last_error = "non-std exception";
      }
    }
    if (warmed) break;
  }
  if (!warmed) {
    warming_up_ = false;
    if (last_error.empty()) last_error = "no error detail captured";
    throw std::runtime_error(
        "MFFTorchEngine warmup failed for all supported external tensor shapes; last error: " + last_error);
  }
  warming_up_ = false;
  if (device_.is_cuda()) torch::cuda::synchronize();
  use_cuda_graph_ = saved_cuda_graph;
}

// Core forward+backward logic shared by eager and CUDA-graph paths.
MFFOutputs MFFTorchEngine::compute(int64_t nlocal, int64_t ntotal,
                                  const torch::Tensor& pos_in,
                                  const torch::Tensor& A_in,
                                  const torch::Tensor& edge_src_in,
                                  const torch::Tensor& edge_dst_in,
                                  const torch::Tensor& edge_shifts_in,
                                  const torch::Tensor& cell_in,
                                  const torch::Tensor& external_tensor_in,
                                  const torch::Tensor& fidelity_ids_in,
                                  bool need_energy,
                                  bool need_atom_virial) {
  if (bundle_mode_ && !warming_up_) {
    ensure_core_for_shape(nlocal, ntotal, edge_src_in.size(0), true);
  }
  if (!loaded_) throw std::runtime_error("MFFTorchEngine not loaded");
  if (nlocal <= 0 || ntotal <= 0) return {};

  // DEBUG: dump the exact graph fed to the model (MFF_DUMP_GRAPH=1) so it can be replayed
  // through the eager model in Python to localize an energy discrepancy. One-shot per run.
  if (std::getenv("MFF_DUMP_GRAPH") && !warming_up_) {
    static bool dumped = false;
    if (!dumped) {
      dumped = true;
      const std::string p = "/tmp/mff_graph_";
      torch::save(pos_in.detach().to(torch::kCPU, torch::kFloat64), p + "pos.pt");
      torch::save(A_in.detach().to(torch::kCPU, torch::kInt64), p + "A.pt");
      torch::save(edge_src_in.detach().to(torch::kCPU, torch::kInt64), p + "es.pt");
      torch::save(edge_dst_in.detach().to(torch::kCPU, torch::kInt64), p + "ed.pt");
      torch::save(edge_shifts_in.detach().to(torch::kCPU, torch::kFloat64), p + "esh.pt");
      torch::save(cell_in.detach().to(torch::kCPU, torch::kFloat64), p + "cell.pt");
      std::ofstream mf(p + "meta.txt");
      mf << nlocal << " " << ntotal << " " << edge_src_in.size(0) << "\n";
      mf.close();
      std::fprintf(stderr, "[MFF_DUMP_GRAPH] nlocal=%lld ntotal=%lld E=%lld -> /tmp/mff_graph_*.pt\n",
                   (long long)nlocal, (long long)ntotal, (long long)edge_src_in.size(0));
    }
  }

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
    } else if (external_tensor_total_numel_ > 0 && external_tensor_in.numel() != external_tensor_total_numel_) {
      throw std::runtime_error(
          "USER-MFFTORCH external_tensor length does not match core.pt metadata");
    } else if (external_tensor_total_numel_ == 0 && external_tensor_in.numel() != 3 && external_tensor_in.numel() != 9) {
      throw std::runtime_error(
          "USER-MFFTORCH currently supports rank-1 (3 values) and rank-2 (9 values) external tensors only");
    } else if (
        should_use_flat_external_tensor(
            external_tensor_total_numel_, external_tensor_has_field_1o_, external_tensor_has_field_1e_) ||
        (external_tensor_total_numel_ == 0 && external_tensor_in.numel() == 3)) {
      external_tensor = (external_tensor_in.device() == device_ && external_tensor_in.dtype() == torch::kFloat32)
                            ? external_tensor_in.reshape({external_tensor_in.numel()})
                            : external_tensor_in.to(device_, torch::kFloat32).reshape({external_tensor_in.numel()});
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
  torch::Tensor fidelity_ids;
  if (core_takes_fidelity_arg_) {
    if (core_requires_runtime_fidelity_ && (!fidelity_ids_in.defined() || fidelity_ids_in.numel() == 0)) {
      throw std::runtime_error(
          "TorchScript core expects fidelity_ids, but pair_style mff/torch did not provide fidelity");
    }
    if (!fidelity_ids_in.defined() || fidelity_ids_in.numel() == 0) {
      const int64_t graph_count = cell.size(0);
      const int64_t fallback_fid = export_fidelity_id_ >= 0 ? export_fidelity_id_ : 0;
      fidelity_ids = torch::full({graph_count},
                                 fallback_fid,
                                 torch::TensorOptions().dtype(torch::kInt64).device(device_));
    } else {
      fidelity_ids = (fidelity_ids_in.device() == device_ && fidelity_ids_in.dtype() == torch::kInt64)
                         ? fidelity_ids_in.reshape({fidelity_ids_in.numel()})
                         : fidelity_ids_in.to(device_, torch::kInt64).reshape({fidelity_ids_in.numel()});
      const int64_t graph_count = cell.size(0);
      if (fidelity_ids.numel() != graph_count) {
        throw std::runtime_error("USER-MFFTORCH fidelity_ids length does not match number of graphs");
      }
      if (num_fidelity_levels_ > 0) {
        auto min_id = fidelity_ids.min().item<int64_t>();
        auto max_id = fidelity_ids.max().item<int64_t>();
        if (min_id < 0 || max_id >= num_fidelity_levels_) {
          throw std::runtime_error("USER-MFFTORCH fidelity_ids value is out of range for core.pt metadata");
        }
      }
    }
  } else if (fidelity_ids_in.defined() && fidelity_ids_in.numel() > 0) {
    throw std::runtime_error("pair_style mff/torch received fidelity input, but core.pt does not accept fidelity_ids");
  }

  if (cached_ntotal_ != ntotal) {
    buf_batch_ = torch::zeros({ntotal},
                              torch::TensorOptions().dtype(torch::kInt64).device(device_));
    cached_ntotal_ = ntotal;
  }
  cached_nedges_ = nedges;

  // AOTI .pt2: force is already in the graph -> single inference call, no autograd.
  if (aoti_mode_) {
    // ntotal > baked N_max (e.g. a ghost-count spike): the N-baked .pt2 can't run it. Fall back to
    // the N-flexible TorchScript core if one was loaded, else error clearly. (Common steps stay on
    // the fast .pt2 path; the fallback is a rarely-hit safety net.)
    if (aoti_nmax_ > 0 && ntotal > aoti_nmax_) {
      if (have_ts_fallback_) {
        if (!aoti_fallback_warned_) {
          aoti_fallback_warned_ = true;
          std::fprintf(stderr, "[mff/torch] ntotal %lld > AOTI N_max %lld -> TorchScript fallback "
                       "(slower). Re-export the .pt2 with a larger --atoms if frequent.\n",
                       (long long)ntotal, (long long)aoti_nmax_);
        }
        return run_forward_backward(pos0, A, edge_src, edge_dst, edge_shifts, cell,
                                    external_tensor, fidelity_ids,
                                    nlocal, ntotal, need_energy, need_atom_virial);
      }
      throw std::runtime_error(
          "mff/torch: ntotal=" + std::to_string(ntotal) + " exceeds the AOTI .pt2 baked N_max=" +
          std::to_string(aoti_nmax_) + " and no TorchScript fallback configured; re-export with a "
          "larger --atoms or add 'fallback <core.pt>' to <core>.pt2.meta.");
    }
    MFFOutputs out = run_aoti(pos0, A, edge_src, edge_dst, edge_shifts, cell);
    // run_aoti fills atom_energy + forces but NOT the scalar out.energy that the pair_style
    // adds to eng_vdwl (the reported PE). Reduce the LOCAL atom energies here (compute() has
    // nlocal), mirroring run_forward_backward's E_local = atom_e[0:nlocal].sum(). Without this
    // PE reads 0 even though forces are correct.
    if (need_energy && out.atom_energy.defined()) {
      out.energy = out.atom_energy.reshape({-1}).narrow(0, 0, nlocal)
                       .to(torch::kFloat64).sum().item<double>();
    }
    return out;
  }

#if MFF_HAS_CUDA_GRAPH
  if (use_cuda_graph_ && device_.is_cuda()) {
    return compute_with_cuda_graph(nlocal, ntotal, pos0, A, edge_src, edge_dst,
                                  edge_shifts, cell, external_tensor, fidelity_ids,
                                  need_energy, need_atom_virial);
  }
#endif

  return run_forward_backward(pos0, A, edge_src, edge_dst, edge_shifts, cell,
                              external_tensor, fidelity_ids,
                              nlocal, ntotal, need_energy, need_atom_virial);
}

MFFOutputs MFFTorchEngine::run_aoti(
    const torch::Tensor& pos0, const torch::Tensor& A,
    const torch::Tensor& edge_src, const torch::Tensor& edge_dst,
    const torch::Tensor& edge_shifts, const torch::Tensor& cell) {
#if MFF_HAS_AOTI
  // Training-signature inputs: (pos, A, batch, edge_src, edge_dst, edge_shifts, cell).
  // The .pt2 computes edge_vec internally and returns (atom_energy[N,1], force[N,3] = -dE/dpos)
  // with the force traced into the graph. The .pt2 BAKES N, so when ntotal < aoti_nmax_ we PAD the
  // node tensors up to aoti_nmax_ with dummy atoms (valid species, no edges -> isolated -> they
  // contribute nothing to the real atoms and are sliced off afterwards). Edges/shifts are E-dynamic
  // and reference only real atoms, so they are passed unchanged. buf_batch_ is the zeros batch index.
  const int64_t ntot = pos0.size(0);
  at::Tensor pos_in = pos0, A_in = A, batch_in = buf_batch_;
  if (aoti_nmax_ > 0 && ntot < aoti_nmax_) {
    const int64_t k = aoti_nmax_ - ntot;
    pos_in = torch::cat({pos0, torch::zeros({k, 3}, pos0.options())}, 0);
    A_in = torch::cat({A, torch::full({k}, aoti_pad_z_, A.options())}, 0);
    batch_in = torch::cat({buf_batch_, torch::zeros({k}, buf_batch_.options())}, 0);
  }
  const std::vector<at::Tensor> inputs = {
      pos_in, A_in, batch_in, edge_src, edge_dst, edge_shifts, cell};
  auto outs = aoti_loader_->run(inputs);
  if (outs.size() < 2) {
    throw std::runtime_error(
        "AOTI .pt2 must return (atom_energy, force); got fewer than 2 outputs");
  }
  MFFOutputs out;
  // Slice off the dummy padding atoms -> outputs for the real ntot atoms only.
  out.atom_energy = outs[0].narrow(0, 0, ntot).contiguous();
  out.forces = outs[1].narrow(0, 0, ntot).contiguous();
  return out;
#else
  (void)pos0; (void)A; (void)edge_src; (void)edge_dst; (void)edge_shifts; (void)cell;
  throw std::runtime_error("run_aoti called but this build lacks AOTIModelPackageLoader");
#endif
}

#if MFF_HAS_CUDA_GRAPH
MFFOutputs MFFTorchEngine::compute_with_cuda_graph(
    int64_t nlocal, int64_t ntotal,
    const torch::Tensor& pos0, const torch::Tensor& A,
    const torch::Tensor& edge_src, const torch::Tensor& edge_dst,
    const torch::Tensor& edge_shifts, const torch::Tensor& cell,
    const torch::Tensor& external_tensor, const torch::Tensor& fidelity_ids,
    bool need_energy, bool need_atom_virial) {

  const int64_t nedges = edge_src.size(0);
  const bool can_replay =
      cg_cache_.valid &&
      cg_cache_.ntotal == ntotal &&
      cg_cache_.nedges == nedges &&
      cg_cache_.nlocal == nlocal &&
      cg_cache_.need_atom_virial == need_atom_virial;

  if (!can_replay) {
    // First call or sizes changed — (re)capture a graph for this shape. Capture runs
    // on placeholder buffers, so we must STILL replay below with the real step data
    // (the capture-run outputs correspond to the placeholder inputs, not this step).
    capture_cuda_graph(nlocal, ntotal, nedges, need_atom_virial);
    if (!cg_cache_.valid) {
      // Capture failed; use_cuda_graph_ was already turned off — fall back to eager.
      return run_forward_backward(pos0, A, edge_src, edge_dst, edge_shifts, cell,
                                  external_tensor, fidelity_ids,
                                  nlocal, ntotal, need_energy, need_atom_virial);
    }
  }

  // Overwrite the static input buffers with this step's data, then replay. This runs on
  // EVERY graph step — including the one right after a (re)capture — so the returned
  // results always correspond to the actual inputs, never the placeholder capture run.
  {
    c10::cuda::CUDAStreamGuard guard(cg_cache_.capture_stream);
    cg_cache_.pos_in.copy_(pos0);
    cg_cache_.A_in.copy_(A);
    cg_cache_.edge_src_in.copy_(edge_src);
    cg_cache_.edge_dst_in.copy_(edge_dst);
    cg_cache_.edge_shifts_in.copy_(edge_shifts);
    cg_cache_.cell_in.copy_(cell);
    if (external_tensor.defined() && external_tensor.numel() > 0 && cg_cache_.external_tensor_in.defined()) {
      cg_cache_.external_tensor_in.copy_(external_tensor);
    }
    if (fidelity_ids.defined() && fidelity_ids.numel() > 0 && cg_cache_.fidelity_ids_in.defined()) {
      cg_cache_.fidelity_ids_in.copy_(fidelity_ids);
    }
    cg_cache_.graph.replay();
    cg_cache_.capture_stream.synchronize();
  }

  MFFOutputs out;
  out.forces = cg_cache_.forces_out;
  out.atom_energy = cg_cache_.atom_e_out;
  out.global_phys = cg_cache_.global_phys_out;
  out.atom_phys = cg_cache_.atom_phys_out;
  out.global_phys_mask = cg_cache_.global_phys_mask_out;
  out.atom_phys_mask = cg_cache_.atom_phys_mask_out;
  out.reciprocal_source = cg_cache_.reciprocal_source_out;
  out.atom_virial = cg_cache_.atom_vir_out;
  if (need_energy) {
    out.energy = cg_cache_.E_local_out.detach().to(torch::kCPU).item<double>();
  }
  return out;
}

void MFFTorchEngine::capture_cuda_graph(
    int64_t nlocal, int64_t ntotal, int64_t nedges,
    bool need_atom_virial) {
  cg_cache_.valid = false;

  // Allocate static input buffers on the engine device.
  auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  auto iopt = torch::TensorOptions().dtype(torch::kInt64).device(device_);
  cg_cache_.pos_in = torch::zeros({ntotal, 3}, fopt);
  cg_cache_.A_in = torch::ones({ntotal}, iopt);
  cg_cache_.edge_src_in = torch::zeros({nedges}, iopt);
  cg_cache_.edge_dst_in = torch::zeros({nedges}, iopt);
  cg_cache_.edge_shifts_in = torch::zeros({nedges, 3}, fopt);
  cg_cache_.cell_in = torch::eye(3, fopt).unsqueeze(0) * 100.0f;
  if (core_takes_external_tensor_arg_) {
    if (external_tensor_total_numel_ > 0) {
      cg_cache_.external_tensor_in = torch::zeros({external_tensor_total_numel_}, fopt);
    } else {
      cg_cache_.external_tensor_in = torch::empty({0}, fopt);
    }
  }
  if (core_takes_fidelity_arg_) {
    cg_cache_.fidelity_ids_in = torch::zeros({1}, iopt);
  }

  // Warmup BEFORE capture (grad enabled, so autograd kernels are exercised too).
  // SEVERAL passes are required, not one: TorchScript's profiling graph executor runs
  // its optimization passes (e.g. EliminateCommonSubexpression, which compares constant
  // tensors via torch.equal -> .item()) on the 2nd execution of a given shape. Those
  // host syncs are illegal during capture, so we must run the model enough times here
  // that the optimized plan is built and cached before we begin capturing. Configurable
  // via MFF_CUDA_GRAPH_WARMUP (default 5).
  int cg_warmup = 5;
  if (const char* w = std::getenv("MFF_CUDA_GRAPH_WARMUP")) {
    int v = std::atoi(w);
    if (v > 0) cg_warmup = v;
  }
  for (int wi = 0; wi < cg_warmup; ++wi) {
    try {
      run_forward_backward(cg_cache_.pos_in, cg_cache_.A_in,
                           cg_cache_.edge_src_in, cg_cache_.edge_dst_in,
                           cg_cache_.edge_shifts_in, cg_cache_.cell_in,
                           cg_cache_.external_tensor_in, cg_cache_.fidelity_ids_in,
                           nlocal, ntotal, false, need_atom_virial);
      if (device_.is_cuda()) torch::cuda::synchronize();
    } catch (...) {
      // Warmup may fail for degenerate inputs; proceed to capture.
    }
  }
  if (device_.is_cuda()) torch::cuda::synchronize();

  // Drain all GPU work from every stream before capture.  Kokkos background
  // ops (cudaFreeHost, etc.) are globally illegal during stream capture.
  torch::cuda::synchronize();

  // Capture the graph on a dedicated non-default stream.
  cg_cache_.capture_stream = c10::cuda::getStreamFromPool(false, device_.index());
  try {
    c10::cuda::CUDAStreamGuard guard(cg_cache_.capture_stream);
    cg_cache_.graph.capture_begin();

    // need_energy=false during capture: computing out.energy does E_local.item(), a
    // host sync that is illegal during capture. The energy tensor is captured
    // separately below (cg_cache_.E_local_out) and read with .item() after replay.
    auto result = run_forward_backward(
        cg_cache_.pos_in, cg_cache_.A_in,
        cg_cache_.edge_src_in, cg_cache_.edge_dst_in,
        cg_cache_.edge_shifts_in, cg_cache_.cell_in,
        cg_cache_.external_tensor_in, cg_cache_.fidelity_ids_in,
        nlocal, ntotal, /*need_energy=*/false, need_atom_virial);

    cg_cache_.forces_out = result.forces;
    cg_cache_.atom_e_out = result.atom_energy;
    cg_cache_.E_local_out = torch::zeros({1}, fopt);
    cg_cache_.global_phys_out = result.global_phys;
    cg_cache_.atom_phys_out = result.atom_phys;
    cg_cache_.global_phys_mask_out = result.global_phys_mask;
    cg_cache_.atom_phys_mask_out = result.atom_phys_mask;
    cg_cache_.reciprocal_source_out = result.reciprocal_source;
    cg_cache_.atom_vir_out = result.atom_virial;

    auto atom_e_flat = result.atom_energy.view({result.atom_energy.size(0)});
    cg_cache_.E_local_out = atom_e_flat.narrow(0, 0, nlocal).sum();

    cg_cache_.graph.capture_end();
    cg_cache_.capture_stream.synchronize();

    cg_cache_.ntotal = ntotal;
    cg_cache_.nedges = nedges;
    cg_cache_.nlocal = nlocal;
    cg_cache_.need_atom_virial = need_atom_virial;
    cg_cache_.valid = true;
  } catch (const std::exception& e) {
    // Capture failed (host sync during capture, unsupported op, etc.) — fall back
    // to eager mode permanently for this run. End the capture first so the stream is
    // not left stuck in "capturing" state (which would make the eager fallback's own
    // CUDA calls fail too); ignore any secondary error from the cleanup.
    try { cg_cache_.graph.capture_end(); } catch (...) {}
    use_cuda_graph_ = false;
    cg_cache_.valid = false;
    fprintf(stderr, "[MFFTorchEngine] CUDA Graph capture failed: %s\n"
                    "[MFFTorchEngine] Falling back to eager mode.\n", e.what());
  } catch (...) {
    try { cg_cache_.graph.capture_end(); } catch (...) {}
    use_cuda_graph_ = false;
    cg_cache_.valid = false;
    fprintf(stderr, "[MFFTorchEngine] CUDA Graph capture failed (unknown error), "
                    "falling back to eager mode.\n");
  }
}
#endif  // MFF_HAS_CUDA_GRAPH

MFFOutputs MFFTorchEngine::run_forward_backward(
    const torch::Tensor& pos0, const torch::Tensor& A,
    const torch::Tensor& edge_src, const torch::Tensor& edge_dst,
    const torch::Tensor& edge_shifts, const torch::Tensor& cell,
    const torch::Tensor& external_tensor, const torch::Tensor& fidelity_ids,
    int64_t nlocal, int64_t ntotal, bool need_energy, bool need_atom_virial) {

  const bool debug_timings = []() {
    const char* env = std::getenv("MFF_DEBUG_TIMINGS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  const auto t_start = std::chrono::steady_clock::now();

  auto pos = pos0.detach().requires_grad_(true);
  auto edge_batch = buf_batch_.index_select(0, edge_src);
  auto edge_cells = cell.index_select(0, edge_batch);
  auto shift_vec = torch::einsum("ni,nij->nj", {edge_shifts, edge_cells});

  torch::Tensor shift_leaf;
  if (need_atom_virial) {
    shift_leaf = shift_vec.detach().requires_grad_(true);
  } else {
    shift_leaf = shift_vec;
  }

  auto edge_vec = pos.index_select(0, edge_dst) - pos.index_select(0, edge_src) + shift_leaf;
  // NOTE: this device sync is a pure timing barrier (only used under debug_timings).
  // It MUST be gated: a device sync is illegal during CUDA-graph capture, and running
  // it unconditionally both broke MFF_CUDA_GRAPH capture and added a needless sync to
  // the eager hot path.
  if (debug_timings && device_.is_cuda()) torch::cuda::synchronize();
  const auto t_after_prep = std::chrono::steady_clock::now();

  std::vector<torch::jit::IValue> inputs;
  inputs.reserve((core_takes_external_tensor_arg_ ? 1 : 0) + (core_takes_fidelity_arg_ ? 10 : 8));
  inputs.push_back(pos);
  inputs.push_back(A);
  inputs.push_back(buf_batch_);
  inputs.push_back(edge_src);
  inputs.push_back(edge_dst);
  inputs.push_back(edge_shifts);
  inputs.push_back(cell);
  inputs.push_back(edge_vec);
  if (core_takes_external_tensor_arg_) inputs.push_back(external_tensor);
  if (core_takes_fidelity_arg_) inputs.push_back(fidelity_ids);

  maybe_dump_forward_inputs_once(inputs);

  auto core_out = core_.forward(inputs);
  if (debug_timings && device_.is_cuda()) torch::cuda::synchronize();  // timing-only; gated for capture-safety
  const auto t_after_forward = std::chrono::steady_clock::now();
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
      throw std::runtime_error("TorchScript core atom_phys last dim must be 31");
    }
    if (global_phys_mask.defined() && global_phys_mask.numel() > 0 && global_phys_mask.numel() != kPhysMaskWidth) {
      throw std::runtime_error("TorchScript core global_phys_mask dim must be 5");
    }
    if (atom_phys_mask.defined() && atom_phys_mask.numel() > 0 && atom_phys_mask.numel() != kPhysMaskWidth) {
      throw std::runtime_error("TorchScript core atom_phys_mask dim must be 5");
    }
  } else {
    throw std::runtime_error("TorchScript core returned unsupported type (expected Tensor or tuple)");
  }
  auto atom_e_flat = atom_e.view({atom_e.size(0)});
  auto E_local = atom_e_flat.narrow(0, 0, nlocal).sum();

  std::vector<torch::Tensor> grad_inputs = {pos};
  if (need_atom_virial) grad_inputs.push_back(shift_leaf);

  OptionalGilRelease no_gil;
  auto grads = torch::autograd::grad({E_local}, grad_inputs, {}, /*retain_graph=*/false,
                                     /*create_graph=*/false, /*allow_unused=*/true);
  if (debug_timings && device_.is_cuda()) torch::cuda::synchronize();  // timing-only; gated for capture-safety
  const auto t_after_grad = std::chrono::steady_clock::now();
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
    auto edge_forces = -grads[1];

    auto r0 = edge_vec.select(1, 0);
    auto r1 = edge_vec.select(1, 1);
    auto r2 = edge_vec.select(1, 2);
    auto f0 = edge_forces.select(1, 0);
    auto f1 = edge_forces.select(1, 1);
    auto f2 = edge_forces.select(1, 2);

    auto edge_vir = torch::stack({
        r0 * f0,
        r1 * f1,
        r2 * f2,
        r0 * f1,
        r0 * f2,
        r1 * f2,
    }, 1);

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
  if (debug_timings) {
    const auto prep_ms =
        std::chrono::duration<double, std::milli>(t_after_prep - t_start).count();
    const auto forward_ms =
        std::chrono::duration<double, std::milli>(t_after_forward - t_after_prep).count();
    const auto grad_ms =
        std::chrono::duration<double, std::milli>(t_after_grad - t_after_forward).count();
    const auto total_ms =
        std::chrono::duration<double, std::milli>(t_after_grad - t_start).count();
    fprintf(stderr,
            "[MFF_DEBUG_TIMINGS] ntotal=%lld nedges=%lld prep_ms=%.3f forward_ms=%.3f grad_ms=%.3f total_ms=%.3f\n",
            static_cast<long long>(ntotal),
            static_cast<long long>(edge_src.size(0)),
            prep_ms,
            forward_ms,
            grad_ms,
            total_ms);
  }
  return out;
}

}  // namespace mfftorch
