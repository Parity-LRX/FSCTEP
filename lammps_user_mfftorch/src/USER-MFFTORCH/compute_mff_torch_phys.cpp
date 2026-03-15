#include "compute_mff_torch_phys.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "pair.h"
#include "pair_mff_torch.h"
#include "update.h"

#include <cstring>
#include <string>

using namespace LAMMPS_NS;

namespace {

constexpr int kGlobalPhysValueCols = 22;
constexpr int kAtomPhysValueCols = 31;
constexpr int kPhysMaskCols = 5;

std::string normalize_quantity_name(std::string name) {
  if (name.size() > 9 && name.rfind("_per_atom") == name.size() - 9) {
    name.resize(name.size() - 9);
  }
  return name;
}

bool lookup_value_selection(const std::string &quantity_in, const std::string *component,
                            int &offset, int &length, bool &scalar_output) {
  const std::string quantity = normalize_quantity_name(quantity_in);
  if (quantity == "charge") {
    offset = 0;
    length = 1;
    scalar_output = true;
    return component == nullptr;
  }
  if (quantity == "dipole") {
    offset = 1;
    if (component == nullptr) {
      length = 3;
      scalar_output = false;
      return true;
    }
    if (*component == "x") {
      offset += 0;
    } else if (*component == "y") {
      offset += 1;
    } else if (*component == "z") {
      offset += 2;
    } else {
      return false;
    }
    length = 1;
    scalar_output = true;
    return true;
  }
  if (quantity == "polarizability" || quantity == "quadrupole") {
    offset = (quantity == "polarizability") ? 4 : 13;
    if (component == nullptr) {
      length = 9;
      scalar_output = false;
      return true;
    }
    if (*component == "xx") {
      offset += 0;
    } else if (*component == "xy") {
      offset += 1;
    } else if (*component == "xz") {
      offset += 2;
    } else if (*component == "yx") {
      offset += 3;
    } else if (*component == "yy") {
      offset += 4;
    } else if (*component == "yz") {
      offset += 5;
    } else if (*component == "zx") {
      offset += 6;
    } else if (*component == "zy") {
      offset += 7;
    } else if (*component == "zz") {
      offset += 8;
    } else {
      return false;
    }
    length = 1;
    scalar_output = true;
    return true;
  }
  if (quantity == "born_effective_charge") {
    offset = 22;
    if (component == nullptr) {
      length = 9;
      scalar_output = false;
      return true;
    }
    if (*component == "xx") {
      offset += 0;
    } else if (*component == "xy") {
      offset += 1;
    } else if (*component == "xz") {
      offset += 2;
    } else if (*component == "yx") {
      offset += 3;
    } else if (*component == "yy") {
      offset += 4;
    } else if (*component == "yz") {
      offset += 5;
    } else if (*component == "zx") {
      offset += 6;
    } else if (*component == "zy") {
      offset += 7;
    } else if (*component == "zz") {
      offset += 8;
    } else {
      return false;
    }
    length = 1;
    scalar_output = true;
    return true;
  }
  return false;
}

bool lookup_mask_selection(const std::string &quantity_in, int &offset) {
  const std::string quantity = normalize_quantity_name(quantity_in);
  if (quantity == "charge") {
    offset = 0;
    return true;
  }
  if (quantity == "dipole") {
    offset = 1;
    return true;
  }
  if (quantity == "polarizability") {
    offset = 2;
    return true;
  }
  if (quantity == "quadrupole") {
    offset = 3;
    return true;
  }
  if (quantity == "born_effective_charge") {
    offset = 4;
    return true;
  }
  return false;
}

}  // namespace

ComputeMFFTorchPhys::ComputeMFFTorchPhys(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg) {
  if (narg < 4 || narg > 6) error->all(FLERR, "Illegal compute mff/torch/phys command");
  if (std::strcmp(arg[1], "all") != 0) {
    error->all(FLERR, "compute mff/torch/phys currently requires group all");
  }

  parse_mode(arg[3]);
  parse_selection(narg, arg);

  scalar_flag = 0;
  vector_flag = 0;
  peratom_flag = 0;
  size_vector = 0;
  size_peratom_cols = 0;
  peratom_flag = 0;
  extvector = 0;
  extscalar = 0;
  peflag = 0;
  pressflag = 0;
  timeflag = 1;

  switch (mode_) {
    case Mode::GLOBAL_VALUES:
    case Mode::GLOBAL_MASK:
    case Mode::ATOM_MASK:
      if (use_scalar_output_) {
        scalar_flag = 1;
      } else {
        vector_flag = 1;
        size_vector = selection_length_;
        memory->create(vector, size_vector, "mff/torch/phys:vector");
      }
      break;
    case Mode::ATOM_VALUES:
      peratom_flag = 1;
      if (use_peratom_vector_output_) {
        size_peratom_cols = 0;
      } else {
        size_peratom_cols = selection_length_;
      }
      break;
  }
}

ComputeMFFTorchPhys::~ComputeMFFTorchPhys() {
  if (vector_flag && vector) memory->destroy(vector);
  if (peratom_flag && use_peratom_vector_output_ && vector_atom) memory->destroy(vector_atom);
  if (peratom_flag && array_atom) memory->destroy(array_atom);
}

void ComputeMFFTorchPhys::parse_mode(const std::string &mode) {
  if (mode == "global") {
    mode_ = Mode::GLOBAL_VALUES;
  } else if (mode == "global/mask") {
    mode_ = Mode::GLOBAL_MASK;
  } else if (mode == "atom") {
    mode_ = Mode::ATOM_VALUES;
  } else if (mode == "atom/mask") {
    mode_ = Mode::ATOM_MASK;
  } else {
    error->all(FLERR,
               "compute mff/torch/phys mode must be one of: global, global/mask, atom, atom/mask");
  }
}

void ComputeMFFTorchPhys::parse_selection(int narg, char **arg) {
  selection_offset_ = 0;
  if (mode_ == Mode::GLOBAL_MASK || mode_ == Mode::ATOM_MASK) {
    selection_length_ = kPhysMaskCols;
  } else if (mode_ == Mode::GLOBAL_VALUES) {
    selection_length_ = kGlobalPhysValueCols;
  } else {
    selection_length_ = kAtomPhysValueCols;
  }
  use_scalar_output_ = false;
  use_peratom_vector_output_ = false;

  if (narg < 5) return;

  const std::string quantity(arg[4]);
  const bool has_component = (narg >= 6);
  const std::string component = has_component ? std::string(arg[5]) : std::string();
  const std::string quantity_norm = normalize_quantity_name(quantity);

  if (mode_ == Mode::GLOBAL_VALUES || mode_ == Mode::ATOM_VALUES) {
    if (mode_ == Mode::GLOBAL_VALUES && quantity_norm == "born_effective_charge") {
      error->all(FLERR,
                 "compute mff/torch/phys global mode does not expose born_effective_charge; use atom or atom/mask");
    }
    int offset = 0;
    int length = 0;
    bool scalar_output = false;
    const std::string *component_ptr = has_component ? &component : nullptr;
    if (!lookup_value_selection(quantity, component_ptr, offset, length, scalar_output)) {
      error->all(FLERR,
                 "compute mff/torch/phys value selection must use charge, dipole[x|y|z], polarizability[xx..zz], quadrupole[xx..zz], or born_effective_charge[xx..zz]");
    }
    selection_offset_ = offset;
    selection_length_ = length;
    if (mode_ == Mode::GLOBAL_VALUES) {
      use_scalar_output_ = scalar_output;
    } else {
      use_peratom_vector_output_ = scalar_output;
    }
  } else {
    if (has_component) {
      error->all(FLERR, "compute mff/torch/phys mask selections do not take a tensor component");
    }
    int offset = 0;
    if (!lookup_mask_selection(quantity, offset)) {
      error->all(FLERR,
                 "compute mff/torch/phys mask selection must use charge, dipole, polarizability, quadrupole, or born_effective_charge");
    }
    selection_offset_ = offset;
    selection_length_ = 1;
    use_scalar_output_ = true;
  }
}

void ComputeMFFTorchPhys::init() {
  if (force->pair == nullptr) {
    error->all(FLERR, "compute mff/torch/phys requires an active pair style");
  }
  pair_mfftorch_ = dynamic_cast<PairMFFTorch *>(force->pair);
  if (pair_mfftorch_ == nullptr) {
    error->all(FLERR, "compute mff/torch/phys requires pair_style mff/torch or mff/torch/kk");
  }
  pair_mfftorch_->set_physical_cache_requested(true);
}

void ComputeMFFTorchPhys::require_current_cache() const {
  if (pair_mfftorch_ == nullptr) {
    error->all(FLERR, "compute mff/torch/phys was not initialized with pair_style mff/torch");
  }
  const int64_t cached_timestep = pair_mfftorch_->cached_phys_timestep();
  const int64_t current_timestep = update ? static_cast<int64_t>(update->ntimestep) : -1;
  if (cached_timestep != current_timestep) {
    error->all(FLERR,
               "compute mff/torch/phys requested physical tensors before pair_style mff/torch cached the current timestep; use it during a run or after run 0");
  }
}

void ComputeMFFTorchPhys::copy_global_tensor_to_scalar(const torch::Tensor &src, int total_cols) {
  scalar = 0.0;
  if (!src.defined() || src.numel() == 0) return;

  auto values = src.contiguous().view({-1, total_cols});
  const double *ptr = values.data_ptr<double>();
  scalar = ptr[selection_offset_];
}

void ComputeMFFTorchPhys::copy_global_tensor_to_vector(const torch::Tensor &src, int total_cols) {
  for (int i = 0; i < selection_length_; ++i) vector[i] = 0.0;
  if (!src.defined() || src.numel() == 0) return;

  auto values = src.contiguous().view({-1, total_cols});
  const double *ptr = values.data_ptr<double>();
  for (int i = 0; i < selection_length_; ++i) vector[i] = ptr[selection_offset_ + i];
}

void ComputeMFFTorchPhys::copy_atom_tensor_to_vector(const torch::Tensor &src, int total_cols) {
  if (atom->nmax > nmax_atom_) {
    if (vector_atom) memory->destroy(vector_atom);
    nmax_atom_ = atom->nmax;
    memory->create(vector_atom, nmax_atom_, "mff/torch/phys:vector_atom");
  }

  const int nlocal = atom->nlocal;
  int *mask = atom->mask;
  for (int i = 0; i < nlocal; ++i) vector_atom[i] = 0.0;
  if (!src.defined() || src.numel() == 0) return;

  auto values = src.contiguous().view({-1, total_cols});
  const double *ptr = values.data_ptr<double>();
  for (int i = 0; i < nlocal; ++i) {
    if (!(mask[i] & groupbit)) continue;
    vector_atom[i] = ptr[i * total_cols + selection_offset_];
  }
}

void ComputeMFFTorchPhys::copy_atom_tensor_to_array(const torch::Tensor &src, int total_cols) {
  if (atom->nmax > nmax_atom_) {
    if (array_atom) memory->destroy(array_atom);
    nmax_atom_ = atom->nmax;
    memory->create(array_atom, nmax_atom_, selection_length_, "mff/torch/phys:array_atom");
  }

  const int nlocal = atom->nlocal;
  int *mask = atom->mask;
  for (int i = 0; i < nlocal; ++i) {
    for (int j = 0; j < selection_length_; ++j) array_atom[i][j] = 0.0;
  }
  if (!src.defined() || src.numel() == 0) return;

  auto values = src.contiguous().view({-1, total_cols});
  const double *ptr = values.data_ptr<double>();
  for (int i = 0; i < nlocal; ++i) {
    if (!(mask[i] & groupbit)) continue;
    for (int j = 0; j < selection_length_; ++j) {
      array_atom[i][j] = ptr[i * total_cols + selection_offset_ + j];
    }
  }
}

double ComputeMFFTorchPhys::compute_scalar() {
  invoked_scalar = update->ntimestep;
  require_current_cache();

  switch (mode_) {
    case Mode::GLOBAL_VALUES:
      copy_global_tensor_to_scalar(pair_mfftorch_->global_phys(), kGlobalPhysValueCols);
      break;
    case Mode::GLOBAL_MASK:
      copy_global_tensor_to_scalar(pair_mfftorch_->global_phys_mask(), kPhysMaskCols);
      break;
    case Mode::ATOM_MASK:
      copy_global_tensor_to_scalar(pair_mfftorch_->atom_phys_mask(), kPhysMaskCols);
      break;
    case Mode::ATOM_VALUES:
      error->all(FLERR, "compute mff/torch/phys atom mode does not provide a global scalar");
      break;
  }
  return scalar;
}

void ComputeMFFTorchPhys::compute_vector() {
  invoked_vector = update->ntimestep;
  require_current_cache();

  switch (mode_) {
    case Mode::GLOBAL_VALUES:
      if (use_scalar_output_) {
        copy_global_tensor_to_scalar(pair_mfftorch_->global_phys(), kGlobalPhysValueCols);
      } else {
        copy_global_tensor_to_vector(pair_mfftorch_->global_phys(), kGlobalPhysValueCols);
      }
      break;
    case Mode::GLOBAL_MASK:
      if (use_scalar_output_) {
        copy_global_tensor_to_scalar(pair_mfftorch_->global_phys_mask(), kPhysMaskCols);
      } else {
        copy_global_tensor_to_vector(pair_mfftorch_->global_phys_mask(), kPhysMaskCols);
      }
      break;
    case Mode::ATOM_MASK:
      if (use_scalar_output_) {
        copy_global_tensor_to_scalar(pair_mfftorch_->atom_phys_mask(), kPhysMaskCols);
      } else {
        copy_global_tensor_to_vector(pair_mfftorch_->atom_phys_mask(), kPhysMaskCols);
      }
      break;
    case Mode::ATOM_VALUES:
      error->all(FLERR, "compute mff/torch/phys atom mode provides per-atom arrays, not a global vector");
      break;
  }
}

void ComputeMFFTorchPhys::compute_peratom() {
  invoked_peratom = update->ntimestep;
  require_current_cache();

  if (mode_ != Mode::ATOM_VALUES) {
    error->all(FLERR, "compute mff/torch/phys per-atom access requires atom mode");
  }
  if (use_peratom_vector_output_) {
    copy_atom_tensor_to_vector(pair_mfftorch_->atom_phys(), kAtomPhysValueCols);
  } else {
    copy_atom_tensor_to_array(pair_mfftorch_->atom_phys(), kAtomPhysValueCols);
  }
}
