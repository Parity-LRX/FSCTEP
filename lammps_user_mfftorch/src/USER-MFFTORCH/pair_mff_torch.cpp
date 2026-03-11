#include "pair_mff_torch.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"
#include "variable.h"

#include "mff_periodic_table.h"
#include "mff_reciprocal_solver.h"
#include "mff_tree_fmm_solver.h"
#include "mff_torch_engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace LAMMPS_NS;

namespace {

std::string normalize_variable_name(const std::string &name) {
  if (name.rfind("v_", 0) == 0) return name.substr(2);
  return name;
}

struct CellGeom {
  double cell[3][3];
  double inv[3][3];
  int pbc[3];
};

CellGeom build_cell_geom(const LAMMPS_NS::Domain *domain) {
  CellGeom g{};
  g.cell[0][0] = domain->xprd;
  g.cell[0][1] = 0.0;
  g.cell[0][2] = 0.0;
  g.cell[1][0] = domain->xy;
  g.cell[1][1] = domain->yprd;
  g.cell[1][2] = 0.0;
  g.cell[2][0] = domain->xz;
  g.cell[2][1] = domain->yz;
  g.cell[2][2] = domain->zprd;
  g.pbc[0] = domain->xperiodic;
  g.pbc[1] = domain->yperiodic;
  g.pbc[2] = domain->zperiodic;

  const double a = g.cell[0][0], b = g.cell[0][1], c = g.cell[0][2];
  const double d = g.cell[1][0], e = g.cell[1][1], f = g.cell[1][2];
  const double h = g.cell[2][0], i = g.cell[2][1], j = g.cell[2][2];
  const double det = a * (e * j - f * i) - b * (d * j - f * h) + c * (d * i - e * h);
  if (std::abs(det) < 1e-12) {
    throw std::runtime_error("mff/torch encountered a singular cell matrix");
  }
  const double inv_det = 1.0 / det;
  g.inv[0][0] = (e * j - f * i) * inv_det;
  g.inv[0][1] = (c * i - b * j) * inv_det;
  g.inv[0][2] = (b * f - c * e) * inv_det;
  g.inv[1][0] = (f * h - d * j) * inv_det;
  g.inv[1][1] = (a * j - c * h) * inv_det;
  g.inv[1][2] = (c * d - a * f) * inv_det;
  g.inv[2][0] = (d * i - e * h) * inv_det;
  g.inv[2][1] = (b * h - a * i) * inv_det;
  g.inv[2][2] = (a * e - b * d) * inv_det;
  return g;
}

inline int nearest_int(double x) {
  return (x >= 0.0) ? static_cast<int>(x + 0.5) : static_cast<int>(x - 0.5);
}

mfftorch::ReciprocalInputs make_reciprocal_inputs(
    MPI_Comm world,
    const torch::Tensor& local_pos,
    const torch::Tensor& local_source,
    const torch::Tensor& cell,
    const CellGeom& geom,
    bool need_energy,
    const torch::Device& preferred_device) {
  mfftorch::ReciprocalInputs inputs;
  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(world, &world_rank);
  MPI_Comm_size(world, &world_size);
  int64_t local_n = local_pos.defined() ? local_pos.size(0) : 0;
  int64_t global_offset = 0;
  MPI_Exscan(&local_n, &global_offset, 1, MPI_LONG_LONG, MPI_SUM, world);
  if (world_rank == 0) global_offset = 0;
  inputs.world = world;
  inputs.local_pos = local_pos;
  inputs.local_source = local_source;
  inputs.local_global_ids =
      torch::arange(global_offset, global_offset + local_n, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  inputs.cell = cell;
  inputs.pbc = {geom.pbc[0], geom.pbc[1], geom.pbc[2]};
  inputs.need_energy = need_energy;
  inputs.preferred_device = preferred_device;
  inputs.world_rank = world_rank;
  inputs.world_size = world_size;
  return inputs;
}

}  // namespace

PairMFFTorch::PairMFFTorch(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

PairMFFTorch::~PairMFFTorch() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairMFFTorch::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) {
      setflag[i][j] = 0;
      cutsq[i][j] = 0.0;
    }
  }
}

void PairMFFTorch::settings(int narg, char **arg) {
  if (narg < 1) error->all(FLERR, "Illegal pair_style mff/torch command");
  cut_global_ = utils::numeric(FLERR, arg[0], false, lmp);
  if (cut_global_ <= 0.0) error->all(FLERR, "pair_style mff/torch cutoff must be > 0");
  cutsq_global_ = cut_global_ * cut_global_;

  use_external_field_ = false;
  external_tensor_rank_ = 0;
  external_field_symmetric_rank2_ = false;
  external_field_var_names_.clear();
  cached_external_field_values_.clear();
  external_tensor_cache_ = torch::Tensor();

  for (int i = 1; i < narg; ++i) {
    const std::string opt(arg[i]);
    if (opt == "cpu" || opt == "cuda") {
      device_str_ = opt;
      continue;
    }
    if (opt == "field") {
      if (i + 3 >= narg) {
        error->all(FLERR, "pair_style mff/torch field expects three equal-style variables: v_Ex v_Ey v_Ez");
      }
      use_external_field_ = true;
      external_tensor_rank_ = 1;
      external_field_symmetric_rank2_ = false;
      external_field_var_names_ = {
          normalize_variable_name(arg[i + 1]),
          normalize_variable_name(arg[i + 2]),
          normalize_variable_name(arg[i + 3]),
      };
      cached_external_field_values_.assign(3, 0.0f);
      i += 3;
      continue;
    }
    if (opt == "field9") {
      if (i + 9 >= narg) {
        error->all(FLERR,
                   "pair_style mff/torch field9 expects nine equal-style variables "
                   "(row-major: xx xy xz yx yy yz zx zy zz)");
      }
      use_external_field_ = true;
      external_tensor_rank_ = 2;
      external_field_symmetric_rank2_ = false;
      external_field_var_names_.clear();
      for (int k = 1; k <= 9; ++k) external_field_var_names_.push_back(normalize_variable_name(arg[i + k]));
      cached_external_field_values_.assign(9, 0.0f);
      i += 9;
      continue;
    }
    if (opt == "field6") {
      if (i + 6 >= narg) {
        error->all(FLERR,
                   "pair_style mff/torch field6 expects six equal-style variables "
                   "(symmetric order: xx yy zz xy xz yz)");
      }
      use_external_field_ = true;
      external_tensor_rank_ = 2;
      external_field_symmetric_rank2_ = true;
      external_field_var_names_.clear();
      for (int k = 1; k <= 6; ++k) external_field_var_names_.push_back(normalize_variable_name(arg[i + k]));
      cached_external_field_values_.assign(6, 0.0f);
      i += 6;
      continue;
    }
    error->all(FLERR, ("Unknown pair_style mff/torch option: " + opt).c_str());
  }
}

void PairMFFTorch::coeff(int narg, char **arg) {
  if (!allocated) allocate();
  if (narg < 3) error->all(FLERR, "Illegal pair_coeff command for mff/torch");

  // Expect: pair_coeff * * core.pt <elem1> <elem2> ... (ntypes entries)
  // arg[0], arg[1] are * *
  core_pt_path_ = std::string(arg[2]);

  const int ntypes = atom->ntypes;
  if (narg != 3 + ntypes) error->all(FLERR, "pair_coeff mff/torch expects one element symbol per atom type");

  type2Z_.assign(ntypes + 1, 0);
  for (int itype = 1; itype <= ntypes; itype++) {
    const std::string sym(arg[2 + itype]);
    if (sym == "NULL" || sym == "null") {
      type2Z_[itype] = 0;
      continue;
    }
    int Z = mfftorch::symbol_to_Z(sym);
    if (Z <= 0) error->all(FLERR, ("Unknown element symbol in pair_coeff mff/torch: " + sym).c_str());
    type2Z_[itype] = static_cast<int64_t>(Z);
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutsq_global_;
      setflag[j][i] = 1;
      cutsq[j][i] = cutsq_global_;
    }
  }

  if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
  engine_loaded_ = false;  // lazy load at init_style/compute
}

void PairMFFTorch::init_style() {
  if (core_pt_path_.empty()) error->all(FLERR, "pair_coeff for mff/torch must specify core.pt path");

  // Request a full neighbor list.
  neighbor->add_request(this, NeighConst::REQ_FULL);

  try {
    if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
    if (!reciprocal_solver_) reciprocal_solver_ = std::make_unique<mfftorch::MFFReciprocalSolver>();
    if (!tree_fmm_solver_) tree_fmm_solver_ = std::make_unique<mfftorch::MFFTreeFmmSolver>();
    engine_->load_core(core_pt_path_, device_str_);
    if (reciprocal_solver_) {
      auto cfg = reciprocal_solver_->config();
      cfg.slab_padding_factor = static_cast<int>(engine_->reciprocal_source_slab_padding_factor());
      cfg.green_mode = (engine_->long_range_green_mode() == "learned_poisson")
                           ? mfftorch::ReciprocalGreenMode::LearnedPoisson
                           : mfftorch::ReciprocalGreenMode::Poisson;
      reciprocal_solver_->set_config(cfg);
    }
    if (tree_fmm_solver_) {
      mfftorch::TreeFmmConfig cfg;
      cfg.theta = engine_->long_range_theta();
      cfg.leaf_size = static_cast<int>(engine_->long_range_leaf_size());
      cfg.multipole_order = static_cast<int>(engine_->long_range_multipole_order());
      cfg.neutralize = engine_->long_range_neutralize();
      cfg.screening = engine_->long_range_screening();
      cfg.softening = engine_->long_range_softening();
      cfg.energy_scale = engine_->long_range_energy_scale();
      cfg.boundary = engine_->long_range_boundary();
      cfg.energy_partition = engine_->long_range_energy_partition();
      tree_fmm_solver_->set_config(cfg);
    }
    validate_external_field_configuration();
    engine_loaded_ = true;
    engine_->warmup(32, 256);
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("Failed to load TorchScript core: ") + e.what()).c_str());
  }
}

double PairMFFTorch::init_one(int i, int j) {
  return cut_global_;
}

void PairMFFTorch::validate_external_field_configuration() {
  if (!engine_) return;

  if (use_external_field_) {
    if (!engine_->accepts_external_tensor()) {
      error->all(FLERR,
                 "pair_style mff/torch field was specified, but core.pt does not accept external_tensor");
    }
    const int expected_nvars = (external_tensor_rank_ == 1) ? 3 : (external_field_symmetric_rank2_ ? 6 : 9);
    if (static_cast<int>(external_field_var_names_.size()) != expected_nvars) {
      error->all(FLERR, "mff/torch external field variable count does not match the selected field mode");
    }
    for (const auto &name : external_field_var_names_) {
      if (name.empty()) error->all(FLERR, "pair_style mff/torch external field variable name is empty");
      const int ivar = input->variable->find(name.c_str());
      if (ivar < 0) {
        error->all(FLERR, ("Unknown LAMMPS variable for mff/torch field: " + name).c_str());
      }
      if (!input->variable->equalstyle(ivar)) {
        error->all(FLERR,
                   ("mff/torch field variables must be equal-style scalars: " + name).c_str());
      }
    }
  } else if (engine_->accepts_external_tensor()) {
    error->all(FLERR,
               "core.pt requires external_tensor, but pair_style mff/torch was not given field/field6/field9");
  }
}

torch::Tensor PairMFFTorch::current_external_tensor(const torch::Device& device) {
  if (!use_external_field_) return torch::Tensor();

  std::vector<float> values(external_field_var_names_.size(), 0.0f);
  for (size_t k = 0; k < external_field_var_names_.size(); ++k) {
    const int ivar = input->variable->find(external_field_var_names_[k].c_str());
    if (ivar < 0) {
      error->all(FLERR, ("Unknown LAMMPS variable for mff/torch field: " + external_field_var_names_[k]).c_str());
    }
    values[k] = static_cast<float>(input->variable->compute_equal(ivar));
  }

  const bool cache_hit =
      external_tensor_cache_.defined() &&
      external_tensor_cache_.device() == device &&
      cached_external_field_values_ == values;
  if (cache_hit) return external_tensor_cache_;

  cached_external_field_values_ = values;
  torch::Tensor cpu;
  if (external_tensor_rank_ == 1) {
    cpu = torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  } else if (external_tensor_rank_ == 2 && external_field_symmetric_rank2_) {
    cpu = torch::tensor(
              {
                  values[0], values[3], values[4],
                  values[3], values[1], values[5],
                  values[4], values[5], values[2],
              },
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .reshape({3, 3});
  } else if (external_tensor_rank_ == 2) {
    cpu = torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
              .reshape({3, 3});
  } else {
    error->all(FLERR, "Unsupported external tensor rank for mff/torch");
  }
  external_tensor_cache_ = (device.is_cpu()) ? cpu : cpu.to(device);
  return external_tensor_cache_;
}

void PairMFFTorch::reset_physical_outputs() {
  global_phys_cpu_ = torch::Tensor();
  atom_phys_cpu_ = torch::Tensor();
  global_phys_mask_cpu_ = torch::Tensor();
  atom_phys_mask_cpu_ = torch::Tensor();
  cached_phys_timestep_ = update ? static_cast<int64_t>(update->ntimestep) : -1;
}

void PairMFFTorch::cache_physical_outputs(const mfftorch::MFFOutputs& out, int nlocal) {
  cached_phys_timestep_ = update ? static_cast<int64_t>(update->ntimestep) : -1;
  if (!physical_cache_requested_) {
    global_phys_cpu_ = torch::Tensor();
    atom_phys_cpu_ = torch::Tensor();
    global_phys_mask_cpu_ = torch::Tensor();
    atom_phys_mask_cpu_ = torch::Tensor();
    return;
  }

  if (out.global_phys.defined()) {
    global_phys_cpu_ = out.global_phys.to(torch::kCPU, torch::kFloat64).contiguous();
  } else {
    global_phys_cpu_ = torch::Tensor();
  }
  if (out.atom_phys.defined()) {
    auto atom_phys = out.atom_phys.to(torch::kCPU, torch::kFloat64).contiguous();
    if (atom_phys.dim() >= 2 && atom_phys.size(0) >= nlocal) {
      atom_phys_cpu_ = atom_phys.narrow(0, 0, nlocal).clone();
    } else {
      atom_phys_cpu_ = atom_phys.clone();
    }
  } else {
    atom_phys_cpu_ = torch::Tensor();
  }
  global_phys_mask_cpu_ = out.global_phys_mask.defined()
                              ? out.global_phys_mask.to(torch::kCPU, torch::kFloat64).contiguous()
                              : torch::Tensor();
  atom_phys_mask_cpu_ = out.atom_phys_mask.defined()
                            ? out.atom_phys_mask.to(torch::kCPU, torch::kFloat64).contiguous()
                            : torch::Tensor();
}

void PairMFFTorch::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  reset_physical_outputs();

  if (!engine_loaded_) init_style();

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int ntotal = nlocal + nghost;

  // Neighbor list
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  // Build type->Z mapped A (CPU then move to engine device).
  std::vector<int64_t> A_cpu(ntotal);
  std::vector<float> pos_cpu;
  pos_cpu.reserve(static_cast<size_t>(ntotal) * 3);
  for (int i = 0; i < ntotal; i++) {
    const int itype = type[i];
    const int64_t Z = (itype >= 0 && itype < static_cast<int>(type2Z_.size())) ? type2Z_[itype] : 0;
    A_cpu[i] = Z;
    pos_cpu.push_back(static_cast<float>(x[i][0]));
    pos_cpu.push_back(static_cast<float>(x[i][1]));
    pos_cpu.push_back(static_cast<float>(x[i][2]));
  }

  const CellGeom geom = build_cell_geom(domain);
  float cell_cpu[9] = {
      static_cast<float>(geom.cell[0][0]), static_cast<float>(geom.cell[0][1]), static_cast<float>(geom.cell[0][2]),
      static_cast<float>(geom.cell[1][0]), static_cast<float>(geom.cell[1][1]), static_cast<float>(geom.cell[1][2]),
      static_cast<float>(geom.cell[2][0]), static_cast<float>(geom.cell[2][1]), static_cast<float>(geom.cell[2][2]),
  };
  auto cell_t = torch::from_blob(cell_cpu, {1, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
  if (nlocal == 0) {
    const bool exports_runtime_source = engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0;
    const bool use_tree_fmm =
        tree_fmm_solver_ && exports_runtime_source && engine_->long_range_runtime_backend() == "tree_fmm";
    const bool use_reciprocal =
        reciprocal_solver_ && exports_runtime_source && !use_tree_fmm;
    if (use_tree_fmm || use_reciprocal) {
      auto empty_pos = torch::zeros({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      auto empty_source = torch::zeros(
          {0, engine_->reciprocal_source_channels()},
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
      try {
        auto reciprocal_inputs = make_reciprocal_inputs(
            world,
            empty_pos,
            empty_source,
            cell_t,
            geom,
            static_cast<bool>(eflag),
            torch::Device(torch::kCPU));
        if (use_tree_fmm) {
          (void)tree_fmm_solver_->compute(reciprocal_inputs);
        } else {
          (void)reciprocal_solver_->compute(reciprocal_inputs);
        }
      } catch (const std::exception &e) {
        error->all(FLERR, (std::string("mff/torch runtime long-range solver failed on empty rank: ") + e.what()).c_str());
      }
    }
    return;
  }

  // Count edges (upper bound) and build edges + lattice shifts.
  int64_t Emax = 0;
  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    Emax += numneigh[i];
  }
  std::vector<int64_t> edge_src_cpu;
  std::vector<int64_t> edge_dst_cpu;
  std::vector<float> edge_shifts_cpu;
  edge_src_cpu.reserve(static_cast<size_t>(Emax));
  edge_dst_cpu.reserve(static_cast<size_t>(Emax));
  edge_shifts_cpu.reserve(static_cast<size_t>(Emax) * 3);

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      const double rawx = x[j][0] - x[i][0];
      const double rawy = x[j][1] - x[i][1];
      const double rawz = x[j][2] - x[i][2];
      const double fracx = rawx * geom.inv[0][0] + rawy * geom.inv[1][0] + rawz * geom.inv[2][0];
      const double fracy = rawx * geom.inv[0][1] + rawy * geom.inv[1][1] + rawz * geom.inv[2][1];
      const double fracz = rawx * geom.inv[0][2] + rawy * geom.inv[1][2] + rawz * geom.inv[2][2];
      const int sx = geom.pbc[0] ? -nearest_int(fracx) : 0;
      const int sy = geom.pbc[1] ? -nearest_int(fracy) : 0;
      const int sz = geom.pbc[2] ? -nearest_int(fracz) : 0;
      const double shiftx = sx * geom.cell[0][0] + sy * geom.cell[1][0] + sz * geom.cell[2][0];
      const double shifty = sx * geom.cell[0][1] + sy * geom.cell[1][1] + sz * geom.cell[2][1];
      const double shiftz = sx * geom.cell[0][2] + sy * geom.cell[1][2] + sz * geom.cell[2][2];
      const double delx = rawx + shiftx;
      const double dely = rawy + shifty;
      const double delz = rawz + shiftz;
      const double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq > cutsq_global_) continue;

      edge_src_cpu.push_back(static_cast<int64_t>(i));
      edge_dst_cpu.push_back(static_cast<int64_t>(j));
      edge_shifts_cpu.push_back(static_cast<float>(sx));
      edge_shifts_cpu.push_back(static_cast<float>(sy));
      edge_shifts_cpu.push_back(static_cast<float>(sz));
    }
  }

  const int64_t E = static_cast<int64_t>(edge_src_cpu.size());
  if (E <= 1) return;

  // Torch tensors (CPU -> device copy).
  auto pos_t = torch::from_blob(pos_cpu.data(), {ntotal, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
  auto A_t = torch::from_blob(A_cpu.data(), {ntotal}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).clone();
  auto edge_src_t = torch::from_blob(edge_src_cpu.data(), {E}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).clone();
  auto edge_dst_t = torch::from_blob(edge_dst_cpu.data(), {E}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).clone();
  auto edge_shifts_t = torch::from_blob(edge_shifts_cpu.data(), {E, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
  auto external_tensor_t = current_external_tensor(torch::kCPU);

  const bool want_atom_virial = static_cast<bool>(vflag_atom);
  mfftorch::MFFOutputs out;
  try {
    out = engine_->compute(nlocal, ntotal, pos_t, A_t, edge_src_t, edge_dst_t, edge_shifts_t, cell_t, external_tensor_t,
                           static_cast<bool>(eflag), want_atom_virial);
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("mff/torch engine compute failed: ") + e.what()).c_str());
  }
  cache_physical_outputs(out, nlocal);

  mfftorch::ReciprocalOutputs reciprocal_out;
  const bool exports_runtime_source = engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0;
  const bool use_tree_fmm =
      tree_fmm_solver_ && exports_runtime_source && engine_->long_range_runtime_backend() == "tree_fmm";
  const bool use_reciprocal =
      reciprocal_solver_ && exports_runtime_source && !use_tree_fmm;
  const bool use_runtime_long_range = use_tree_fmm || use_reciprocal;
  if (use_tree_fmm || use_reciprocal) {
    try {
      const auto reciprocal_device = engine_->device();
      auto local_source = out.reciprocal_source.defined()
                              ? out.reciprocal_source.narrow(0, 0, nlocal).to(reciprocal_device, torch::kFloat32).contiguous()
                              : torch::zeros(
                                    {nlocal, engine_->reciprocal_source_channels()},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(reciprocal_device));
      if (use_tree_fmm && engine_->long_range_source_kind() != "latent_charge") {
        throw std::runtime_error("tree_fmm runtime currently requires long_range_source_kind=latent_charge");
      }
      auto reciprocal_inputs = make_reciprocal_inputs(
          world,
          pos_t.narrow(0, 0, nlocal).to(reciprocal_device, torch::kFloat32).contiguous(),
          local_source,
          cell_t.to(reciprocal_device, torch::kFloat32).contiguous(),
          geom,
          static_cast<bool>(eflag),
          reciprocal_device);
      reciprocal_out = use_tree_fmm ? tree_fmm_solver_->compute(reciprocal_inputs)
                                    : reciprocal_solver_->compute(reciprocal_inputs);
    } catch (const std::exception &e) {
      error->all(FLERR, (std::string("mff/torch runtime long-range solver failed: ") + e.what()).c_str());
    }
  }

  if (eflag) eng_vdwl += out.energy;
  if (use_runtime_long_range) eng_vdwl += reciprocal_out.energy;

  // When virial is needed, ghost forces must be in f[] for virial_fdotr_compute()
  // to produce correct results (it sums over nall = nlocal + nghost).
  const int nwrite = (force->newton_pair || vflag_fdotr) ? ntotal : nlocal;
  auto forces_cpu = out.forces.to(torch::kCPU, torch::kFloat64).contiguous();
  const double *fp = forces_cpu.data_ptr<double>();
  for (int i = 0; i < nwrite; i++) {
    f[i][0] += fp[i * 3 + 0];
    f[i][1] += fp[i * 3 + 1];
    f[i][2] += fp[i * 3 + 2];
  }
  if (use_runtime_long_range && reciprocal_out.forces_local.defined()) {
    auto reciprocal_forces_cpu = reciprocal_out.forces_local.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *rfp = reciprocal_forces_cpu.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) {
      f[i][0] += rfp[i * 3 + 0];
      f[i][1] += rfp[i * 3 + 1];
      f[i][2] += rfp[i * 3 + 2];
    }
  }

  if (eflag_atom && eatom && out.atom_energy.defined()) {
    auto ae_cpu = out.atom_energy.to(torch::kCPU, torch::kFloat64).contiguous().view({ntotal});
    const double *ep = ae_cpu.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }
  if (eflag_atom && eatom && use_runtime_long_range && reciprocal_out.atom_energy_local.defined()) {
    auto ae_recip = reciprocal_out.atom_energy_local.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *ep = ae_recip.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }

  if (vflag_atom && vatom && out.atom_virial.defined()) {
    auto vir_cpu = out.atom_virial.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *vp = vir_cpu.data_ptr<double>();
    const int nvir = force->newton_pair ? ntotal : nlocal;
    for (int i = 0; i < nvir; i++) {
      vatom[i][0] += vp[i * 6 + 0];
      vatom[i][1] += vp[i * 6 + 1];
      vatom[i][2] += vp[i * 6 + 2];
      vatom[i][3] += vp[i * 6 + 3];
      vatom[i][4] += vp[i * 6 + 4];
      vatom[i][5] += vp[i * 6 + 5];
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
