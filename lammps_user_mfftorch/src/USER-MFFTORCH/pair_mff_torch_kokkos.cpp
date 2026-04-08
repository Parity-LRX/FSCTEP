#include "pair_mff_torch_kokkos.h"

#ifdef LMP_KOKKOS

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "neigh_list.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"

#include "mff_torch_engine.h"
#include "mff_reciprocal_solver.h"
#include "mff_tree_fmm_solver.h"

#include <Kokkos_Core.hpp>
#include <chrono>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace LAMMPS_NS;

namespace {

struct CellGeom {
  float cell[3][3];
  float inv[3][3];
  int pbc[3];
};

CellGeom build_cell_geom(const LAMMPS_NS::Domain* domain) {
  CellGeom g{};
  g.cell[0][0] = static_cast<float>(domain->xprd);
  g.cell[0][1] = 0.0f;
  g.cell[0][2] = 0.0f;
  g.cell[1][0] = static_cast<float>(domain->xy);
  g.cell[1][1] = static_cast<float>(domain->yprd);
  g.cell[1][2] = 0.0f;
  g.cell[2][0] = static_cast<float>(domain->xz);
  g.cell[2][1] = static_cast<float>(domain->yz);
  g.cell[2][2] = static_cast<float>(domain->zprd);
  g.pbc[0] = domain->xperiodic;
  g.pbc[1] = domain->yperiodic;
  g.pbc[2] = domain->zperiodic;

  const double a = g.cell[0][0], b = g.cell[0][1], c = g.cell[0][2];
  const double d = g.cell[1][0], e = g.cell[1][1], f = g.cell[1][2];
  const double h = g.cell[2][0], i = g.cell[2][1], j = g.cell[2][2];
  const double det = a * (e * j - f * i) - b * (d * j - f * h) + c * (d * i - e * h);
  if (std::abs(det) < 1e-12) {
    throw std::runtime_error("mff/torch/kk encountered a singular cell matrix");
  }
  const double inv_det = 1.0 / det;
  g.inv[0][0] = static_cast<float>((e * j - f * i) * inv_det);
  g.inv[0][1] = static_cast<float>((c * i - b * j) * inv_det);
  g.inv[0][2] = static_cast<float>((b * f - c * e) * inv_det);
  g.inv[1][0] = static_cast<float>((f * h - d * j) * inv_det);
  g.inv[1][1] = static_cast<float>((a * j - c * h) * inv_det);
  g.inv[1][2] = static_cast<float>((c * d - a * f) * inv_det);
  g.inv[2][0] = static_cast<float>((d * i - e * h) * inv_det);
  g.inv[2][1] = static_cast<float>((b * h - a * i) * inv_det);
  g.inv[2][2] = static_cast<float>((a * e - b * d) * inv_det);
  return g;
}

KOKKOS_INLINE_FUNCTION
int nearest_int_device(const float x) {
  return (x >= 0.0f) ? static_cast<int>(x + 0.5f) : static_cast<int>(x - 0.5f);
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

template <class DeviceType>
PairMFFTorchKokkos<DeviceType>::PairMFFTorchKokkos(LAMMPS *lmp) : PairMFFTorch(lmp) {
  kokkosable = 1;
  atomKK = (AtomKokkos *)atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
}

template <class DeviceType>
void PairMFFTorchKokkos<DeviceType>::init_style() {
  const bool debug_bundle = std::getenv("MFF_DEBUG_BUNDLE") != nullptr;
  if (core_pt_path_.empty()) error->all(FLERR, "pair_coeff for mff/torch must specify core.pt path");

  // Request a full neighbor list (same as base class).
  neighbor->add_request(this, NeighConst::REQ_FULL);

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> && !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();

  if (!engine_) engine_ = std::make_unique<mfftorch::MFFTorchEngine>();
  if (!engine_loaded_) {
    // Each MPI rank uses its own GPU: gpu_id = local_rank % num_gpus.
    std::string dev = device_str_;
    if (dev == "cuda") {
      int ngpus = lmp->kokkos->ngpus;
      int gpu_id = 0;
      if (ngpus > 0) {
        gpu_id = comm->me % ngpus;
      }
      dev = "cuda:" + std::to_string(gpu_id);
    }
    try {
      if (!reciprocal_solver_) reciprocal_solver_ = std::make_unique<mfftorch::MFFReciprocalSolver>();
      if (!tree_fmm_solver_) tree_fmm_solver_ = std::make_unique<mfftorch::MFFTreeFmmSolver>();
      if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style before load_core\n");
      engine_->load_core(core_pt_path_, dev);
      if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style after load_core\n");
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
      if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style after validate\n");
      engine_loaded_ = true;
    } catch (const std::exception &e) {
      error->all(FLERR, (std::string("Failed to load TorchScript core: ") + e.what()).c_str());
    }
  }

  if (engine_->is_cuda()) {
    if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style before initial warmup\n");
    auto t = torch::from_blob(type2Z_.data(), {(int64_t)type2Z_.size()},
                              torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                 .clone();
    type2Z_cuda_ = t.to(engine_->device());

    engine_->warmup(32, 256);
    if (debug_bundle) std::fprintf(stderr, "[USER-MFFTORCH] kk init_style after initial warmup\n");
  }
}

template <class DeviceType>
void PairMFFTorchKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
  const bool debug_kk_timings = []() {
    const char* env = std::getenv("MFF_DEBUG_KK_TIMINGS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  const auto t_total_start = std::chrono::steady_clock::now();
  auto t_last = t_total_start;
  double edge_count_ms = 0.0;
  double offset_scan_ms = 0.0;
  double fill_ms = 0.0;
  double prepare_ms = 0.0;
  double compute_ms = 0.0;
  double reciprocal_ms = 0.0;
  double force_ms = 0.0;
  double atom_output_ms = 0.0;
  double virial_ms = 0.0;
  auto finish_segment = [&](double& bucket) {
    if (!debug_kk_timings) return;
    Kokkos::fence();
    const auto now = std::chrono::steady_clock::now();
    bucket += std::chrono::duration<double, std::milli>(now - t_last).count();
    t_last = now;
  };

  if (!engine_loaded_) init_style();
  if (!engine_ || !engine_->is_cuda()) {
    PairMFFTorch::compute(eflag_in, vflag_in);
    return;
  }

  int eflag = eflag_in;
  int vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;
  // Use default allocation behavior so eatom/vatom are available when
  // computes like pe/atom or stress/atom request per-atom quantities.
  ev_init(eflag, vflag);
  reset_physical_outputs();

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  atomKK->modified(execution_space, F_MASK);

  auto x = atomKK->k_x.template view<DeviceType>();
  auto f = atomKK->k_f.template view<DeviceType>();
  auto type = atomKK->k_type.template view<DeviceType>();

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  const int ntotal = nall;

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;
  const int inum = list->inum;
  auto dev = engine_->device();
  const CellGeom geom = build_cell_geom(domain);
  if (nlocal == 0) {
    const std::array<float, 9> cell_values = {
        geom.cell[0][0], geom.cell[0][1], geom.cell[0][2],
        geom.cell[1][0], geom.cell[1][1], geom.cell[1][2],
        geom.cell[2][0], geom.cell[2][1], geom.cell[2][2],
    };
    if (!cached_cell_valid_ || cached_cell_values_ != cell_values || !cached_cell_t_.defined() ||
        cached_cell_t_.device() != dev) {
      cached_cell_values_ = cell_values;
      cached_cell_t_ = torch::from_blob(
                           cached_cell_values_.data(),
                           {1, 3, 3},
                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                           .clone()
                           .to(dev);
      cached_cell_valid_ = true;
    }
    auto cell_t = cached_cell_t_;
    const bool use_tree_fmm =
        tree_fmm_solver_ && engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0 &&
        engine_->long_range_runtime_backend() == "tree_fmm";
    const bool use_reciprocal =
        reciprocal_solver_ && engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0 &&
        !use_tree_fmm;
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
            cell_t.to(torch::kCPU, torch::kFloat32).contiguous(),
            geom,
            static_cast<bool>(eflag_global || eflag_atom),
            dev);
        if (use_tree_fmm) {
          (void)tree_fmm_solver_->compute(reciprocal_inputs);
        } else {
          (void)reciprocal_solver_->compute(reciprocal_inputs);
        }
      } catch (const std::exception &e) {
        error->all(FLERR, (std::string("mff/torch/kk runtime long-range solver failed on empty rank: ") + e.what()).c_str());
      }
    }
    return;
  }
  if (inum == 0) return;

  // Count total edges via device reduce.
  int64_t Etotal = 0;
  Kokkos::parallel_reduce(
      "mfftorch::count_edges", inum,
      KOKKOS_LAMBDA(const int ii, int64_t &acc) {
        acc += static_cast<int64_t>(d_numneigh[d_ilist[ii]]);
      },
      Etotal);
  finish_segment(edge_count_ms);
  if (Etotal <= 1) return;

  // Exclusive-scan offsets on device; reuse cached view when inum unchanged.
  if (cached_inum_ != inum) {
    cached_d_offsets_ = Kokkos::View<int64_t *, DeviceType>("mfftorch::offsets", inum);
    cached_inum_ = inum;
  }
  auto d_offsets = cached_d_offsets_;
  Kokkos::parallel_scan(
      "mfftorch::scan_offsets", inum,
      KOKKOS_LAMBDA(const int ii, int64_t &update, const bool is_final) {
        if (is_final) d_offsets(ii) = update;
        update += static_cast<int64_t>(d_numneigh[d_ilist[ii]]);
      });
  finish_segment(offset_scan_ms);

  // Reuse CUDA tensor buffers when edge count / atom count unchanged.
  using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

  if (cached_Etotal_ != Etotal) {
    buf_edge_src_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_edge_dst_ = torch::empty({Etotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_edge_shifts_ = torch::empty({Etotal, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    cached_Etotal_ = Etotal;
  }
  if (cached_ntotal_ != static_cast<int64_t>(ntotal)) {
    buf_type_idx_ = torch::empty({ntotal}, torch::TensorOptions().dtype(torch::kInt64).device(dev));
    buf_pos_ = torch::empty({ntotal, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    cached_ntotal_ = ntotal;
  }

  Kokkos::View<int64_t *, DeviceType, Unmanaged> edge_src_v(buf_edge_src_.data_ptr<int64_t>(), Etotal);
  Kokkos::View<int64_t *, DeviceType, Unmanaged> edge_dst_v(buf_edge_dst_.data_ptr<int64_t>(), Etotal);
  Kokkos::View<int64_t *, DeviceType, Unmanaged> type_idx_v(buf_type_idx_.data_ptr<int64_t>(), ntotal);
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> edge_shifts_v(buf_edge_shifts_.data_ptr<float>(), Etotal, 3);
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> pos_v(buf_pos_.data_ptr<float>(), ntotal, 3);

  const float i00 = geom.inv[0][0], i01 = geom.inv[0][1], i02 = geom.inv[0][2];
  const float i10 = geom.inv[1][0], i11 = geom.inv[1][1], i12 = geom.inv[1][2];
  const float i20 = geom.inv[2][0], i21 = geom.inv[2][1], i22 = geom.inv[2][2];
  const int px = geom.pbc[0], py = geom.pbc[1], pz = geom.pbc[2];
  const float c00 = geom.cell[0][0], c01 = geom.cell[0][1], c02 = geom.cell[0][2];
  const float c10 = geom.cell[1][0], c11 = geom.cell[1][1], c12 = geom.cell[1][2];
  const float c20 = geom.cell[2][0], c21 = geom.cell[2][1], c22 = geom.cell[2][2];
  const float cutsq = static_cast<float>(cutsq_global_);

  Kokkos::parallel_for(
      "mfftorch::fill_pos_and_type", ntotal,
      KOKKOS_LAMBDA(const int i) {
        pos_v(i, 0) = static_cast<float>(x(i, 0));
        pos_v(i, 1) = static_cast<float>(x(i, 1));
        pos_v(i, 2) = static_cast<float>(x(i, 2));
        type_idx_v(i) = static_cast<int64_t>(type(i));
      });

  Kokkos::parallel_for(
      "mfftorch::fill_edges", inum, KOKKOS_LAMBDA(const int ii) {
        const int i = d_ilist[ii];
        const int jnum = d_numneigh[i];
        const int64_t base = d_offsets(ii);
        for (int jj = 0; jj < jnum; jj++) {
          int j = d_neighbors(i, jj) & NEIGHMASK;
          const int64_t idx = base + jj;
          const float rawx = static_cast<float>(x(j, 0) - x(i, 0));
          const float rawy = static_cast<float>(x(j, 1) - x(i, 1));
          const float rawz = static_cast<float>(x(j, 2) - x(i, 2));
          const float fracx = rawx * i00 + rawy * i10 + rawz * i20;
          const float fracy = rawx * i01 + rawy * i11 + rawz * i21;
          const float fracz = rawx * i02 + rawy * i12 + rawz * i22;
          const int sx = px ? -nearest_int_device(fracx) : 0;
          const int sy = py ? -nearest_int_device(fracy) : 0;
          const int sz = pz ? -nearest_int_device(fracz) : 0;
          const float shiftx = sx * c00 + sy * c10 + sz * c20;
          const float shifty = sx * c01 + sy * c11 + sz * c21;
          const float shiftz = sx * c02 + sy * c12 + sz * c22;
          const float delx = rawx + shiftx;
          const float dely = rawy + shifty;
          const float delz = rawz + shiftz;
          const float rsq = delx * delx + dely * dely + delz * delz;
          if (rsq > cutsq) {
            edge_src_v(idx) = static_cast<int64_t>(-1);
            edge_dst_v(idx) = static_cast<int64_t>(-1);
            edge_shifts_v(idx, 0) = 0.0f;
            edge_shifts_v(idx, 1) = 0.0f;
            edge_shifts_v(idx, 2) = 0.0f;
            continue;
          }
          edge_src_v(idx) = static_cast<int64_t>(i);
          edge_dst_v(idx) = static_cast<int64_t>(j);
          edge_shifts_v(idx, 0) = static_cast<float>(sx);
          edge_shifts_v(idx, 1) = static_cast<float>(sy);
          edge_shifts_v(idx, 2) = static_cast<float>(sz);
        }
      });
  finish_segment(fill_ms);

  auto valid_mask = buf_edge_src_.ge(0);
  const int64_t Efiltered = valid_mask.sum().item<int64_t>();
  if (Efiltered <= 1) return;
  if (Efiltered != Etotal) {
    auto valid_idx = torch::nonzero(valid_mask).view({-1});
    buf_edge_src_ = buf_edge_src_.index_select(0, valid_idx);
    buf_edge_dst_ = buf_edge_dst_.index_select(0, valid_idx);
    buf_edge_shifts_ = buf_edge_shifts_.index_select(0, valid_idx);
    cached_Etotal_ = Efiltered;
    Etotal = Efiltered;
  }

  const bool need_energy = static_cast<bool>(eflag_global || eflag_atom);
  const bool need_atom_virial = static_cast<bool>(vflag_atom);
  mfftorch::MFFOutputs out;

  const std::array<float, 9> cell_values = {
      geom.cell[0][0], geom.cell[0][1], geom.cell[0][2],
      geom.cell[1][0], geom.cell[1][1], geom.cell[1][2],
      geom.cell[2][0], geom.cell[2][1], geom.cell[2][2],
  };
  if (!cached_cell_valid_ || cached_cell_values_ != cell_values || !cached_cell_t_.defined() ||
      cached_cell_t_.device() != dev) {
    cached_cell_values_ = cell_values;
    cached_cell_t_ = torch::from_blob(
                         cached_cell_values_.data(),
                         {1, 3, 3},
                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                         .clone()
                         .to(dev);
    cached_cell_valid_ = true;
  }
  auto cell_t = cached_cell_t_;

  try {
    Kokkos::fence();
    if (engine_->is_bundle_manifest() &&
        (prepared_nlocal_ != nlocal || prepared_ntotal_ != ntotal || prepared_nedges_ != Etotal)) {
      engine_->prepare_for_shape(nlocal, ntotal, Etotal);
      prepared_nlocal_ = nlocal;
      prepared_ntotal_ = ntotal;
      prepared_nedges_ = Etotal;
    }
    finish_segment(prepare_ms);
    if (engine_->prefers_kokkos_host_staging()) {
      // Native cue custom CUDA ops are unstable when fed Kokkos-managed device
      // tensors and an external CUDA stream. Stage inputs through CPU so the
      // engine follows the same stable transfer path as plain mff/torch.
      auto pos_cpu = buf_pos_.to(torch::kCPU, torch::kFloat32).contiguous();
      auto A_cpu = type2Z_cuda_.index_select(0, buf_type_idx_).to(torch::kCPU, torch::kInt64).contiguous();
      auto edge_src_cpu = buf_edge_src_.to(torch::kCPU, torch::kInt64).contiguous();
      auto edge_dst_cpu = buf_edge_dst_.to(torch::kCPU, torch::kInt64).contiguous();
      auto edge_shifts_cpu = buf_edge_shifts_.to(torch::kCPU, torch::kFloat32).contiguous();
      auto external_tensor = current_external_tensor(torch::kCPU);
      auto fidelity_ids = current_fidelity_tensor(torch::kCPU);
      out = engine_->compute(
          nlocal,
          ntotal,
          pos_cpu,
          A_cpu,
          edge_src_cpu,
          edge_dst_cpu,
          edge_shifts_cpu,
          cell_t.to(torch::kCPU, torch::kFloat32).contiguous(),
          external_tensor,
          fidelity_ids,
          need_energy,
          need_atom_virial);
      if (engine_->is_cuda()) torch::cuda::synchronize();
    } else {
      auto kk_stream = Kokkos::Cuda().cuda_stream();
      auto torch_stream = c10::cuda::getStreamFromExternal(kk_stream, dev.index());
      c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

      auto A = type2Z_cuda_.index_select(0, buf_type_idx_);
      auto external_tensor = current_external_tensor(dev);
      auto fidelity_ids = current_fidelity_tensor(dev);
      out = engine_->compute(nlocal, ntotal, buf_pos_, A, buf_edge_src_, buf_edge_dst_, buf_edge_shifts_, cell_t, external_tensor,
                             fidelity_ids,
                             need_energy, need_atom_virial);
    }
    finish_segment(compute_ms);
  } catch (const std::exception &e) {
    error->all(FLERR, (std::string("mff/torch/kk engine compute failed: ") + e.what()).c_str());
  }
  cache_physical_outputs(out, nlocal);

  mfftorch::ReciprocalOutputs reciprocal_out;
  torch::Tensor reciprocal_forces_dev;
  const bool exports_runtime_source = engine_->exports_reciprocal_source() && engine_->reciprocal_source_channels() > 0;
  const bool use_tree_fmm =
      tree_fmm_solver_ && exports_runtime_source && engine_->long_range_runtime_backend() == "tree_fmm";
  const bool use_reciprocal =
      reciprocal_solver_ && exports_runtime_source && !use_tree_fmm;
  const bool use_runtime_long_range = use_tree_fmm || use_reciprocal;
  if (use_tree_fmm || use_reciprocal) {
    try {
      auto local_source = out.reciprocal_source.defined()
                              ? out.reciprocal_source.narrow(0, 0, nlocal).contiguous()
                              : torch::zeros(
                                    {nlocal, engine_->reciprocal_source_channels()},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(dev));
      if (use_tree_fmm && engine_->long_range_source_kind() != "latent_charge") {
        throw std::runtime_error("tree_fmm runtime currently requires long_range_source_kind=latent_charge");
      }
      auto reciprocal_inputs = make_reciprocal_inputs(
          world,
          buf_pos_.narrow(0, 0, nlocal).contiguous(),
          local_source,
          cell_t.contiguous(),
          geom,
          need_energy,
          dev);
      reciprocal_out = use_tree_fmm ? tree_fmm_solver_->compute(reciprocal_inputs)
                                    : reciprocal_solver_->compute(reciprocal_inputs);
    } catch (const std::exception &e) {
      error->all(FLERR, (std::string("mff/torch/kk runtime long-range solver failed: ") + e.what()).c_str());
    }
  }
  finish_segment(reciprocal_ms);

  if (eflag_global) eng_vdwl += out.energy;
  if (eflag_global && use_runtime_long_range) eng_vdwl += reciprocal_out.energy;

  // Write forces on device (no host transfer).
  auto forces = out.forces.contiguous();
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> forces_v(forces.data_ptr<float>(), ntotal, 3);

  const int nwrite = force->newton_pair ? ntotal : nlocal;
  Kokkos::parallel_for(
      "mfftorch::add_forces", nwrite, KOKKOS_LAMBDA(const int i) {
        f(i, 0) += forces_v(i, 0);
        f(i, 1) += forces_v(i, 1);
        f(i, 2) += forces_v(i, 2);
      });
  if (use_runtime_long_range && reciprocal_out.forces_local.defined()) {
    reciprocal_forces_dev = reciprocal_out.forces_local.to(dev, torch::kFloat32).contiguous();
    Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> reciprocal_forces_v(
        reciprocal_forces_dev.data_ptr<float>(), nlocal, 3);
    Kokkos::parallel_for(
        "mfftorch::add_reciprocal_forces", nlocal, KOKKOS_LAMBDA(const int i) {
          f(i, 0) += reciprocal_forces_v(i, 0);
          f(i, 1) += reciprocal_forces_v(i, 1);
          f(i, 2) += reciprocal_forces_v(i, 2);
        });
  }
  finish_segment(force_ms);

  // Per-atom energy: copy NN atom_energy to LAMMPS eatom (local atoms only).
  if (eflag_atom && eatom && out.atom_energy.defined()) {
    auto ae = out.atom_energy.to(torch::kCPU, torch::kFloat64).contiguous().view({ntotal});
    const double *ep = ae.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }
  if (eflag_atom && eatom && use_runtime_long_range && reciprocal_out.atom_energy_local.defined()) {
    auto ae_recip = reciprocal_out.atom_energy_local.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *ep = ae_recip.data_ptr<double>();
    for (int i = 0; i < nlocal; i++) eatom[i] += ep[i];
  }
  finish_segment(atom_output_ms);

  // Per-atom virial: engine computed atom_virial [ntotal, 6] on GPU via edge-force
  // outer products (rij ⊗ edge_forces), scatter-added 50/50 to src and dst.
  // With newton OFF + FULL list, only write LOCAL atoms (consistent with ev_tally
  // convention — each pair is visited twice, so each local atom gets its full share).
  if (vflag_atom && vatom && out.atom_virial.defined()) {
    auto avir = out.atom_virial.to(torch::kCPU, torch::kFloat64).contiguous();
    const double *vp = avir.data_ptr<double>();
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
  finish_segment(virial_ms);

#ifdef MFF_ENABLE_VIRIAL
  if (vflag_global) {
    // f·r virial must sum over ALL atoms (local + ghost).
    // Ghost atoms carry the "other half" of cross-boundary PBC interactions.
    // Using forces_v (NN output) directly guarantees ghost forces are included.
    Kokkos::View<double[6], DeviceType> d_virial("mfftorch::d_virial");
    Kokkos::deep_copy(d_virial, 0.0);

    Kokkos::parallel_for(
        "mfftorch::virial_fdotr", ntotal, KOKKOS_LAMBDA(const int i) {
          const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
          const double fx = static_cast<double>(forces_v(i, 0));
          const double fy = static_cast<double>(forces_v(i, 1));
          const double fz = static_cast<double>(forces_v(i, 2));
          Kokkos::atomic_add(&d_virial(0), fx * xi);  // xx
          Kokkos::atomic_add(&d_virial(1), fy * yi);  // yy
          Kokkos::atomic_add(&d_virial(2), fz * zi);  // zz
          Kokkos::atomic_add(&d_virial(3), fy * xi);  // xy
          Kokkos::atomic_add(&d_virial(4), fz * xi);  // xz
          Kokkos::atomic_add(&d_virial(5), fz * yi);  // yz
        });
    Kokkos::fence();

    auto h_virial = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_virial);
    for (int n = 0; n < 6; n++) virial[n] += h_virial(n);
    if (use_runtime_long_range && reciprocal_forces_dev.defined()) {
      Kokkos::View<double[6], DeviceType> d_recip_virial("mfftorch::d_recip_virial");
      Kokkos::deep_copy(d_recip_virial, 0.0);
      Kokkos::View<float **, Kokkos::LayoutRight, DeviceType, Unmanaged> reciprocal_forces_v(
          reciprocal_forces_dev.data_ptr<float>(), nlocal, 3);
      Kokkos::parallel_for(
          "mfftorch::reciprocal_virial_fdotr", nlocal, KOKKOS_LAMBDA(const int i) {
            const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
            const double fx = static_cast<double>(reciprocal_forces_v(i, 0));
            const double fy = static_cast<double>(reciprocal_forces_v(i, 1));
            const double fz = static_cast<double>(reciprocal_forces_v(i, 2));
            Kokkos::atomic_add(&d_recip_virial(0), fx * xi);
            Kokkos::atomic_add(&d_recip_virial(1), fy * yi);
            Kokkos::atomic_add(&d_recip_virial(2), fz * zi);
            Kokkos::atomic_add(&d_recip_virial(3), fy * xi);
            Kokkos::atomic_add(&d_recip_virial(4), fz * xi);
            Kokkos::atomic_add(&d_recip_virial(5), fz * yi);
          });
      Kokkos::fence();
      auto h_recip_virial = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_recip_virial);
      for (int n = 0; n < 6; n++) virial[n] += h_recip_virial(n);
    }
  }
#endif
  if (debug_kk_timings) {
    Kokkos::fence();
    const auto total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total_start).count();
    std::fprintf(
        stderr,
        "[MFF_DEBUG_KK_TIMINGS] nlocal=%d ntotal=%d nedges=%lld edge_count_ms=%.3f offset_scan_ms=%.3f fill_ms=%.3f prepare_ms=%.3f compute_ms=%.3f reciprocal_ms=%.3f force_ms=%.3f atom_output_ms=%.3f virial_ms=%.3f total_ms=%.3f\n",
        nlocal,
        ntotal,
        static_cast<long long>(Etotal),
        edge_count_ms,
        offset_scan_ms,
        fill_ms,
        prepare_ms,
        compute_ms,
        reciprocal_ms,
        force_ms,
        atom_output_ms,
        virial_ms,
        total_ms);
  }
}

namespace LAMMPS_NS {
template class PairMFFTorchKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMFFTorchKokkos<LMPHostType>;
#endif
}  // namespace LAMMPS_NS

#endif  // LMP_KOKKOS
