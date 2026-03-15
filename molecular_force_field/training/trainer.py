"""Training module for molecular force field models."""

import os
import time
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from molecular_force_field.utils.scatter import scatter
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, StepLR
import copy

from molecular_force_field.models import E3_TransformerLayer_multi, MainNet
from molecular_force_field.models.losses import RMSELoss
from molecular_force_field.utils.tensor_utils import build_physical_tensor_label_blocks, map_tensor_values
from molecular_force_field.utils.config import ModelConfig


def log_gradient_statistics(networks, batch_idx, logger):
    """Log gradient statistics for all networks."""
    for net in networks:
        for name, param in net.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_norm = torch.norm(grad).item()
                grad_max = grad.max().item()
                grad_min = grad.min().item()
                grad_mean = grad.mean().item()
                
                logger.debug(
                    f"Gradient/Batch_{batch_idx}/Network_{net.__class__.__name__}/Layer_{name}:\n"
                    f"  Norm: {grad_norm:.6e}\n"
                    f"  Max: {grad_max:.6e}\n"
                    f"  Min: {grad_min:.6e}\n"
                    f"  Mean: {grad_mean:.6e}\n"
                )
            else:
                logger.debug(
                    f"Gradient/Batch_{batch_idx}/Network_{net.__class__.__name__}/Layer_{name}:\n"
                    "  No gradients"
                )


class Trainer:
    """Trainer class for molecular force field models."""
    
    def __init__(
        self,
        model,
        e3trans,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        device,
        config=None,
        learning_rate=2e-4,
        min_learning_rate=1e-5,
        initial_learning_rate_for_weight=0.1,
        epoch_numbers=1000,
        patience=20,
        dump_frequency=250,
        energy_log_frequency=10,
        vhat_clamp_interval=2000,
        warmup_batches=1000,
        patience_opim=1000,
        gamma_value=0.98,
        max_vhat_growth_factor=5,
        max_norm_value=0.5,
        gradient_log_interval=100,
        update_param=1000,
        force_shift_value=1,
        a=1,
        b=10,
        weight_a_growth=1.05,
        weight_b_decay=0.98,
        a_min=1.0,
        a_max=1000.0,
        b_min=1.0,
        b_max=1000.0,
        swa_start_epoch=None,
        swa_a=None,
        swa_b=None,
        ema_start_epoch=None,
        ema_decay=0.999,
        use_ema_for_validation=False,
        save_ema_model=False,
        checkpoint_path='combined_model.pth',
        atomic_energy_keys=None,
        atomic_energy_values=None,
        distributed=False,
        rank=0,
        world_size=1,
        train_sampler=None,
        val_sampler=None,
        save_val_csv=False,
        use_checkpoint_loss_weights=True,
        train_eval_sample_ratio=0.2,
        log_val_batch_energy_to_console=False,
        tensor_product_mode='spherical',
        c=0.0,
        c_min=0.0,
        c_max=1000.0,
        # Validation-only torch.compile (safe for create_graph=False in validate)
        compile_val: str = "none",
        compile_val_mode: str = "reduce-overhead",
        compile_val_fullgraph: bool = False,
        compile_val_dynamic: bool = False,
        compile_val_precache: bool = False,
        # 推理模式：保存到 checkpoint，evaluate/inference_ddp 可读取；TorchScript/LAMMPS 导出始终只输出能量和力
        inference_output_physical_tensors: bool = False,
        # 物理张量 loss 系数：{"charge": 1.0, "dipole": 2.0, ...}，未指定则 1.0；charge_per_atom 用 charge
        physical_tensor_weights: dict[str, float] | None = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: MainNet model
            e3trans: E3_TransformerLayer_multi model
            train_loader: Training data loader
            val_loader: Validation data loader
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: Device to use
            config: ModelConfig object (optional)
            learning_rate: Initial learning rate
            min_learning_rate: Minimum learning rate
            initial_learning_rate_for_weight: Initial learning rate ratio for weight
            epoch_numbers: Number of epochs
            patience: Early stopping patience
            dump_frequency: Frequency to dump validation results
            vhat_clamp_interval: Interval to clamp v_hat
            warmup_batches: Number of warmup batches
            patience_opim: Patience for optimizer
            gamma_value: Learning rate decay factor
            max_vhat_growth_factor: Maximum v_hat growth factor
            max_norm_value: Maximum gradient norm
            gradient_log_interval: Interval to log gradients
            update_param: Interval to update loss weights
            force_shift_value: Force shift value
            a: Energy loss weight
            b: Force loss weight
            swa_start_epoch: Epoch to start SWA (Stochastic Weight Averaging) for loss weights.
                After this epoch, a and b will be set to swa_a and swa_b directly.
            swa_a: Energy weight a after SWA starts
            swa_b: Force weight b after SWA starts
            ema_start_epoch: Epoch to start EMA (Exponential Moving Average) for e3trans weights
            ema_decay: EMA decay factor in (0, 1) (default: 0.999)
            use_ema_for_validation: Whether to use EMA weights for validation forward
            save_ema_model: Whether to save EMA weights into checkpoints
            checkpoint_path: Path to save checkpoints
            atomic_energy_keys: Atomic energy keys tensor
            atomic_energy_values: Atomic energy values tensor
            distributed: Whether to use distributed training
            rank: Process rank in distributed training
            world_size: Total number of processes
            train_sampler: Distributed sampler for training data
            save_val_csv: Whether to save validation energy and force predictions to CSV files
                (default: True). If False, only metrics are logged but CSV files are not saved.
            use_checkpoint_loss_weights: Whether to use loss weights (a, b) from checkpoint when loading.
                If True (default), use checkpoint values. If False, use the values passed to __init__.
            tensor_product_mode: Tensor product mode used (default: 'spherical').
                Options: 'spherical', 'spherical-save', 'spherical-save-cue',
                'partial-cartesian', 'partial-cartesian-loose',
                'pure-cartesian', 'pure-cartesian-sparse', 'pure-cartesian-sparse-save',
                'pure-cartesian-ictd', 'pure-cartesian-ictd-save'
        """
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.is_main_process = (rank == 0)
        self.tensor_product_mode = tensor_product_mode
        self.inference_output_physical_tensors = inference_output_physical_tensors
        self.physical_tensor_weights = physical_tensor_weights or {}

        # Validation-only compile settings
        self.compile_val = str(compile_val)
        self.compile_val_mode = str(compile_val_mode)
        self.compile_val_fullgraph = bool(compile_val_fullgraph)
        self.compile_val_dynamic = bool(compile_val_dynamic)
        self.compile_val_precache = bool(compile_val_precache)
        self._val_compiled_current = None
        self._val_compiled_ema = None
        self._val_compile_failed_current = False
        self._val_compile_failed_ema = False
        
        # Convert tensor_product_mode to suffix format (e.g., 'pure-cartesian' -> '_pure_cartesian')
        mode_suffix_map = {
            'spherical': '_spherical',
            'spherical-save': '_spherical_save',
            'spherical-save-cue': '_spherical_save_cue',
            'partial-cartesian': '_partial_cartesian',
            'partial-cartesian-loose': '_partial_cartesian_loose',
            'pure-cartesian': '_pure_cartesian',
            'pure-cartesian-sparse': '_pure_cartesian_sparse',
            'pure-cartesian-sparse-save': '_pure_cartesian_sparse_save',
            'pure-cartesian-ictd': '_pure_cartesian_ictd',
            'pure-cartesian-ictd-save': '_pure_cartesian_ictd_save',
        }
        self.tensor_product_suffix = mode_suffix_map.get(tensor_product_mode, '_spherical')
        
        self.model = model
        self.e3trans = e3trans
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.config = config
        self.e3_arch_metadata = self._collect_e3_arch_metadata()
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.initial_learning_rate_for_weight = initial_learning_rate_for_weight
        self.epoch_numbers = epoch_numbers
        self.patience = patience
        self.dump_frequency = dump_frequency
        self.energy_log_frequency = energy_log_frequency
        self.vhat_clamp_interval = vhat_clamp_interval
        self.warmup_batches = warmup_batches
        self.patience_opim = patience_opim
        self.gamma_value = gamma_value
        self.max_vhat_growth_factor = max_vhat_growth_factor
        self.max_norm_value = max_norm_value
        self.gradient_log_interval = gradient_log_interval
        self.update_param = update_param
        self.force_shift_value = force_shift_value
        self.a = a
        self.b = b
        self.weight_a_growth = weight_a_growth
        self.weight_b_decay = weight_b_decay
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.c = c
        self.c_min = c_min
        self.c_max = c_max
        self.swa_start_epoch = swa_start_epoch
        self.swa_a = swa_a
        self.swa_b = swa_b
        self.swa_applied = False  # Track if SWA has been applied
        self.ema_start_epoch = ema_start_epoch
        self.ema_decay = ema_decay
        self.use_ema_for_validation = use_ema_for_validation
        self.save_ema_model = save_ema_model
        self.ema_enabled = False
        self.e3trans_ema = None  # will be initialized when EMA starts
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = 'checkpoint'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create validation results directory if save_val_csv is enabled
        self.save_val_csv = save_val_csv  # Control whether to save val_energy.csv and val_force.csv
        if self.save_val_csv:
            self.val_results_dir = 'validation'
            os.makedirs(self.val_results_dir, exist_ok=True)
        else:
            self.val_results_dir = None
        
        # Store original checkpoint path for loading, but save new checkpoints to checkpoint directory
        self.checkpoint_path = checkpoint_path
        # If checkpoint_path is a filename (not a full path), save it in checkpoint directory
        if os.path.dirname(checkpoint_path) == '':
            base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            self.checkpoint_save_path = os.path.join(self.checkpoint_dir, f'{base_name}{self.tensor_product_suffix}.pth')
        else:
            # If it's a full path, extract filename and save to checkpoint directory
            base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            self.checkpoint_save_path = os.path.join(self.checkpoint_dir, f'{base_name}{self.tensor_product_suffix}.pth')
        self.use_checkpoint_loss_weights = use_checkpoint_loss_weights  # Whether to use checkpoint a/b
        self.train_eval_sample_ratio = train_eval_sample_ratio  # Ratio of training set to evaluate (0.0-1.0)
        self.log_val_batch_energy_to_console = log_val_batch_energy_to_console  # Whether to log validation batch energy to console (default: False, only to file)
        
        # Initialize loss tracking for CSV (under checkpoint_dir so resume appends to same file)
        self.loss_csv_path = os.path.join(self.checkpoint_dir, 'loss.csv')
        self.loss_records = []
        
        # Initialize training state (will be updated if loading from checkpoint)
        self.start_epoch = 1
        self.batch_count = 0
        # Best validation metric used for early stopping / best checkpoint.
        # NOTE: This tracks a composite metric:
        #   metric = val_energy_rmse + (b / a) * val_force_rmse
        # where val_*_rmse are restored to original units.
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.resumed_from_checkpoint = False
        self.best_checkpoint_path = None  # Path to the best checkpoint file

        # Clamp initial weights (and any user-provided weights) to bounds
        self._clamp_loss_weights()
        
        if atomic_energy_keys is None:
            self.keys = torch.tensor([1, 6, 7, 8]).to(device)
        else:
            self.keys = atomic_energy_keys.to(device)
            
        if atomic_energy_values is None:
            self.values = torch.tensor([
                -430.53299511, -821.03326787, -1488.18856918, -2044.3509823
            ]).to(device)
        else:
            self.values = atomic_energy_values.to(device)
        
        # Loss functions
        self.criterion = nn.SmoothL1Loss(beta=0.5)
        self.criterion_2 = RMSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(e3trans.parameters()) + list(model.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
            amsgrad=True
        )
        
        # Learning rate scheduler
        milestones = [warmup_batches]
        
        def warmup_lambda(current_step):
            progress = min(current_step / warmup_batches, 1.0)
            initial_ratio = initial_learning_rate_for_weight
            return initial_ratio + (1 - initial_ratio) * progress
        
        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        step_scheduler = StepLR(
            self.optimizer,
            step_size=patience_opim,
            gamma=gamma_value
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=milestones
        )
        
        # Wrap models with DDP if distributed training is enabled
        if self.distributed:
            # Enable find_unused_parameters for models with conditional computation paths
            # (e.g., MainNet is not used in e3trans forward pass)
            self.e3trans = DDP(
                self.e3trans,
                device_ids=[device.index] if device.type == 'cuda' else None,
                output_device=device.index if device.type == 'cuda' else None,
                find_unused_parameters=True,  # Required: some parameters don't participate in every forward pass
                broadcast_buffers=True
            )
            # Note: model (MainNet) is not used in forward pass, so we don't wrap it
            # But if it's used elsewhere, wrap it too:
            # self.model = DDP(self.model, ...)
        
        # These will be overwritten if loading from checkpoint
        # Kept here for clarity of initialization sequence
        self.loss_out = []
        
        # Log training configuration (only on main process)
        if self.is_main_process:
            logging.info("=" * 60)
            logging.info("Training Configuration:")
            logging.info(f"  Epochs: {self.epoch_numbers}")
            logging.info(f"  Learning Rate: {self.learning_rate}")
            logging.info(f"  Min Learning Rate: {self.min_learning_rate}")
            logging.info(f"  Training Samples: {len(self.train_dataset)}")
            logging.info(f"  Validation Samples: {len(self.val_dataset)}")
            logging.info(f"  Energy Loss Weight (a): {self.a}")
            logging.info(f"  Force Loss Weight (b): {self.b}")
            logging.info(f"  Stress Loss Weight (c): {self.c}")
            if self.swa_start_epoch is not None:
                logging.info(f"  SWA enabled: Will switch to a={self.swa_a}, b={self.swa_b} at epoch {self.swa_start_epoch}")
            if self.ema_start_epoch is not None:
                logging.info(f"  EMA enabled: Will start at epoch {self.ema_start_epoch} with decay={self.ema_decay}")
                if self.use_ema_for_validation:
                    logging.info("    Using EMA model for validation")
                if self.save_ema_model:
                    logging.info("    Will save EMA model in checkpoint")
            logging.info(f"  Validation Frequency: every {self.dump_frequency} batches")
            if self.train_eval_sample_ratio < 1.0:
                logging.info(f"  Train Evaluation: {self.train_eval_sample_ratio*100:.1f}% sampling (faster validation)")
            else:
                logging.info(f"  Train Evaluation: Full dataset (100%)")
            logging.info(f"  Energy Log Frequency: every {self.energy_log_frequency} batches")
            logging.info(f"  Early Stopping Patience: {self.patience} epochs")
            if self.distributed:
                logging.info(f"  Distributed Training: {self.world_size} GPUs")
            logging.info("=" * 60)
        
        # Load checkpoint if exists
        # Try original path first, then try checkpoint directory
        checkpoint_to_load = None
        if os.path.exists(checkpoint_path):
            checkpoint_to_load = checkpoint_path
        else:
            # Try in checkpoint directory
            checkpoint_in_dir = os.path.join(self.checkpoint_dir, os.path.basename(checkpoint_path))
            if os.path.exists(checkpoint_in_dir):
                checkpoint_to_load = checkpoint_in_dir
        
        if checkpoint_to_load:
            self.load_checkpoint(checkpoint_to_load)

    def _cudagraph_step_begin_if_available(self):
        try:
            torch.compiler.cudagraph_mark_step_begin()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _maybe_compile_validation_model(self, eval_model: nn.Module, batch_tensors, *, is_ema: bool) -> nn.Module:
        """
        Optionally compile the validation forward (create_graph=False path only).
        We compile only the selected eval_model (current or EMA) and cache it.
        """
        if self.compile_val == "none":
            return eval_model
        if self.compile_val != "e3trans":
            return eval_model
        if self.device.type != "cuda":
            return eval_model
        if not hasattr(torch, "compile"):
            return eval_model

        if is_ema:
            if self._val_compiled_ema is not None:
                return self._val_compiled_ema
            if self._val_compile_failed_ema:
                return eval_model
        else:
            if self._val_compiled_current is not None:
                return self._val_compiled_current
            if self._val_compile_failed_current:
                return eval_model

        # Precache to avoid CPU-side cache build during tracing (ICTD direction_harmonics_fast)
        if self.compile_val_precache:
            try:
                (pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell) = batch_tensors
                with torch.no_grad():
                    self._cudagraph_step_begin_if_available()
                    _ = eval_model(pos.detach(), A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
            except Exception as e:
                if self.is_main_process:
                    logging.warning(f"Validation compile precache failed; continuing. Error: {e}")

        # NOTE: torch.compile(mode="reduce-overhead") may enable CUDA Graphs. With custom CUDA
        # extensions (e.g., cuEquivariance ops) this can occasionally segfault. Validation
        # correctness does not require CUDA Graphs, so we disable cudagraph capture here.
        # (Inductor still provides fusion / lowered kernels without cudagraphs.)
        old_inductor_cfg = {}
        try:
            import torch._inductor.config as inductor_config  # type: ignore

            for key in ("cudagraphs", "triton.cudagraphs"):
                # Some versions expose nested configs; we handle both patterns.
                if key == "cudagraphs" and hasattr(inductor_config, "cudagraphs"):
                    old_inductor_cfg[key] = getattr(inductor_config, "cudagraphs")
                    setattr(inductor_config, "cudagraphs", False)
                elif key == "triton.cudagraphs" and hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                    old_inductor_cfg[key] = getattr(inductor_config.triton, "cudagraphs")
                    setattr(inductor_config.triton, "cudagraphs", False)
        except Exception:
            old_inductor_cfg = {}

        try:
            if self.is_main_process:
                logging.info(
                    "Compiling validation e3trans (mode=%s, fullgraph=%s, dynamic=%s)",
                    self.compile_val_mode,
                    self.compile_val_fullgraph,
                    self.compile_val_dynamic,
                )
            compiled = torch.compile(
                eval_model,
                mode=self.compile_val_mode,
                fullgraph=self.compile_val_fullgraph,
                dynamic=self.compile_val_dynamic,
            )
            if is_ema:
                self._val_compiled_ema = compiled
            else:
                self._val_compiled_current = compiled
            return compiled
        except Exception as e:
            if self.is_main_process:
                logging.warning(f"Validation torch.compile failed; continuing without compile. Error: {e}")
            if is_ema:
                self._val_compile_failed_ema = True
            else:
                self._val_compile_failed_current = True
            return eval_model
        finally:
            # Restore inductor config.
            try:
                import torch._inductor.config as inductor_config  # type: ignore

                if "cudagraphs" in old_inductor_cfg and hasattr(inductor_config, "cudagraphs"):
                    setattr(inductor_config, "cudagraphs", old_inductor_cfg["cudagraphs"])
                if "triton.cudagraphs" in old_inductor_cfg and hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                    setattr(inductor_config.triton, "cudagraphs", old_inductor_cfg["triton.cudagraphs"])
            except Exception:
                pass
    
    def _clamp_loss_weights(self):
        """Clamp loss weights a/b/c to configured bounds."""
        if self.a_min is not None:
            self.a = max(self.a, self.a_min)
        if self.a_max is not None:
            self.a = min(self.a, self.a_max)
        if self.b_min is not None:
            self.b = max(self.b, self.b_min)
        if self.b_max is not None:
            self.b = min(self.b, self.b_max)
        if self.c_min is not None:
            self.c = max(self.c, self.c_min)
        if self.c_max is not None:
            self.c = min(self.c, self.c_max)

    def _collect_e3_arch_metadata(self):
        """
        Collect minimal architecture metadata required to rebuild optional heads at inference.
        Kept best-effort for backward compatibility.
        """
        base = self.e3trans.module if (self.distributed and hasattr(self.e3trans, "module")) else self.e3trans
        meta = {}

        cfg = self.config
        if cfg is not None:
            model_hparams = {
                "dtype": str(getattr(cfg, "dtype", torch.float64)).replace("torch.", ""),
                "channel_in": int(getattr(cfg, "channel_in", 64)),
                "channel_in2": int(getattr(cfg, "channel_in2", 32)),
                "channel_in3": int(getattr(cfg, "channel_in3", 32)),
                "channel_in4": int(getattr(cfg, "channel_in4", 32)),
                "channel_in5": int(getattr(cfg, "channel_in5", 32)),
                "max_atomvalue": int(getattr(cfg, "max_atomvalue", 10)),
                "embedding_dim": int(getattr(cfg, "embedding_dim", 16)),
                "embed_size": list(getattr(cfg, "embed_size", [128, 128, 128])),
                "output_size": int(getattr(cfg, "output_size", 8)),
                "lmax": int(getattr(cfg, "lmax", 2)),
                "irreps_output_conv_channels": getattr(cfg, "irreps_output_conv_channels", None),
                "function_type": str(getattr(cfg, "function_type", "gaussian")),
                "max_radius": float(getattr(cfg, "max_radius", 5.0)),
                "max_radius_main": float(getattr(cfg, "max_radius_main", getattr(cfg, "max_radius", 5.0))),
                "number_of_basis": int(getattr(cfg, "number_of_basis", 8)),
                "number_of_basis_main": int(getattr(cfg, "number_of_basis_main", 8)),
                "num_layers": int(getattr(cfg, "num_layers", 1)),
                "main_hidden_sizes3": list(getattr(cfg, "main_hidden_sizes3", [64, 32])),
                "emb_number_main_2": list(getattr(cfg, "emb_number_main_2", [64, 64, 64])),
            }
            num_interaction = getattr(base, "num_interaction", None)
            if num_interaction is not None:
                model_hparams["num_interaction"] = int(num_interaction)
            invariant_channels = getattr(base, "invariant_channels", None)
            if invariant_channels is not None:
                model_hparams["invariant_channels"] = int(invariant_channels)
            for attr_name in (
                "max_rank_other",
                "k_policy",
                "ictd_tp_path_policy",
                "ictd_tp_max_rank_other",
                "long_range_mode",
                "long_range_hidden_dim",
                "long_range_boundary",
                "long_range_neutralize",
                "long_range_filter_hidden_dim",
                "long_range_kmax",
                "long_range_mesh_size",
                "long_range_slab_padding_factor",
                "long_range_include_k0",
                "long_range_source_channels",
                "long_range_backend",
                "long_range_reciprocal_backend",
                "long_range_energy_partition",
                "long_range_green_mode",
                "long_range_assignment",
                "long_range_theta",
                "long_range_leaf_size",
                "long_range_multipole_order",
                "long_range_num_k",
                "feature_spectral_mode",
                "feature_spectral_bottleneck_dim",
                "feature_spectral_mesh_size",
                "feature_spectral_filter_hidden_dim",
                "feature_spectral_boundary",
                "feature_spectral_slab_padding_factor",
                "feature_spectral_neutralize",
                "feature_spectral_include_k0",
                "feature_spectral_gate_init",
                "zbl_enabled",
                "zbl_inner_cutoff",
                "zbl_outer_cutoff",
                "zbl_exponent",
                "zbl_energy_scale",
            ):
                if hasattr(base, attr_name):
                    value = getattr(base, attr_name)
                    if value is not None:
                        model_hparams[attr_name] = value
            meta["model_hyperparameters"] = model_hparams

        phys_specs = getattr(base, "_physical_tensor_specs", None)
        if phys_specs:
            meta["physical_tensor_outputs"] = copy.deepcopy(phys_specs)

        e3_conv_emb = getattr(base, "e3_conv_emb", None)
        ext_rank = getattr(e3_conv_emb, "external_tensor_rank", None) if e3_conv_emb is not None else None
        if ext_rank is not None:
            meta["external_tensor_rank"] = int(ext_rank)

        return meta

    def _compute_physical_tensor_loss(self, extras, phys_pred, batch_idx, base_e3):
        """Compute weighted physical tensor supervised loss from extras and predicted heads."""
        if phys_pred is None or not isinstance(phys_pred, dict):
            return torch.tensor(0.0, device=self.device)
        if not hasattr(self, "_phys_label_embedders"):
            self._phys_label_embedders = {}
        emb_cache = self._phys_label_embedders
        phys_weights = self.physical_tensor_weights
        phys_loss = torch.tensor(0.0, device=self.device)
        lmax_model = int(getattr(base_e3, "lmax", 2))
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0

        def _phys_weight(name: str) -> float:
            base = name.replace("_per_atom", "")
            return phys_weights.get(name, phys_weights.get(base, 1.0))

        def _get_embed(y_tensor: torch.Tensor, rank: int, *, include_trace_chain: bool):
            return build_physical_tensor_label_blocks(
                y_tensor,
                rank=rank,
                lmax=lmax_model,
                include_trace_chain=include_trace_chain,
                representation=getattr(base_e3, "physical_tensor_representation", "ictd"),
                device=self.device,
                cache=emb_cache,
            )

        # charge / charge_per_atom: scalar l=0
        for name in ("charge", "charge_per_atom"):
            if name in extras and name in phys_pred:
                w = _phys_weight(name)
                mask = extras.get(f"{name}_mask", None)
                mask = mask.to(dtype=torch.bool) if torch.is_tensor(mask) else None
                y = extras[name].view(-1)
                p0 = phys_pred[name].get(0) if isinstance(phys_pred[name], dict) else None
                if p0 is None:
                    continue
                if name.endswith("_per_atom"):
                    p = p0.view(y.shape[0], -1).mean(dim=1) if p0.shape[0] == y.shape[0] else None
                else:
                    p = p0.view(num_graphs, -1).mean(dim=1) if p0.shape[0] == num_graphs else None
                if p is None:
                    continue
                if mask is not None and mask.numel() == p.numel():
                    phys_loss = phys_loss + w * self.criterion(p[mask], y[mask])
                else:
                    phys_loss = phys_loss + w * self.criterion(p, y)

        # dipole / dipole_per_atom: rank1 -> l=1
        for name in ("dipole", "dipole_per_atom"):
            if name in extras and name in phys_pred:
                w = _phys_weight(name)
                mask = extras.get(f"{name}_mask", None)
                mask = mask.to(dtype=torch.bool) if torch.is_tensor(mask) else None
                y_cart = extras[name]
                y_blocks = _get_embed(y_cart, 1, include_trace_chain=False)
                p1 = phys_pred[name].get(1) if isinstance(phys_pred[name], dict) else None
                if p1 is None or 1 not in y_blocks:
                    continue
                if mask is not None and mask.shape[0] == p1.shape[0]:
                    phys_loss = phys_loss + w * self.criterion(p1[mask].view(-1), y_blocks[1][mask].view(-1))
                else:
                    phys_loss = phys_loss + w * self.criterion(p1.view(-1), y_blocks[1].view(-1))

        # polarizability / per_atom: rank2 -> l=0 + l=2
        for name in ("polarizability", "polarizability_per_atom"):
            if name in extras and name in phys_pred:
                w = _phys_weight(name)
                mask = extras.get(f"{name}_mask", None)
                mask = mask.to(dtype=torch.bool) if torch.is_tensor(mask) else None
                y_cart = extras[name]
                y_blocks = _get_embed(y_cart, 2, include_trace_chain=True)
                for l in (0, 2):
                    pl = phys_pred[name].get(l) if isinstance(phys_pred[name], dict) else None
                    if pl is None or l not in y_blocks:
                        continue
                    if mask is not None and mask.shape[0] == pl.shape[0]:
                        phys_loss = phys_loss + w * self.criterion(pl[mask].view(-1), y_blocks[l][mask].view(-1))
                    else:
                        phys_loss = phys_loss + w * self.criterion(pl.view(-1), y_blocks[l].view(-1))

        # quadrupole / per_atom: rank2 compare STF l=2
        for name in ("quadrupole", "quadrupole_per_atom"):
            if name in extras and name in phys_pred:
                w = _phys_weight(name)
                mask = extras.get(f"{name}_mask", None)
                mask = mask.to(dtype=torch.bool) if torch.is_tensor(mask) else None
                y_cart = extras[name]
                y_blocks = _get_embed(y_cart, 2, include_trace_chain=True)
                p2 = phys_pred[name].get(2) if isinstance(phys_pred[name], dict) else None
                if p2 is None or 2 not in y_blocks:
                    continue
                if mask is not None and mask.shape[0] == p2.shape[0]:
                    phys_loss = phys_loss + w * self.criterion(p2[mask].view(-1), y_blocks[2][mask].view(-1))
                else:
                    phys_loss = phys_loss + w * self.criterion(p2.view(-1), y_blocks[2].view(-1))

        return phys_loss

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle loading into DDP model
        if self.distributed:
            self.e3trans.module.load_state_dict(checkpoint['e3trans_state_dict'])
        else:
            self.e3trans.load_state_dict(checkpoint['e3trans_state_dict'])

        # Restore scheduler state when available.
        # This avoids needing to "fast-forward" with scheduler.step(epoch), which triggers
        # PyTorch's epoch-argument deprecation warning.
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                if self.is_main_process:
                    logging.warning(f"Failed to load scheduler state_dict from checkpoint: {e}")
        else:
            if self.is_main_process:
                logging.info("Checkpoint has no scheduler_state_dict; LR scheduler will restart fresh.")

        # Restore LR values without saving optimizer state (keeps checkpoints small).
        # If missing, we accept that the first resumed batch will use the freshly-initialized LR.
        if 'optimizer_lrs' in checkpoint:
            try:
                lrs = checkpoint['optimizer_lrs']
                if isinstance(lrs, (list, tuple)) and len(lrs) == len(self.optimizer.param_groups):
                    for pg, lr in zip(self.optimizer.param_groups, lrs):
                        pg['lr'] = float(lr)
            except Exception as e:
                if self.is_main_process:
                    logging.warning(f"Failed to restore LR values from checkpoint: {e}")
        
        # Load loss weights (only if use_checkpoint_loss_weights is True)
        if self.use_checkpoint_loss_weights:
            if 'a' in checkpoint:
                self.a = checkpoint['a']
            if 'b' in checkpoint:
                self.b = checkpoint['b']
            if 'c' in checkpoint:
                self.c = checkpoint['c']
            self._clamp_loss_weights()
            if self.is_main_process:
                logging.info(f"  Using loss weights from checkpoint: a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}")
        else:
            self._clamp_loss_weights()
            if self.is_main_process:
                logging.info(f"  Using new loss weights (ignoring checkpoint): a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}")
        if 'swa_applied' in checkpoint:
            self.swa_applied = checkpoint['swa_applied']
        if 'ema_enabled' in checkpoint:
            self.ema_enabled = checkpoint['ema_enabled']
        if 'e3trans_ema_state_dict' in checkpoint:
            # Lazily initialize EMA model
            if self.e3trans_ema is None:
                base = self.e3trans.module if self.distributed else self.e3trans
                self.e3trans_ema = copy.deepcopy(base).to(self.device)
                self.e3trans_ema.eval()
            self.e3trans_ema.load_state_dict(checkpoint['e3trans_ema_state_dict'])
        
        # Load training state for restart
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        if 'batch_count' in checkpoint:
            self.batch_count = checkpoint['batch_count']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']
        if 'physical_tensor_weights' in checkpoint:
            self.physical_tensor_weights = checkpoint['physical_tensor_weights']
        
        self.resumed_from_checkpoint = True
        
        # Load existing loss CSV so resumed training appends instead of overwriting
        if self.is_main_process and os.path.exists(self.loss_csv_path):
            try:
                self.loss_records = pd.read_csv(self.loss_csv_path).to_dict('records')
                logging.info(f"  Loaded {len(self.loss_records)} existing loss records from {self.loss_csv_path}")
            except Exception as e:
                logging.warning(f"  Failed to load existing loss CSV, will start fresh: {e}")
        
        if self.is_main_process:
            logging.info("=" * 80)
            logging.info("Checkpoint Loaded Successfully!")
            logging.info(f"  Resuming from epoch: {self.start_epoch}")
            logging.info(f"  Batch count: {self.batch_count}")
            if self.use_checkpoint_loss_weights:
                logging.info(f"  Loss weights (from checkpoint): a={self.a:.4f}, b={self.b:.4f}")
            else:
                logging.info(f"  Loss weights (new, ignoring checkpoint): a={self.a:.4f}, b={self.b:.4f}")
            logging.info(f"  Best validation metric (E_RMSE + b/a * F_RMSE): {self.best_val_loss:.6f}")
            logging.info(f"  Early stopping patience counter: {self.patience_counter}/{self.patience}")
            logging.info("=" * 80)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        # Set epoch for distributed sampler
        if self.distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        self.model.train()
        self.e3trans.train()  # DDP model's train() will call internal module's train()
        
        total_batch_loss = []
        total_batch_energy_loss = []
        total_batch_energy_rmse = []
        total_batch_energy_rmse_avg = []
        total_batch_force_loss = []
        total_batch_force_rmse = []
        total_batch_stress_loss = []
        total_batch_stress_rmse = []
        total_batch_phys_loss = []
        
        all_nets = [self.e3trans, self.model]
        all_parameters = [param for net in all_nets for param in net.parameters()]
        
        # Log epoch start
        total_batches = len(self.train_loader)
        epoch_start_time = time.time()
        # Log epoch start only to file
        if self.is_main_process:
            logging.info(f"Epoch {epoch} started - Total batches: {total_batches}")
        
        batch_times = []
        
        for batch_idx_loader, batch_data in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            if batch_data is None:
                continue
            
            extras = {}
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 11:
                (pos, A, batch_idx, force_ref, target_energies,
                 edge_src, edge_dst, edge_shifts, cell, stress_ref, extras) = batch_data
            else:
                (pos, A, batch_idx, force_ref, target_energies,
                 edge_src, edge_dst, edge_shifts, cell, stress_ref) = batch_data
            
            # Move to device
            pos = pos.to(self.device)
            A = A.to(self.device)
            batch_idx = batch_idx.to(self.device)
            force_ref = force_ref.to(self.device)
            target_energies = target_energies.to(self.device)
            edge_src = edge_src.to(self.device)
            edge_dst = edge_dst.to(self.device)
            edge_shifts = edge_shifts.to(self.device)
            cell = cell.to(self.device)
            stress_ref = stress_ref.to(self.device)
            extras = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in (extras or {}).items()}
            
            pos.requires_grad_(True)
            
            self.batch_count += 1

            # Initialize EMA when reaching ema_start_epoch
            if self.ema_start_epoch is not None and epoch >= self.ema_start_epoch and not self.ema_enabled:
                base = self.e3trans.module if self.distributed else self.e3trans
                self.e3trans_ema = copy.deepcopy(base).to(self.device)
                self.e3trans_ema.eval()
                for p in self.e3trans_ema.parameters():
                    p.requires_grad_(False)
                self.ema_enabled = True
                
                # Sync EMA enabled state across all processes
                if self.distributed:
                    ema_enabled_tensor = torch.tensor([1 if self.ema_enabled else 0], dtype=torch.int, device=self.device)
                    dist.broadcast(ema_enabled_tensor, src=0)
                    self.ema_enabled = bool(ema_enabled_tensor.item())
                
                if self.is_main_process:
                    logging.info(f"EMA initialized at epoch {epoch} with decay={self.ema_decay}")
            
            # Update loss weights: either SWA (direct switch) or continuous growth/decay
            if self.swa_start_epoch is not None and epoch >= self.swa_start_epoch and not self.swa_applied:
                # Apply SWA: directly set a and b to SWA values
                self.a = self.swa_a
                self.b = self.swa_b
                self._clamp_loss_weights()
                self.swa_applied = True

                # IMPORTANT: early-stopping metric is energy_loss + force_loss (unweighted).
                # When we switch (a, b), the metric scale/priority changes, so we should
                # reset early-stopping state to avoid premature stopping.
                self.patience_counter = 0
                self.best_val_loss = float('inf')
                self.best_checkpoint_path = None  # Reset best checkpoint path when SWA is applied

                # Sync SWA state across all processes
                if self.distributed:
                    swa_applied_tensor = torch.tensor([1 if self.swa_applied else 0], dtype=torch.int, device=self.device)
                    dist.broadcast(swa_applied_tensor, src=0)
                    self.swa_applied = bool(swa_applied_tensor.item())
                    # Sync a and b values
                    ab_tensor = torch.tensor([self.a, self.b], dtype=torch.float64, device=self.device)
                    dist.broadcast(ab_tensor, src=0)
                    self.a = ab_tensor[0].item()
                    self.b = ab_tensor[1].item()

                if self.is_main_process:
                    logging.info(
                        f"SWA applied at epoch {epoch}: a={self.a:.4f}, b={self.b:.4f}. "
                        f"Early-stopping state reset (best_val_loss=inf, patience_counter=0)."
                    )
            elif self.swa_start_epoch is None or epoch < self.swa_start_epoch:
                # Continuous growth/decay (only if SWA not applied yet)
                # Update weights on rank 0, then broadcast to all processes
                weights_updated = False
                if self.is_main_process and self.batch_count % self.update_param == 0:
                    self.a *= self.weight_a_growth
                    self.b *= self.weight_b_decay
                    # Clamp weights to configured bounds
                    if self.a_min is not None:
                        self.a = max(self.a, self.a_min)
                    if self.a_max is not None:
                        self.a = min(self.a, self.a_max)
                    if self.b_min is not None:
                        self.b = max(self.b, self.b_min)
                    if self.b_max is not None:
                        self.b = min(self.b, self.b_max)
                    weights_updated = True
                
                # Sync loss weights (a, b) across all processes after update
                if self.distributed:
                    ab_tensor = torch.tensor([self.a, self.b], dtype=torch.float64, device=self.device)
                    dist.broadcast(ab_tensor, src=0)
                    self.a = ab_tensor[0].item()
                    self.b = ab_tensor[1].item()
            
            self.optimizer.zero_grad()
            
            # Map atomic energies
            mapped_A = map_tensor_values(A, self.keys, self.values).to(self.device)
            E_offset_mol = scatter(mapped_A, batch_idx, dim=0, reduce='sum')
            
            # Apply per-molecule strain for stress computation
            compute_stress = (self.c > 0)
            if compute_stress:
                num_mol = cell.shape[0]
                strain = torch.zeros((num_mol, 3, 3), dtype=pos.dtype, device=self.device, requires_grad=True)
                symmetric_strain = 0.5 * (strain + strain.transpose(-1, -2))
                I3 = torch.eye(3, dtype=pos.dtype, device=self.device)
                deformation = I3 + symmetric_strain  # [B, 3, 3]
                deformation_per_atom = deformation[batch_idx]  # [N_atoms, 3, 3]
                pos_input = torch.einsum('ni,nij->nj', pos, deformation_per_atom)
                cell_input = torch.bmm(cell, deformation)
            else:
                strain = None
                pos_input = pos
                cell_input = cell

            # Forward pass (optional external field + optional physical tensor heads)
            forward_kwargs = {}
            if "external_field" in extras:
                forward_kwargs["external_tensor"] = extras["external_field"]

            want_phys = any(
                k in extras for k in (
                    "charge", "dipole", "polarizability", "quadrupole",
                    "charge_per_atom", "dipole_per_atom", "polarizability_per_atom", "quadrupole_per_atom",
                )
            )
            base_e3 = self.e3trans.module if (self.distributed and hasattr(self.e3trans, "module")) else self.e3trans
            supports_phys = hasattr(base_e3, "physical_tensor_heads") and base_e3.physical_tensor_heads is not None

            phys_pred = None
            if want_phys and supports_phys:
                try:
                    E_per_atom, phys_pred = self.e3trans(
                        pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input,
                        return_physical_tensors=True,
                        **forward_kwargs,
                    )
                except TypeError:
                    E_per_atom, phys_pred = self.e3trans(
                        pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input,
                        return_physical_tensors=True,
                    )
            else:
                try:
                    E_per_atom = self.e3trans(pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input, **forward_kwargs)
                except TypeError:
                    E_per_atom = self.e3trans(pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input)
            E_conv_mol = scatter(E_per_atom, batch_idx, dim=0, reduce='sum').squeeze(-1)
            E_mean = E_conv_mol + E_offset_mol
            
            # Compute forces (and stress if enabled)
            grad_targets = [pos]
            if compute_stress:
                grad_targets.append(strain)
            
            grads = torch.autograd.grad(
                E_mean.sum(),
                grad_targets,
                create_graph=True,
                retain_graph=True
            )
            
            force_grads = grads[0]
            fx_pred = self.train_dataset.restore_force(-force_grads[:, 0])
            fy_pred = self.train_dataset.restore_force(-force_grads[:, 1])
            fz_pred = self.train_dataset.restore_force(-force_grads[:, 2])
            f_pred = torch.stack([fx_pred, fy_pred, fz_pred], dim=1)
            
            # Compute predicted stress: σ = (1/V) * dE/dε
            if compute_stress:
                stress_grads = grads[1]  # [B, 3, 3]
                volume = torch.abs(torch.det(cell))  # [B]
                volume = torch.clamp(volume, min=1e-10)
                stress_pred = stress_grads / volume.view(-1, 1, 1)  # [B, 3, 3]
            
            force_ref_scaled = force_ref * self.force_shift_value
            force_loss = self.criterion(f_pred.view(-1), force_ref_scaled.view(-1))
            
            with torch.no_grad():
                force_rmse = torch.sqrt(self.criterion_2(f_pred.view(-1), force_ref_scaled.view(-1)))
            
            # Stress loss
            if compute_stress:
                stress_loss = self.criterion(stress_pred.view(-1), stress_ref.view(-1))
                with torch.no_grad():
                    stress_rmse = torch.sqrt(self.criterion_2(stress_pred.view(-1), stress_ref.view(-1)))
            else:
                stress_loss = torch.tensor(0.0, device=self.device)
                stress_rmse = torch.tensor(0.0, device=self.device)

            # Energy loss
            num_atoms_per_mol = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum')
            E_avg_pred = E_mean / num_atoms_per_mol
            target_energy_avg = target_energies / num_atoms_per_mol
            energy_loss = self.criterion(E_avg_pred, target_energy_avg)
            
            # Log energy predictions every N batches (only to log file, not console, only on main process)
            if self.is_main_process and self.batch_count % self.energy_log_frequency == 0:
                # Get unique molecule indices in this batch
                unique_mol_indices = torch.unique(batch_idx).cpu().numpy()
                num_molecules = len(unique_mol_indices)
                
                # Log summary for all molecules in batch
                E_pred_np = E_mean.detach().cpu().numpy()
                E_target_np = target_energies.detach().cpu().numpy()
                E_avg_pred_np = E_avg_pred.detach().cpu().numpy()
                E_avg_target_np = target_energy_avg.detach().cpu().numpy()
                
                logging.debug(f"Batch {self.batch_count} - Energy Predictions ({num_molecules} molecules):")
                for i, mol_idx in enumerate(unique_mol_indices):
                    mol_mask = (batch_idx == mol_idx)
                    num_atoms = mol_mask.sum().item()
                    e_pred_val = E_pred_np[mol_idx]
                    e_target_val = E_target_np[mol_idx]
                    e_avg_pred_val = E_avg_pred_np[mol_idx]
                    e_avg_target_val = E_avg_target_np[mol_idx]
                    error = e_pred_val - e_target_val
                    error_avg = e_avg_pred_val - e_avg_target_val
                    
                    logging.debug(
                        f"  Mol {i+1} (idx={mol_idx}, {num_atoms} atoms): "
                        f"E_pred={e_pred_val:12.6f} eV, E_target={e_target_val:12.6f} eV, "
                        f"Error={error:8.4f} eV | "
                        f"E_avg_pred={e_avg_pred_val:10.6f} eV/atom, E_avg_target={e_avg_target_val:10.6f} eV/atom, "
                        f"Error={error_avg:8.4f} eV/atom"
                    )
            
            with torch.no_grad():
                energy_rmse = torch.sqrt(self.criterion_2(E_mean, target_energies))
                energy_rmse_avg = torch.sqrt(self.criterion_2(E_avg_pred, target_energy_avg))
            
            # Total loss
            total_loss = self.a * energy_loss + self.b * force_loss + self.c * stress_loss

            # Optional: physical tensor supervised loss (Cartesian label -> ICTD irreps)
            phys_loss = torch.tensor(0.0, device=self.device)
            if want_phys and supports_phys and phys_pred is not None:
                phys_loss = self._compute_physical_tensor_loss(extras, phys_pred, batch_idx, base_e3)
                total_loss = total_loss + phys_loss
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=self.max_norm_value, norm_type=2.0)
            
            if self.batch_count % self.gradient_log_interval == 0:
                log_gradient_statistics(all_nets, self.batch_count, logging)
            
            self.optimizer.step()

            # Update EMA after optimizer step
            if self.ema_enabled and self.e3trans_ema is not None:
                with torch.no_grad():
                    current = self.e3trans.module if self.distributed else self.e3trans
                    for ema_param, cur_param in zip(self.e3trans_ema.parameters(), current.parameters()):
                        ema_param.data.mul_(self.ema_decay).add_(cur_param.data, alpha=1.0 - self.ema_decay)
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] < self.min_learning_rate:
                    param_group['lr'] = self.min_learning_rate
            
            if self.batch_count % self.vhat_clamp_interval == 0:
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        state = self.optimizer.state[param]
                        if 'v_hat' in state:
                            current_max = state['v_hat'].max().item()
                            state['v_hat'].clamp_(max=current_max * self.max_vhat_growth_factor)
            
            self.scheduler.step()
            
            total_batch_loss.append(total_loss.item())
            total_batch_energy_loss.append(energy_loss.item())
            total_batch_energy_rmse.append(energy_rmse.item())
            total_batch_energy_rmse_avg.append(energy_rmse_avg.item())
            total_batch_force_loss.append(force_loss.item())
            total_batch_force_rmse.append(force_rmse.item())
            total_batch_stress_loss.append(stress_loss.item())
            total_batch_stress_rmse.append(stress_rmse.item())
            total_batch_phys_loss.append(phys_loss.item())
            
            # Record batch time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            # Log training progress only to file, not console (only on main process)
            if self.is_main_process:
                progress_interval = max(1, min(10, total_batches // 20))  # Show at least 20 updates per epoch
                if (batch_idx_loader + 1) % progress_interval == 0 or (batch_idx_loader + 1) == total_batches:
                    current_energy_loss = np.mean(total_batch_energy_loss[-10:]) if len(total_batch_energy_loss) >= 10 else total_batch_energy_loss[-1] if total_batch_energy_loss else 0
                    current_force_loss = np.mean(total_batch_force_loss[-10:]) if len(total_batch_force_loss) >= 10 else total_batch_force_loss[-1] if total_batch_force_loss else 0
                    current_stress_loss = np.mean(total_batch_stress_loss[-10:]) if len(total_batch_stress_loss) >= 10 else total_batch_stress_loss[-1] if total_batch_stress_loss else 0
                    current_phys_loss = np.mean(total_batch_phys_loss[-10:]) if len(total_batch_phys_loss) >= 10 else total_batch_phys_loss[-1] if total_batch_phys_loss else 0
                    current_lr = self.optimizer.param_groups[0]['lr']
                    avg_batch_time = np.mean(batch_times[-10:]) if len(batch_times) >= 10 else np.mean(batch_times)
                    elapsed_time = batch_end_time - epoch_start_time
                    # Estimate remaining time
                    remaining_batches = total_batches - (batch_idx_loader + 1)
                    eta = remaining_batches * avg_batch_time
                    # Log to file only
                    stress_log = f" | Stress Loss: {current_stress_loss:.6f}" if self.c > 0 else ""
                    phys_log = f" | Phys Loss: {current_phys_loss:.6f}" if current_phys_loss > 0 else ""
                    logging.info(
                        f"Epoch {epoch} | Batch {batch_idx_loader + 1}/{total_batches} | "
                        f"Energy Loss: {current_energy_loss:.6f} | Force Loss: {current_force_loss:.6f}{stress_log}{phys_log} | "
                        f"LR: {current_lr:.2e} | Batch Count: {self.batch_count} | "
                        f"Batch Time: {batch_time:.3f}s | Avg: {avg_batch_time:.3f}s | "
                        f"Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s"
                    )
            
            if self.batch_count % self.dump_frequency == 0:
                full_train_metrics = self.compute_full_train_metrics(epoch)
                self.validate(epoch, full_train_metrics)
        
        # Log epoch summary
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        if self.is_main_process:
            logging.info(
                f"Epoch {epoch} finished - Total time: {epoch_time:.1f}s | "
                f"Avg batch time: {avg_batch_time:.3f}s | Total batches: {len(batch_times)}"
            )
        
        return {
            'total_loss': np.mean(total_batch_loss) if total_batch_loss else 0,
            'energy_loss': np.mean(total_batch_energy_loss) if total_batch_energy_loss else 0,
            'energy_rmse': np.mean(total_batch_energy_rmse) if total_batch_energy_rmse else 0,
            'energy_rmse_avg': np.mean(total_batch_energy_rmse_avg) if total_batch_energy_rmse_avg else 0,
            'force_loss': np.mean(total_batch_force_loss) if total_batch_force_loss else 0,
            'force_rmse': np.mean(total_batch_force_rmse) if total_batch_force_rmse else 0,
            'stress_loss': np.mean(total_batch_stress_loss) if total_batch_stress_loss else 0,
            'stress_rmse': np.mean(total_batch_stress_rmse) if total_batch_stress_rmse else 0,
            'phys_loss': np.mean(total_batch_phys_loss) if total_batch_phys_loss else 0,
            'epoch_time': epoch_time,
            'avg_batch_time': avg_batch_time,
        }
    
    def _gather_variable_tensors(self, local_tensor):
        """
        Gather tensors of variable sizes from all processes.
        
        Args:
            local_tensor: Local tensor on this process (can have different size on each process)
            
        Returns:
            Concatenated tensor from all processes (on all ranks)
        """
        if not self.distributed:
            return local_tensor
        
        # Get local size
        local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device=self.device)
        
        # Gather all sizes
        all_sizes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, local_size)
        all_sizes = [s.item() for s in all_sizes]
        max_size = max(all_sizes)
        
        # Pad local tensor to max size
        if local_tensor.dim() == 1:
            padded_tensor = torch.zeros(max_size, dtype=local_tensor.dtype, device=self.device)
            padded_tensor[:local_tensor.shape[0]] = local_tensor.to(self.device)
        else:
            # For multi-dimensional tensors (e.g., forces with shape [N, 3])
            padded_shape = list(local_tensor.shape)
            padded_shape[0] = max_size
            padded_tensor = torch.zeros(padded_shape, dtype=local_tensor.dtype, device=self.device)
            padded_tensor[:local_tensor.shape[0]] = local_tensor.to(self.device)
        
        # Gather padded tensors
        gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, padded_tensor)
        
        # Concatenate and remove padding
        result_parts = []
        for i, tensor in enumerate(gathered_tensors):
            result_parts.append(tensor[:all_sizes[i]].cpu())
        
        return torch.cat(result_parts)
    
    def validate(self, epoch, train_metrics=None):
        """Validate model.
        
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary containing training metrics at this batch count
        """
        # Set epoch for distributed sampler to ensure consistent data sharding
        if self.distributed and self.val_sampler is not None:
            self.val_sampler.set_epoch(epoch)
        
        self.model.eval()
        self.e3trans.eval()  # DDP model's eval() will call internal module's eval()
        
        val_E_preds = []
        val_E_targets = []
        val_E_avg_preds = []
        val_E_avg_targets = []
        val_F_preds = []
        val_F_targets = []
        val_S_preds = []
        val_S_targets = []
        val_phys_loss_list = []
        
        total_batches = len(self.val_loader)
        if self.is_main_process:
            logging.info("=" * 80)
            logging.info(f"Validation Started - Epoch {epoch} | Batch {self.batch_count}")
            logging.info("=" * 80)
        
        for batch_idx_loader, batch in enumerate(self.val_loader):
            if batch is None:
                continue
            
            extras = {}
            if isinstance(batch, (list, tuple)) and len(batch) == 11:
                (pos, A, batch_idx, force_ref, target_energies,
                 edge_src, edge_dst, edge_shifts, cell, stress_ref, extras) = batch
            else:
                (pos, A, batch_idx, force_ref, target_energies,
                 edge_src, edge_dst, edge_shifts, cell, stress_ref) = batch
            
            pos = pos.to(self.device)
            A = A.to(self.device)
            batch_idx = batch_idx.to(self.device)
            force_ref = force_ref.to(self.device)
            target_energies = target_energies.to(self.device)
            edge_src = edge_src.to(self.device)
            edge_dst = edge_dst.to(self.device)
            edge_shifts = edge_shifts.to(self.device)
            cell = cell.to(self.device)
            stress_ref = stress_ref.to(self.device)
            extras = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in (extras or {}).items()}
            
            pos.requires_grad = True
            
            mapped_A = map_tensor_values(A, self.keys, self.values).to(self.device)
            E_offset_val = scatter(mapped_A, batch_idx, dim=0, reduce='sum')

            # Choose evaluation model: EMA (optional) vs current
            if self.use_ema_for_validation and self.ema_enabled and self.e3trans_ema is not None:
                eval_model = self.e3trans_ema
                is_ema = True
            else:
                eval_model = self.e3trans.module if self.distributed else self.e3trans
                is_ema = False

            # Optional: compile validation forward (CUDA only)
            eval_model = self._maybe_compile_validation_model(
                eval_model,
                batch_tensors=(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell),
                is_ema=is_ema,
            )
            if self.device.type == "cuda" and self.compile_val != "none":
                self._cudagraph_step_begin_if_available()

            # Apply per-molecule strain for stress computation
            compute_stress = (self.c > 0)
            if compute_stress:
                num_mol = cell.shape[0]
                strain = torch.zeros((num_mol, 3, 3), dtype=pos.dtype, device=self.device, requires_grad=True)
                symmetric_strain = 0.5 * (strain + strain.transpose(-1, -2))
                I3 = torch.eye(3, dtype=pos.dtype, device=self.device)
                deformation = I3 + symmetric_strain
                deformation_per_atom = deformation[batch_idx]
                pos_input = torch.einsum('ni,nij->nj', pos, deformation_per_atom)
                cell_input = torch.bmm(cell, deformation)
            else:
                strain = None
                pos_input = pos
                cell_input = cell

            forward_kwargs = {}
            if "external_field" in extras:
                forward_kwargs["external_tensor"] = extras["external_field"]
            want_phys = any(
                k in extras for k in (
                    "charge", "dipole", "polarizability", "quadrupole",
                    "charge_per_atom", "dipole_per_atom", "polarizability_per_atom", "quadrupole_per_atom",
                )
            )
            supports_phys = hasattr(eval_model, "physical_tensor_heads") and eval_model.physical_tensor_heads is not None
            phys_pred = None
            if want_phys and supports_phys:
                try:
                    out = eval_model(
                        pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input,
                        return_physical_tensors=True,
                        **forward_kwargs,
                    )
                except TypeError:
                    out = eval_model(pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input)
            else:
                try:
                    out = eval_model(pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input, **forward_kwargs)
                except TypeError:
                    out = eval_model(pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input)
            E_per_atom = out[0] if isinstance(out, tuple) else out
            if isinstance(out, tuple) and len(out) >= 2:
                phys_pred = out[1]
            E_conv_val = scatter(E_per_atom, batch_idx, dim=0, reduce='sum')
            E_mean_val = E_conv_val.view(-1) + E_offset_val
            
            # Calculate forces (and stress if enabled)
            grad_targets = [pos]
            if compute_stress:
                grad_targets.append(strain)
            
            grads = torch.autograd.grad(
                E_mean_val.sum(), grad_targets, create_graph=False, retain_graph=False
            )
            
            force_grads = grads[0]
            fx_pred = self.val_dataset.restore_force(-force_grads[:, 0])
            fy_pred = self.val_dataset.restore_force(-force_grads[:, 1])
            fz_pred = self.val_dataset.restore_force(-force_grads[:, 2])
            f_pred_final = torch.stack([fx_pred, fy_pred, fz_pred], dim=1)
            
            if compute_stress:
                stress_grads = grads[1]  # [B, 3, 3]
                volume = torch.abs(torch.det(cell))
                volume = torch.clamp(volume, min=1e-10)
                stress_pred = stress_grads / volume.view(-1, 1, 1)
            
            f_ref_final = force_ref * self.force_shift_value
            
            # Detach and move to CPU (no gradients needed after force calculation)
            with torch.no_grad():
                E_pred_batch = E_mean_val.detach().cpu()
                E_target_batch = target_energies.detach().cpu()
                
                val_E_preds.append(E_pred_batch)
                val_E_targets.append(E_target_batch)
                
                num_atoms_per_mol = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum')
                val_E_avg_preds.append((E_mean_val / num_atoms_per_mol).detach().cpu())
                val_E_avg_targets.append((target_energies / num_atoms_per_mol).detach().cpu())
                
                val_F_preds.append(f_pred_final.detach().cpu())
                val_F_targets.append(f_ref_final.detach().cpu())

                if compute_stress:
                    val_S_preds.append(stress_pred.detach().cpu())
                    val_S_targets.append(stress_ref.detach().cpu())
                if want_phys and supports_phys and phys_pred is not None:
                    val_phys_loss_list.append(
                        self._compute_physical_tensor_loss(extras, phys_pred, batch_idx, eval_model).detach().item()
                    )
                
                # Stream output: log first sample's restored energy (consistent with original train-val.py)
                # Only log on main process
                if self.is_main_process:
                    restored_pred = self.val_dataset.restore_energy(E_mean_val[0].item())
                    restored_target = self.val_dataset.restore_energy(target_energies[0].item())
                    # Use debug level if console output is disabled (will only go to file)
                    # Use info level if console output is enabled
                    log_level = logging.info if self.log_val_batch_energy_to_console else logging.debug
                    log_level(f"Batch {batch_idx_loader + 1}/{total_batches} Sample -> Pred: {restored_pred:.4f} , True: {restored_target:.4f}")
        
        # Check if we have any validation data on this process
        local_has_data = len(val_E_preds) > 0
        
        # In distributed mode, check if any process has no data
        if self.distributed:
            has_data_tensor = torch.tensor([1 if local_has_data else 0], device=self.device)
            dist.all_reduce(has_data_tensor, op=dist.ReduceOp.SUM)
            if has_data_tensor.item() == 0:
                if self.is_main_process:
                    logging.warning("No validation data found on any process! Skipping validation metrics.")
                return
        elif not local_has_data:
            if self.is_main_process:
                logging.warning("No validation data found! Skipping validation metrics.")
            return
        
        # If this process has no data, create empty tensors for gathering
        if not local_has_data:
            # Create empty tensors with correct dtype
            all_E_preds = torch.tensor([], dtype=torch.get_default_dtype())
            all_E_targets = torch.tensor([], dtype=torch.get_default_dtype())
            all_E_avg_preds = torch.tensor([], dtype=torch.get_default_dtype())
            all_E_avg_targets = torch.tensor([], dtype=torch.get_default_dtype())
            all_F_preds = torch.zeros((0, 3), dtype=torch.get_default_dtype())
            all_F_targets = torch.zeros((0, 3), dtype=torch.get_default_dtype())
            all_S_preds = torch.zeros((0, 3, 3), dtype=torch.get_default_dtype())
            all_S_targets = torch.zeros((0, 3, 3), dtype=torch.get_default_dtype())
        else:
            # Concatenate all predictions and targets
            all_E_preds = torch.cat(val_E_preds)
            all_E_targets = torch.cat(val_E_targets)
            all_E_avg_preds = torch.cat(val_E_avg_preds)
            all_E_avg_targets = torch.cat(val_E_avg_targets)
            all_F_preds = torch.cat(val_F_preds)
            all_F_targets = torch.cat(val_F_targets)
            if val_S_preds:
                all_S_preds = torch.cat(val_S_preds)
                all_S_targets = torch.cat(val_S_targets)
            else:
                all_S_preds = torch.zeros((0, 3, 3), dtype=torch.get_default_dtype())
                all_S_targets = torch.zeros((0, 3, 3), dtype=torch.get_default_dtype())
        
        # In distributed training, gather results from all processes using variable-size gather
        if self.distributed:
            all_E_preds = self._gather_variable_tensors(all_E_preds)
            all_E_targets = self._gather_variable_tensors(all_E_targets)
            all_E_avg_preds = self._gather_variable_tensors(all_E_avg_preds)
            all_E_avg_targets = self._gather_variable_tensors(all_E_avg_targets)
            all_F_preds = self._gather_variable_tensors(all_F_preds)
            all_F_targets = self._gather_variable_tensors(all_F_targets)
            if all_S_preds.numel() > 0:
                all_S_preds = self._gather_variable_tensors(all_S_preds)
                all_S_targets = self._gather_variable_tensors(all_S_targets)
        
        # Verify shapes match
        assert all_E_preds.shape == all_E_targets.shape, \
            f"Energy shape mismatch: pred {all_E_preds.shape} vs target {all_E_targets.shape}"
        assert all_E_avg_preds.shape == all_E_avg_targets.shape, \
            f"Energy avg shape mismatch: pred {all_E_avg_preds.shape} vs target {all_E_avg_targets.shape}"
        assert all_F_preds.shape == all_F_targets.shape, \
            f"Force shape mismatch: pred {all_F_preds.shape} vs target {all_F_targets.shape}"
        
        # Log total number of samples (only on main process)
        if self.is_main_process:
            num_samples = len(all_E_preds)
            logging.info(f"[Validation] Processing {num_samples} samples")
        
        # Calculate all metrics (restore units consistent with original train-val.py)
        val_energy_rmse = torch.sqrt(F.mse_loss(all_E_preds, all_E_targets)).item()
        val_energy_rmse = self.val_dataset.restore_energy(val_energy_rmse)  # Restore units
        
        val_energy_mae = F.l1_loss(all_E_preds, all_E_targets).item()
        val_energy_mae = self.val_dataset.restore_energy(val_energy_mae)  # Restore units
        
        val_energy_rmse_avg = torch.sqrt(F.mse_loss(all_E_avg_preds, all_E_avg_targets)).item()
        val_energy_rmse_avg = self.val_dataset.restore_energy(val_energy_rmse_avg)  # Restore units
        
        val_energy_mae_avg = F.l1_loss(all_E_avg_preds, all_E_avg_targets).item()
        val_energy_mae_avg = self.val_dataset.restore_energy(val_energy_mae_avg)  # Restore units
        
        val_energy_loss = F.mse_loss(all_E_avg_preds, all_E_avg_targets).item()  # For logging
        
        # Force metrics (already restored in loop, but restore again for consistency)
        val_force_rmse = torch.sqrt(F.mse_loss(all_F_preds.view(-1), all_F_targets.view(-1))).item()
        val_force_rmse = self.val_dataset.restore_force(val_force_rmse)  # Restore units for consistency
        val_force_mae = F.l1_loss(all_F_preds.view(-1), all_F_targets.view(-1)).item()
        val_force_mae = self.val_dataset.restore_force(val_force_mae)  # Restore units for consistency
        val_force_loss = F.mse_loss(all_F_preds.view(-1), all_F_targets.view(-1)).item()  # For logging
        
        # Stress metrics
        if all_S_preds.numel() > 0:
            val_stress_rmse = torch.sqrt(F.mse_loss(all_S_preds.view(-1), all_S_targets.view(-1))).item()
            val_stress_mae = F.l1_loss(all_S_preds.view(-1), all_S_targets.view(-1)).item()
            val_stress_loss = F.mse_loss(all_S_preds.view(-1), all_S_targets.view(-1)).item()
        else:
            val_stress_rmse = 0.0
            val_stress_mae = 0.0
            val_stress_loss = 0.0

        # Keep total loss for logging, but use a composite metric for best/early-stop:
        # metric = val_energy_rmse + (b/a) * val_force_rmse
        val_phys_loss = float(np.mean(val_phys_loss_list)) if val_phys_loss_list else 0.0
        current_val_loss = self.a * val_energy_loss + self.b * val_force_loss + self.c * val_stress_loss + val_phys_loss
        ab_ratio = (self.b / self.a) if self.a != 0 else 0.0
        ac_ratio = (self.c / self.a) if self.a != 0 else 0.0
        current_val_metric = val_energy_rmse + ab_ratio * val_force_rmse + ac_ratio * val_stress_rmse

        # Log validation results (only on main process)
        if self.is_main_process:
            stress_log_lines = ""
            if self.c > 0:
                stress_log_lines = f"""
                                "Val Stress Loss (MSE)": {val_stress_loss},
                                "Val Stress RMSE": {val_stress_rmse},
                                "Val Stress MAE": {val_stress_mae},"""
            phys_log_lines = f"""
                                "Val Phys Loss (weighted)": {val_phys_loss},""" if val_phys_loss > 0 else ""
            logging.info(f"""
                                "Val Loss (MSE)": {val_energy_loss},
                                "Val Force Loss (MSE)": {val_force_loss},{stress_log_lines}{phys_log_lines}
                                "Val Total Loss (E+F+S, weighted)": {current_val_loss},
                                "Val Energy RMSE": {val_energy_rmse},
                                "Val Energy MAE": {val_energy_mae},
                                "Val Energy RMSE/atom": {val_energy_rmse_avg},
                                "Val Energy MAE/atom": {val_energy_mae_avg},
                                "Val Force RMSE":  {val_force_rmse},
                                "Val Force MAE": {val_force_mae},
                                "Val Metric (E_RMSE + b/a * F_RMSE + c/a * S_RMSE)": {current_val_metric},
                                "Current learning rate": {self.optimizer.param_groups[0]['lr']},
                                "a (energy weight)": {self.a},
                                "b (force weight)": {self.b},
                                "c (stress weight)": {self.c}
                                """)

        # Save validation results (only on main process)
        if self.is_main_process:
            # Save validation CSV files if enabled
            if self.save_val_csv:
                df_energy_val = pd.DataFrame({
                    "Target_Energy": all_E_targets.numpy(),
                    "Predicted_Energy": all_E_preds.numpy(),
                    "Delta": (all_E_preds - all_E_targets).numpy()
                })
                energy_filename = f"val_energy_epoch{epoch}_batch{self.batch_count}.csv"
                energy_save_path = os.path.join(self.val_results_dir, energy_filename)
                df_energy_val.to_csv(energy_save_path, index=False)
                
                f_pred_np = all_F_preds.numpy()
                f_true_np = all_F_targets.numpy()
                
                df_force_val = pd.DataFrame({
                    "Fx_True": f_true_np[:, 0], "Fy_True": f_true_np[:, 1], "Fz_True": f_true_np[:, 2],
                    "Fx_Pred": f_pred_np[:, 0], "Fy_Pred": f_pred_np[:, 1], "Fz_Pred": f_pred_np[:, 2]
                })
                force_filename = f"val_force_epoch{epoch}_batch{self.batch_count}.csv"
                force_save_path = os.path.join(self.val_results_dir, force_filename)
                df_force_val.to_csv(force_save_path, index=False)
        
        # Early stopping / best checkpoint - use composite metric only
        if self.distributed:
            # Broadcast current_val_metric from rank 0 to ensure consistency
            val_metric_tensor = torch.tensor([current_val_metric], device=self.device)
            dist.broadcast(val_metric_tensor, src=0)
            current_val_metric_synced = val_metric_tensor.item()
        else:
            current_val_metric_synced = current_val_metric
        
        if current_val_metric_synced < self.best_val_loss:
            self.best_val_loss = current_val_metric_synced
            self.patience_counter = 0
            
            # Save best checkpoint when validation metric improves
            if self.is_main_process:
                # Extract state_dict from DDP model if needed
                if self.distributed:
                    e3trans_state_dict = self.e3trans.module.state_dict()
                else:
                    e3trans_state_dict = self.e3trans.state_dict()
                
                best_checkpoint_dict = {
                    'e3trans_state_dict': e3trans_state_dict,
                    'a': self.a,
                    'b': self.b,
                    'c': self.c,
                    'batch_count': self.batch_count,
                    'epoch': epoch,
                    'best_val_loss': self.best_val_loss,
                    'patience_counter': self.patience_counter,
                    'swa_applied': self.swa_applied,
                    'ema_enabled': self.ema_enabled,
                    'tensor_product_mode': self.tensor_product_mode,
                    'inference_output_physical_tensors': self.inference_output_physical_tensors,
                    'physical_tensor_weights': self.physical_tensor_weights,
                    'atomic_energy_keys': self.keys.detach().cpu(),
                    'atomic_energy_values': self.values.detach().cpu(),
                }
                best_checkpoint_dict.update(self.e3_arch_metadata)
                if self.config is not None:
                    best_checkpoint_dict['max_radius'] = getattr(self.config, 'max_radius', 5.0)
                
                if self.save_ema_model and self.ema_enabled and self.e3trans_ema is not None:
                    best_checkpoint_dict['e3trans_ema_state_dict'] = self.e3trans_ema.state_dict()
                
                # Save best checkpoint to a temporary file
                self.best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model_temp.pth')
                torch.save(best_checkpoint_dict, self.best_checkpoint_path)
                logging.info(f"New best validation metric: {self.best_val_loss:.6f} - Best checkpoint saved")

            # Barrier: ensure rank 0 finishes writing before others proceed
            if self.distributed:
                dist.barrier()
        else:
            self.patience_counter += 1
        
        # Sync patience_counter across all processes
        if self.distributed:
            patience_tensor = torch.tensor([self.patience_counter], device=self.device)
            dist.broadcast(patience_tensor, src=0)
            self.patience_counter = int(patience_tensor.item())
        
        # Record validation metrics to CSV
        record = {
            'epoch': epoch,
            'batch_count': self.batch_count,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'a': self.a,
            'b': self.b,
            'c': self.c,
            'train_total_loss': 0,
            'train_energy_loss': 0,
            'train_force_loss': 0,
            'train_stress_loss': 0,
            'train_phys_loss': 0,
            'train_energy_rmse': 0,
            'train_energy_mae': 0,
            'train_energy_rmse_avg': 0,
            'train_energy_mae_avg': 0,
            'train_force_rmse': 0,
            'train_force_mae': 0,
            'train_stress_rmse': 0,
            'train_stress_mae': 0,
            'val_total_loss': current_val_loss,
            'val_energy_loss': val_energy_loss,
            'val_force_loss': val_force_loss,
            'val_stress_loss': val_stress_loss,
            'val_phys_loss': val_phys_loss,
            'val_energy_rmse': val_energy_rmse,
            'val_energy_mae': val_energy_mae,
            'val_energy_rmse_avg': val_energy_rmse_avg,
            'val_energy_mae_avg': val_energy_mae_avg,
            'val_force_rmse': val_force_rmse,
            'val_force_mae': val_force_mae,
            'val_stress_rmse': val_stress_rmse,
            'val_stress_mae': val_stress_mae,
        }
        
        # Add training metrics if provided
        if train_metrics:
            record.update({
                'train_total_loss': train_metrics.get('train_total_loss', 0),
                'train_energy_loss': train_metrics.get('train_energy_loss', 0),
                'train_force_loss': train_metrics.get('train_force_loss', 0),
                'train_stress_loss': train_metrics.get('train_stress_loss', 0),
                'train_phys_loss': train_metrics.get('train_phys_loss', 0),
                'train_energy_rmse': train_metrics.get('train_energy_rmse', 0),
                'train_energy_mae': train_metrics.get('train_energy_mae', 0),
                'train_energy_rmse_avg': train_metrics.get('train_energy_rmse_avg', 0),
                'train_energy_mae_avg': train_metrics.get('train_energy_mae_avg', 0),
                'train_force_rmse': train_metrics.get('train_force_rmse', 0),
                'train_force_mae': train_metrics.get('train_force_mae', 0),
                'train_stress_rmse': train_metrics.get('train_stress_rmse', 0),
                'train_stress_mae': train_metrics.get('train_stress_mae', 0),
            })
        
        # Only save on main process
        if self.is_main_process:
            self.loss_records.append(record)
            
            # Save to CSV (full history so resume continues recording without overwriting)
            df = pd.DataFrame(self.loss_records)
            df.to_csv(self.loss_csv_path, index=False)
            
            # Save model (only save e3trans as MainNet is not used in inference)
            # Extract state_dict from DDP model if needed
            if self.distributed:
                e3trans_state_dict = self.e3trans.module.state_dict()
            else:
                e3trans_state_dict = self.e3trans.state_dict()
            
            checkpoint_dict = {
                'e3trans_state_dict': e3trans_state_dict,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_lrs': [pg.get('lr', None) for pg in self.optimizer.param_groups],
                'a': self.a,
                'b': self.b,
                'c': self.c,
                'batch_count': self.batch_count,
                'epoch': epoch,
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
                'swa_applied': self.swa_applied,
                'ema_enabled': self.ema_enabled,
                'tensor_product_mode': self.tensor_product_mode,
                'inference_output_physical_tensors': self.inference_output_physical_tensors,
                'physical_tensor_weights': self.physical_tensor_weights,
                'atomic_energy_keys': self.keys.detach().cpu(),
                'atomic_energy_values': self.values.detach().cpu(),
            }
            checkpoint_dict.update(self.e3_arch_metadata)
            if self.config is not None:
                checkpoint_dict['max_radius'] = getattr(self.config, 'max_radius', 5.0)

            if self.save_ema_model and self.ema_enabled and self.e3trans_ema is not None:
                checkpoint_dict['e3trans_ema_state_dict'] = self.e3trans_ema.state_dict()

            # Add tensor product mode suffix to checkpoint filename
            base_filename = f'combined_model_epoch{epoch}_batch_count{self.batch_count}'
            checkpoint_filename = f'{base_filename}{self.tensor_product_suffix}.pth'
            checkpoint_filepath = os.path.join(self.checkpoint_dir, checkpoint_filename)
            torch.save(checkpoint_dict, checkpoint_filepath)

        if self.distributed:
            dist.barrier()

        self.model.train()
        self.e3trans.train()

    def compute_full_train_metrics(self, epoch):
        """Run a full pass (or sampled pass) over the training set to compute accurate train loss."""
        # Switch to eval mode to disable dropout / norm updates
        self.model.eval()
        self.e3trans.eval()

        eval_model = self.e3trans.module if self.distributed else self.e3trans

        energy_loss_sum = 0.0
        force_loss_sum = 0.0
        stress_loss_sum = 0.0
        phys_loss_sum = 0.0
        energy_count = 0
        force_count = 0
        stress_count = 0
        phys_count = 0

        energy_mse_sum = 0.0
        energy_mse_avg_sum = 0.0
        force_mse_sum = 0.0
        stress_mse_sum = 0.0
        energy_mae_sum = 0.0
        energy_mae_avg_sum = 0.0
        force_mae_sum = 0.0
        stress_mae_sum = 0.0

        # Determine if we should use sampling
        use_sampling = self.train_eval_sample_ratio < 1.0
        if use_sampling:
            # Get total number of batches for this process
            total_batches = len(self.train_loader)
            num_batches_to_eval = max(1, int(total_batches * self.train_eval_sample_ratio))
            
            # Generate random indices for sampling
            # Sync batch_count across all processes to ensure same random seed
            import random
            if self.distributed:
                # Broadcast batch_count from rank 0 to ensure all processes use the same seed
                batch_count_tensor = torch.tensor([self.batch_count], dtype=torch.long, device=self.device)
                dist.broadcast(batch_count_tensor, src=0)
                sync_batch_count = batch_count_tensor.item()
            else:
                sync_batch_count = self.batch_count
            # Use epoch and synced batch_count as seed for reproducibility (same seed across all processes)
            random.seed(epoch * 10000 + sync_batch_count)
            
            # Sample batch indices (each process samples from its own data subset)
            all_indices = list(range(total_batches))
            sampled_indices = set(random.sample(all_indices, num_batches_to_eval))
            
            if self.is_main_process:
                logging.info("=" * 80)
                logging.info(f"Train Evaluation Started (Sampled {self.train_eval_sample_ratio*100:.1f}%) - Epoch {epoch} | Batch {self.batch_count}")
                logging.info(f"  Evaluating {num_batches_to_eval}/{total_batches} batches per process")
                logging.info("=" * 80)
        else:
            sampled_indices = None
            if self.is_main_process:
                logging.info("=" * 80)
                logging.info(f"Full Train Evaluation Started - Epoch {epoch} | Batch {self.batch_count}")
                logging.info("=" * 80)

        batch_idx_loader = 0
        for batch in self.train_loader:
            # Skip batches if using sampling
            if use_sampling and batch_idx_loader not in sampled_indices:
                batch_idx_loader += 1
                continue
            
            batch_idx_loader += 1
            if batch is None:
                continue

            extras = {}
            if isinstance(batch, (list, tuple)) and len(batch) == 11:
                (pos, A, batch_idx, force_ref, target_energies,
                 edge_src, edge_dst, edge_shifts, cell, stress_ref, extras) = batch
            else:
                (pos, A, batch_idx, force_ref, target_energies,
                 edge_src, edge_dst, edge_shifts, cell, stress_ref) = batch

            pos = pos.to(self.device)
            A = A.to(self.device)
            batch_idx = batch_idx.to(self.device)
            force_ref = force_ref.to(self.device)
            target_energies = target_energies.to(self.device)
            edge_src = edge_src.to(self.device)
            edge_dst = edge_dst.to(self.device)
            edge_shifts = edge_shifts.to(self.device)
            cell = cell.to(self.device)
            stress_ref = stress_ref.to(self.device)

            pos.requires_grad_(True)

            compute_stress = (self.c > 0)
            if compute_stress:
                num_mol = cell.shape[0]
                strain = torch.zeros((num_mol, 3, 3), dtype=pos.dtype, device=self.device, requires_grad=True)
                symmetric_strain = 0.5 * (strain + strain.transpose(-1, -2))
                I3 = torch.eye(3, dtype=pos.dtype, device=self.device)
                deformation = I3 + symmetric_strain
                deformation_per_atom = deformation[batch_idx]
                pos_input = torch.einsum('ni,nij->nj', pos, deformation_per_atom)
                cell_input = torch.bmm(cell, deformation)
            else:
                strain = None
                pos_input = pos
                cell_input = cell

            mapped_A = map_tensor_values(A, self.keys, self.values).to(self.device)
            E_offset_mol = scatter(mapped_A, batch_idx, dim=0, reduce='sum')

            extras = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in (extras or {}).items()}
            want_phys = any(
                k in extras for k in (
                    "charge", "dipole", "polarizability", "quadrupole",
                    "charge_per_atom", "dipole_per_atom", "polarizability_per_atom", "quadrupole_per_atom",
                )
            )
            supports_phys = hasattr(eval_model, "physical_tensor_heads") and eval_model.physical_tensor_heads is not None
            phys_pred = None
            if want_phys and supports_phys:
                try:
                    out = eval_model(
                        pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input,
                        return_physical_tensors=True,
                    )
                except TypeError:
                    out = eval_model(pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input)
            else:
                out = eval_model(pos_input, A, batch_idx, edge_src, edge_dst, edge_shifts, cell_input)
            E_per_atom = out[0] if isinstance(out, tuple) else out
            if isinstance(out, tuple) and len(out) >= 2:
                phys_pred = out[1]
            E_conv_mol = scatter(E_per_atom, batch_idx, dim=0, reduce='sum').squeeze(-1)
            E_mean = E_conv_mol + E_offset_mol

            grad_targets = [pos]
            if compute_stress:
                grad_targets.append(strain)

            grads = torch.autograd.grad(
                E_mean.sum(),
                grad_targets,
                create_graph=False,
                retain_graph=False
            )

            force_grads = grads[0]
            fx_pred = self.train_dataset.restore_force(-force_grads[:, 0])
            fy_pred = self.train_dataset.restore_force(-force_grads[:, 1])
            fz_pred = self.train_dataset.restore_force(-force_grads[:, 2])
            f_pred = torch.stack([fx_pred, fy_pred, fz_pred], dim=1)

            if compute_stress:
                stress_grads = grads[1]
                volume = torch.abs(torch.det(cell))
                volume = torch.clamp(volume, min=1e-10)
                stress_pred = stress_grads / volume.view(-1, 1, 1)

            force_ref_scaled = force_ref * self.force_shift_value
            force_loss = self.criterion(f_pred.view(-1), force_ref_scaled.view(-1))

            num_atoms_per_mol = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum')
            E_avg_pred = E_mean / num_atoms_per_mol
            target_energy_avg = target_energies / num_atoms_per_mol
            energy_loss = self.criterion(E_avg_pred, target_energy_avg)

            # Accumulate sums for weighted averages
            num_molecules = int(E_avg_pred.numel())
            num_force_elements = int(force_ref_scaled.numel())

            energy_loss_sum += energy_loss.item() * num_molecules
            force_loss_sum += force_loss.item() * num_force_elements
            energy_count += num_molecules
            force_count += num_force_elements
            if want_phys and supports_phys and phys_pred is not None:
                p_loss = self._compute_physical_tensor_loss(extras, phys_pred, batch_idx, eval_model)
                phys_loss_sum += p_loss.item()
                phys_count += 1

            energy_mse_sum += F.mse_loss(E_mean, target_energies, reduction='sum').item()
            energy_mse_avg_sum += F.mse_loss(E_avg_pred, target_energy_avg, reduction='sum').item()
            force_mse_sum += F.mse_loss(f_pred.view(-1), force_ref_scaled.view(-1), reduction='sum').item()
            energy_mae_sum += F.l1_loss(E_mean, target_energies, reduction='sum').item()
            energy_mae_avg_sum += F.l1_loss(E_avg_pred, target_energy_avg, reduction='sum').item()
            force_mae_sum += F.l1_loss(f_pred.view(-1), force_ref_scaled.view(-1), reduction='sum').item()

            if compute_stress:
                num_stress_elements = int(stress_ref.numel())
                s_loss = self.criterion(stress_pred.view(-1), stress_ref.view(-1))
                stress_loss_sum += s_loss.item() * num_stress_elements
                stress_count += num_stress_elements
                stress_mse_sum += F.mse_loss(stress_pred.view(-1), stress_ref.view(-1), reduction='sum').item()
                stress_mae_sum += F.l1_loss(stress_pred.view(-1), stress_ref.view(-1), reduction='sum').item()

        # Sync metrics across processes
        if self.distributed:
            stats = torch.tensor([
                energy_loss_sum, force_loss_sum, stress_loss_sum,
                phys_loss_sum, energy_count, force_count, stress_count, phys_count,
                energy_mse_sum, energy_mse_avg_sum, force_mse_sum, stress_mse_sum,
                energy_mae_sum, energy_mae_avg_sum, force_mae_sum, stress_mae_sum,
            ], dtype=torch.float64, device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            (
                energy_loss_sum, force_loss_sum, stress_loss_sum,
                phys_loss_sum, energy_count, force_count, stress_count, phys_count,
                energy_mse_sum, energy_mse_avg_sum, force_mse_sum, stress_mse_sum,
                energy_mae_sum, energy_mae_avg_sum, force_mae_sum, stress_mae_sum,
            ) = stats.tolist()

        energy_loss_avg = energy_loss_sum / max(1, energy_count)
        force_loss_avg = force_loss_sum / max(1, force_count)
        stress_loss_avg = stress_loss_sum / max(1, stress_count)
        phys_loss_avg = phys_loss_sum / max(1, phys_count)

        energy_rmse = np.sqrt(energy_mse_sum / max(1, energy_count))
        energy_rmse_avg = np.sqrt(energy_mse_avg_sum / max(1, energy_count))
        force_rmse = np.sqrt(force_mse_sum / max(1, force_count))
        stress_rmse = np.sqrt(stress_mse_sum / max(1, stress_count))
        energy_mae = energy_mae_sum / max(1, energy_count)
        energy_mae_avg = energy_mae_avg_sum / max(1, energy_count)
        force_mae = force_mae_sum / max(1, force_count)
        stress_mae = stress_mae_sum / max(1, stress_count)

        energy_rmse = self.train_dataset.restore_energy(energy_rmse)
        energy_rmse_avg = self.train_dataset.restore_energy(energy_rmse_avg)
        energy_mae = self.train_dataset.restore_energy(energy_mae)
        energy_mae_avg = self.train_dataset.restore_energy(energy_mae_avg)
        
        force_rmse = self.train_dataset.restore_force(force_rmse)
        force_mae = self.train_dataset.restore_force(force_mae)

        train_total_loss = self.a * energy_loss_avg + self.b * force_loss_avg + self.c * stress_loss_avg + phys_loss_avg

        if self.is_main_process:
            eval_type = "Sampled" if use_sampling else "Full"
            sample_info = f" ({self.train_eval_sample_ratio*100:.1f}% sampled)" if use_sampling else ""
            stress_log = f" | Train Stress Loss: {stress_loss_avg:.6f}" if self.c > 0 else ""
            phys_log = f" | Train Phys Loss: {phys_loss_avg:.6f}" if phys_loss_avg > 0 else ""
            logging.info(
                f"{eval_type} Train Evaluation Done{sample_info} - "
                f"Train Loss (a*E+b*F+c*S+phys): {train_total_loss:.6f} | "
                f"Train Energy Loss: {energy_loss_avg:.6f} | "
                f"Train Force Loss: {force_loss_avg:.6f}{stress_log}{phys_log}"
            )

        # Restore training mode
        self.model.train()
        self.e3trans.train()

        return {
            'train_total_loss': train_total_loss,
            'train_energy_loss': energy_loss_avg,
            'train_force_loss': force_loss_avg,
            'train_stress_loss': stress_loss_avg,
            'train_phys_loss': phys_loss_avg,
            'train_energy_rmse': energy_rmse,
            'train_energy_rmse_avg': energy_rmse_avg,
            'train_energy_mae': energy_mae,
            'train_energy_mae_avg': energy_mae_avg,
            'train_force_rmse': force_rmse,
            'train_force_mae': force_mae,
            'train_stress_rmse': stress_rmse,
            'train_stress_mae': stress_mae,
        }
    
    def run_training(self):
        """Main training loop."""
        if self.is_main_process:
            if self.resumed_from_checkpoint:
                logging.info(f"Resuming training from epoch {self.start_epoch}/{self.epoch_numbers}")
            else:
                logging.info(f"Training started - Total epochs: {self.epoch_numbers}")
            logging.info(f"Checkpoint directory: {self.checkpoint_dir}")
            logging.info(f"Final checkpoint will be saved to: {self.checkpoint_save_path}")
        
        # Check if training is already complete
        if self.start_epoch > self.epoch_numbers:
            if self.is_main_process:
                logging.info(f"Training already completed (start epoch {self.start_epoch} > total epochs {self.epoch_numbers})")
                logging.info(f"Model is already at epoch {self.start_epoch - 1}")
                logging.info("No further training needed.")
            return
        
        last_epoch = self.start_epoch - 1  # Track the last completed epoch
        for epoch in range(self.start_epoch, self.epoch_numbers + 1):
            last_epoch = epoch
            metrics = self.train_epoch(epoch)
            
            if self.is_main_process:
                phys_log = f" | Phys Loss: {metrics.get('phys_loss', 0):.6f}" if metrics.get('phys_loss', 0) > 0 else ""
                logging.info(
                    f"Epoch {epoch}/{self.epoch_numbers} completed - "
                    f"Avg Energy Loss: {metrics['energy_loss']:.6f} | "
                    f"Avg Force Loss: {metrics['force_loss']:.6f}{phys_log} | "
                    f"Energy RMSE: {metrics['energy_rmse']:.6f} | "
                    f"Force RMSE: {metrics['force_rmse']:.6f} | "
                    f"Epoch Time: {metrics['epoch_time']:.1f}s | "
                    f"Avg Batch Time: {metrics['avg_batch_time']:.3f}s"
                )
            
            # Note: patience_counter is already synced in validate() method
            if self.patience_counter >= self.patience:
                if self.is_main_process:
                    logging.info(f"Early stopping triggered at epoch {epoch} (patience: {self.patience})")
                break
        
        # Save final checkpoint (only save e3trans as MainNet is not used in inference)
        # Save the best checkpoint (with best validation metric) as the final checkpoint
        if self.is_main_process:
            if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                # Load best checkpoint and save as final checkpoint
                logging.info(f"Loading best checkpoint from {self.best_checkpoint_path}...")
                best_checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
                
                best_epoch = best_checkpoint.get('epoch', 'unknown')
                best_batch_count = best_checkpoint.get('batch_count', 'unknown')
                
                # Update epoch to last_epoch for final checkpoint
                best_checkpoint['epoch'] = last_epoch
                
                logging.info(f"Saving final checkpoint (best model) to {self.checkpoint_save_path}...")
                logging.info(f"  Best validation metric (E_RMSE + b/a * F_RMSE): {self.best_val_loss:.6f}")
                logging.info(f"  Best model from epoch: {best_epoch}, batch_count: {best_batch_count}")
                
                torch.save(best_checkpoint, self.checkpoint_save_path)
                
                # Clean up temporary best checkpoint file
                os.remove(self.best_checkpoint_path)
                logging.info(f"Training completed! Best model saved to {self.checkpoint_save_path}")
            else:
                # Fallback: if no best checkpoint exists, save current model
                logging.warning("No best checkpoint found, saving current model as final checkpoint...")
                if self.distributed:
                    e3trans_state_dict = self.e3trans.module.state_dict()
                else:
                    e3trans_state_dict = self.e3trans.state_dict()
                
                checkpoint_dict = {
                    'epoch': last_epoch,
                    'e3trans_state_dict': e3trans_state_dict,
                    'a': self.a,
                    'b': self.b,
                    'c': self.c,
                    'batch_count': self.batch_count,
                    'best_val_loss': self.best_val_loss,
                    'patience_counter': self.patience_counter,
                    'swa_applied': self.swa_applied,
                    'ema_enabled': self.ema_enabled,
                    'tensor_product_mode': self.tensor_product_mode,
                    'inference_output_physical_tensors': self.inference_output_physical_tensors,
                    'physical_tensor_weights': self.physical_tensor_weights,
                    'atomic_energy_keys': self.keys.detach().cpu(),
                    'atomic_energy_values': self.values.detach().cpu(),
                }
                checkpoint_dict.update(self.e3_arch_metadata)
                if self.config is not None:
                    checkpoint_dict['max_radius'] = getattr(self.config, 'max_radius', 5.0)
                if self.save_ema_model and self.ema_enabled and self.e3trans_ema is not None:
                    checkpoint_dict['e3trans_ema_state_dict'] = self.e3trans_ema.state_dict()

                torch.save(checkpoint_dict, self.checkpoint_save_path)
                logging.info(f"Training completed! Final model saved to {self.checkpoint_save_path}")

        # Barrier: all ranks wait for rank 0 to finish writing final checkpoint
        if self.distributed:
            dist.barrier()
