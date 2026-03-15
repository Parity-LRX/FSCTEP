from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_PER_FIDELITY_STAT_KEYS = {
    "graph_count": 0,
    "atom_count": 1,
    "force_elem_count": 2,
    "stress_elem_count": 3,
    "energy_mse_sum": 4,
    "energy_mae_sum": 5,
    "energy_avg_mse_sum": 6,
    "energy_avg_mae_sum": 7,
    "force_mse_sum": 8,
    "force_mae_sum": 9,
    "stress_mse_sum": 10,
    "stress_mae_sum": 11,
}


def apply_fidelity_embedding(
    atom_features: torch.Tensor,
    batch: torch.Tensor,
    fidelity_ids: torch.Tensor | None,
    fidelity_embedding: nn.Embedding | None,
) -> torch.Tensor:
    """Add graph-level fidelity embeddings to per-atom features."""
    if fidelity_embedding is None:
        return atom_features
    if fidelity_ids is None:
        raise ValueError("Model was configured with num_fidelity_levels but fidelity_ids was not provided")
    if batch.dim() != 1:
        raise ValueError(f"batch must be 1D, got shape {tuple(batch.shape)}")
    if atom_features.shape[0] != batch.shape[0]:
        raise ValueError(
            f"atom_features first dim {atom_features.shape[0]} must match batch size {batch.shape[0]}"
        )
    fidelity_ids = fidelity_ids.to(device=atom_features.device, dtype=torch.long).view(-1)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    if fidelity_ids.shape[0] != num_graphs:
        raise ValueError(
            f"fidelity_ids length {fidelity_ids.shape[0]} must match num_graphs {num_graphs}"
        )
    if fidelity_ids.numel():
        min_id = int(fidelity_ids.min().item())
        max_id = int(fidelity_ids.max().item())
        if min_id < 0 or max_id >= fidelity_embedding.num_embeddings:
            raise ValueError(
                f"fidelity_ids must be in [0, {fidelity_embedding.num_embeddings - 1}], got [{min_id}, {max_id}]"
            )
    return atom_features + fidelity_embedding(fidelity_ids)[batch]


def parse_fidelity_loss_weights(spec: str | None) -> dict[int, float]:
    """Parse CLI spec like ``0:1.0,1:3.0`` into fidelity->weight mapping."""
    if not spec:
        return {}
    parsed: dict[int, float] = {}
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --fidelity-loss-weights part {item!r}; expected fidelity_id:weight")
        fid_str, weight_str = item.split(":", 1)
        try:
            fid = int(fid_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid fidelity id in --fidelity-loss-weights part {item!r}; expected integer fidelity_id"
            ) from exc
        try:
            weight = float(weight_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight in --fidelity-loss-weights part {item!r}; expected float weight"
            ) from exc
        if fid < 0:
            raise ValueError(f"Fidelity id must be >= 0 in --fidelity-loss-weights, got {fid}")
        if weight < 0:
            raise ValueError(f"Fidelity loss weight must be >= 0, got {weight} for fidelity id {fid}")
        parsed[fid] = weight
    return parsed


def get_graph_fidelity_weights(
    fidelity_ids: torch.Tensor | None,
    fidelity_loss_weights: dict[int, float] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Map graph-level fidelity ids to graph-level scalar weights."""
    if fidelity_ids is None or not fidelity_loss_weights:
        return None
    fidelity_ids = fidelity_ids.to(device=device, dtype=torch.long).view(-1)
    weights = torch.ones_like(fidelity_ids, dtype=dtype, device=device)
    for fid, weight in fidelity_loss_weights.items():
        weights = torch.where(fidelity_ids == int(fid), torch.full_like(weights, float(weight)), weights)
    return weights


def smooth_l1_loss_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 0.5,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted SmoothL1 mean, summed loss, and summed normalizer."""
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    if weights is None:
        loss_sum = loss.sum()
        normalizer = torch.tensor(float(loss.numel()), device=loss.device, dtype=loss.dtype)
        return loss_sum / normalizer.clamp_min(1e-12), loss_sum, normalizer
    weight_tensor = weights.to(device=loss.device, dtype=loss.dtype)
    while weight_tensor.dim() < loss.dim():
        weight_tensor = weight_tensor.unsqueeze(-1)
    expanded_weights = torch.ones_like(loss) * weight_tensor
    loss_sum = (loss * expanded_weights).sum()
    normalizer = expanded_weights.sum()
    return loss_sum / normalizer.clamp_min(1e-12), loss_sum, normalizer


def init_per_fidelity_metric_sums(
    num_fidelity_levels: int,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Allocate a compact tensor accumulator for per-fidelity metrics."""
    if num_fidelity_levels <= 0:
        return None
    return torch.zeros(
        (int(num_fidelity_levels), len(_PER_FIDELITY_STAT_KEYS)),
        dtype=torch.float64,
        device=device,
    )


def update_per_fidelity_metric_sums(
    stats: torch.Tensor | None,
    *,
    graph_fidelity_ids: torch.Tensor | None,
    batch_idx: torch.Tensor,
    energy_preds: torch.Tensor,
    energy_targets: torch.Tensor,
    energy_avg_preds: torch.Tensor,
    energy_avg_targets: torch.Tensor,
    force_preds: torch.Tensor,
    force_targets: torch.Tensor,
    stress_preds: torch.Tensor | None = None,
    stress_targets: torch.Tensor | None = None,
) -> None:
    """Accumulate per-fidelity regression metric numerators and counts."""
    if stats is None or graph_fidelity_ids is None or graph_fidelity_ids.numel() == 0:
        return
    graph_fidelity_ids = graph_fidelity_ids.to(device=stats.device, dtype=torch.long).view(-1)
    atom_fidelity_ids = graph_fidelity_ids[batch_idx.to(device=stats.device, dtype=torch.long)]
    energy_preds = energy_preds.to(device=stats.device, dtype=torch.float64)
    energy_targets = energy_targets.to(device=stats.device, dtype=torch.float64)
    energy_avg_preds = energy_avg_preds.to(device=stats.device, dtype=torch.float64)
    energy_avg_targets = energy_avg_targets.to(device=stats.device, dtype=torch.float64)
    force_preds = force_preds.to(device=stats.device, dtype=torch.float64)
    force_targets = force_targets.to(device=stats.device, dtype=torch.float64)
    stress_preds = None if stress_preds is None else stress_preds.to(device=stats.device, dtype=torch.float64)
    stress_targets = None if stress_targets is None else stress_targets.to(device=stats.device, dtype=torch.float64)

    for fid_tensor in torch.unique(graph_fidelity_ids):
        fid = int(fid_tensor.item())
        if fid < 0 or fid >= stats.shape[0]:
            continue
        graph_mask = graph_fidelity_ids == fid
        atom_mask = atom_fidelity_ids == fid

        stats[fid, _PER_FIDELITY_STAT_KEYS["graph_count"]] += float(graph_mask.sum().item())
        stats[fid, _PER_FIDELITY_STAT_KEYS["atom_count"]] += float(atom_mask.sum().item())

        e_diff = (energy_preds[graph_mask] - energy_targets[graph_mask]).view(-1)
        e_avg_diff = (energy_avg_preds[graph_mask] - energy_avg_targets[graph_mask]).view(-1)
        f_diff = (force_preds[atom_mask] - force_targets[atom_mask]).view(-1)

        stats[fid, _PER_FIDELITY_STAT_KEYS["energy_mse_sum"]] += (e_diff.square()).sum()
        stats[fid, _PER_FIDELITY_STAT_KEYS["energy_mae_sum"]] += e_diff.abs().sum()
        stats[fid, _PER_FIDELITY_STAT_KEYS["energy_avg_mse_sum"]] += (e_avg_diff.square()).sum()
        stats[fid, _PER_FIDELITY_STAT_KEYS["energy_avg_mae_sum"]] += e_avg_diff.abs().sum()
        stats[fid, _PER_FIDELITY_STAT_KEYS["force_mse_sum"]] += (f_diff.square()).sum()
        stats[fid, _PER_FIDELITY_STAT_KEYS["force_mae_sum"]] += f_diff.abs().sum()
        stats[fid, _PER_FIDELITY_STAT_KEYS["force_elem_count"]] += float(f_diff.numel())

        if stress_preds is not None and stress_targets is not None and stress_preds.numel() and stress_targets.numel():
            s_diff = (stress_preds[graph_mask] - stress_targets[graph_mask]).view(-1)
            stats[fid, _PER_FIDELITY_STAT_KEYS["stress_mse_sum"]] += (s_diff.square()).sum()
            stats[fid, _PER_FIDELITY_STAT_KEYS["stress_mae_sum"]] += s_diff.abs().sum()
            stats[fid, _PER_FIDELITY_STAT_KEYS["stress_elem_count"]] += float(s_diff.numel())


def finalize_per_fidelity_metric_sums(
    stats: torch.Tensor | None,
    *,
    restore_energy,
    restore_force,
) -> dict[int, dict[str, float]]:
    """Convert accumulated per-fidelity sums into RMSE/MAE metrics."""
    if stats is None:
        return {}
    stats_cpu = stats.detach().cpu()
    result: dict[int, dict[str, float]] = {}
    for fid in range(stats_cpu.shape[0]):
        graph_count = float(stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["graph_count"]].item())
        if graph_count <= 0:
            continue
        atom_count = float(stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["atom_count"]].item())
        force_elem_count = float(stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["force_elem_count"]].item())
        stress_elem_count = float(stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["stress_elem_count"]].item())

        energy_rmse = restore_energy((stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["energy_mse_sum"]].item() / graph_count) ** 0.5)
        energy_mae = restore_energy(stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["energy_mae_sum"]].item() / graph_count)
        energy_rmse_avg = restore_energy(
            (stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["energy_avg_mse_sum"]].item() / graph_count) ** 0.5
        )
        energy_mae_avg = restore_energy(
            stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["energy_avg_mae_sum"]].item() / graph_count
        )

        metrics = {
            "num_graphs": graph_count,
            "num_atoms": atom_count,
            "energy_rmse": energy_rmse,
            "energy_mae": energy_mae,
            "energy_rmse_avg": energy_rmse_avg,
            "energy_mae_avg": energy_mae_avg,
        }

        if force_elem_count > 0:
            metrics["force_rmse"] = restore_force(
                (stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["force_mse_sum"]].item() / force_elem_count) ** 0.5
            )
            metrics["force_mae"] = restore_force(
                stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["force_mae_sum"]].item() / force_elem_count
            )
        if stress_elem_count > 0:
            metrics["stress_rmse"] = (stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["stress_mse_sum"]].item() / stress_elem_count) ** 0.5
            metrics["stress_mae"] = stats_cpu[fid, _PER_FIDELITY_STAT_KEYS["stress_mae_sum"]].item() / stress_elem_count
        result[fid] = metrics
    return result


def flatten_per_fidelity_metrics(
    per_fidelity_metrics: dict[int, dict[str, float]],
    *,
    prefix: str,
) -> dict[str, float]:
    """Flatten nested per-fidelity metrics into CSV-friendly scalar keys."""
    flat: dict[str, float] = {}
    for fid, metrics in sorted(per_fidelity_metrics.items()):
        for key, value in metrics.items():
            flat[f"{prefix}_{key}_fid_{fid}"] = float(value)
    return flat


def zero_init_module_output(module: nn.Module) -> None:
    """Zero-initialize a simple MLP-style output layer when present."""
    output = getattr(module, "output", None)
    if isinstance(output, nn.Linear):
        with torch.no_grad():
            output.weight.zero_()
            if output.bias is not None:
                output.bias.zero_()


def apply_delta_energy_heads(
    base_atom_energies: torch.Tensor,
    invariant_features: torch.Tensor,
    batch: torch.Tensor,
    fidelity_ids: torch.Tensor | None,
    delta_proj_heads: nn.ModuleDict | None,
    delta_sum_heads: nn.ModuleDict | None,
) -> torch.Tensor:
    """Apply per-fidelity residual energy heads on top of baseline atom energies."""
    if delta_proj_heads is None or delta_sum_heads is None or len(delta_proj_heads) == 0:
        return base_atom_energies
    if fidelity_ids is None:
        return base_atom_energies
    graph_ids = fidelity_ids.to(device=base_atom_energies.device, dtype=torch.long).view(-1)
    if graph_ids.numel() == 0:
        return base_atom_energies
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    if graph_ids.shape[0] != num_graphs:
        raise ValueError(f"fidelity_ids length {graph_ids.shape[0]} must match num_graphs {num_graphs}")
    atom_ids = graph_ids[batch]
    out = base_atom_energies
    for fid_str, proj_head in delta_proj_heads.items():
        delta_proj = proj_head(invariant_features)
        delta_atom = delta_sum_heads[fid_str](delta_proj).sum(dim=-1, keepdim=True)
        fid_mask = (atom_ids == int(fid_str)).to(dtype=delta_atom.dtype).view(-1, 1)
        out = out + delta_atom * fid_mask
    return out
