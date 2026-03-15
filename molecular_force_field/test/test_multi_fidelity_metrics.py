import math

import torch

from molecular_force_field.utils.fidelity import (
    finalize_per_fidelity_metric_sums,
    flatten_per_fidelity_metrics,
    get_graph_fidelity_weights,
    init_per_fidelity_metric_sums,
    parse_fidelity_loss_weights,
    smooth_l1_loss_stats,
    update_per_fidelity_metric_sums,
)


def test_parse_fidelity_loss_weights():
    assert parse_fidelity_loss_weights(None) == {}
    assert parse_fidelity_loss_weights("0:1.0, 1:3.5") == {0: 1.0, 1: 3.5}


def test_weighted_smooth_l1_loss_respects_fidelity_weights():
    fidelity_ids = torch.tensor([0, 1], dtype=torch.long)
    graph_weights = get_graph_fidelity_weights(
        fidelity_ids,
        {1: 4.0},
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    pred = torch.tensor([0.0, 4.0], dtype=torch.float64)
    target = torch.zeros_like(pred)
    unweighted_mean, _, _ = smooth_l1_loss_stats(pred, target, beta=0.5, weights=None)
    weighted_mean, _, _ = smooth_l1_loss_stats(pred, target, beta=0.5, weights=graph_weights)
    assert weighted_mean.item() > unweighted_mean.item()


def test_per_fidelity_metric_sums_separate_metrics():
    stats = init_per_fidelity_metric_sums(2, device=torch.device("cpu"))
    graph_fidelity_ids = torch.tensor([0, 1], dtype=torch.long)
    batch_idx = torch.tensor([0, 0, 1], dtype=torch.long)

    energy_preds = torch.tensor([1.0, 2.0], dtype=torch.float64)
    energy_targets = torch.tensor([0.0, 0.0], dtype=torch.float64)
    energy_avg_preds = torch.tensor([0.5, 1.0], dtype=torch.float64)
    energy_avg_targets = torch.tensor([0.0, 0.0], dtype=torch.float64)
    force_preds = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    force_targets = torch.zeros_like(force_preds)

    update_per_fidelity_metric_sums(
        stats,
        graph_fidelity_ids=graph_fidelity_ids,
        batch_idx=batch_idx,
        energy_preds=energy_preds,
        energy_targets=energy_targets,
        energy_avg_preds=energy_avg_preds,
        energy_avg_targets=energy_avg_targets,
        force_preds=force_preds,
        force_targets=force_targets,
    )

    metrics = finalize_per_fidelity_metric_sums(
        stats,
        restore_energy=lambda x: float(x),
        restore_force=lambda x: float(x),
    )
    assert set(metrics) == {0, 1}
    assert math.isclose(metrics[0]["energy_rmse"], 1.0, rel_tol=1e-8)
    assert math.isclose(metrics[1]["energy_rmse"], 2.0, rel_tol=1e-8)
    assert math.isclose(metrics[0]["energy_rmse_avg"], 0.5, rel_tol=1e-8)
    assert math.isclose(metrics[1]["energy_rmse_avg"], 1.0, rel_tol=1e-8)
    assert math.isclose(metrics[0]["force_rmse"], math.sqrt(2.0 / 6.0), rel_tol=1e-8)
    assert math.isclose(metrics[1]["force_rmse"], math.sqrt(4.0 / 3.0), rel_tol=1e-8)

    flat = flatten_per_fidelity_metrics(metrics, prefix="val")
    assert "val_energy_rmse_fid_0" in flat
    assert "val_force_rmse_fid_1" in flat
