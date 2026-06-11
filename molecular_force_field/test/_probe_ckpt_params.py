import sys
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix.pth"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
am = ckpt.get("model_hyperparameters", {}) or {}
print(f"=== {CKPT} ===")
# the ictd-fix knobs + avg_num_neighbors that were previously NOT saved (root-fix target)
keys = ["avg_num_neighbors", "ictd_fix_route", "ictd_fix_product_backend", "ictd_fix_fusion_heads",
        "ictd_fix_fusion_head_weight_mode", "ictd_fix_interaction_attn_heads", "ictd_fix_interaction_scale",
        "ictd_fix_fusion_scale_init", "ictd_fix_gmix_gate_init", "ictd_fix_gmix_output_lmax"]
present = 0
for k in keys:
    if k in ckpt:
        v, where = ckpt[k], "ckpt"
    elif k in am:
        v, where = am[k], "arch_meta"
    else:
        v, where = "<ABSENT>", "-"
    if where != "-":
        present += 1
    print(f"  {k:38s} {str(v):>14}   [{where}]")
print(f"  -> {present}/{len(keys)} present")
