"""One-time repair: inject the silently-missing arch params (avg_num_neighbors + ictd_fix_*)
into the OLD checkpoint's model_hyperparameters so it self-describes like a fixed-trainer ckpt.
Values come from reconstructing the model via from_checkpoint (shape-inferred + avg override),
then reading the model attrs. Writes a *_repaired.pth and verifies a clean re-import."""
import torch
from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

OLD = "/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix.pth"
OUT = "/home/ylzhang/lrx/lr_ablation/checkpoint/model_pure_cartesian_ictd_fix_repaired.pth"
AVG = 82.5075188429
ELEMS = ["H", "O", "F", "K"]
ATTRS = ["avg_num_neighbors", "ictd_fix_route", "ictd_fix_product_backend", "ictd_fix_fusion_heads",
         "ictd_fix_fusion_head_weight_mode", "ictd_fix_interaction_attn_heads", "ictd_fix_interaction_scale",
         "ictd_fix_fusion_scale_init", "ictd_fix_gmix_gate_init", "ictd_fix_gmix_output_lmax"]

obj = LAMMPS_MLIAP_MFF.from_checkpoint(checkpoint_path=OLD, element_types=ELEMS, device="cpu", avg_num_neighbors=AVG)
m = obj.wrapper.model
ckpt = torch.load(OLD, map_location="cpu", weights_only=False)
am = ckpt.get("model_hyperparameters") or {}
added = {}
for a in ATTRS:
    v = getattr(m, a, None)
    if v is not None:
        am[a] = v
        added[a] = v
ckpt["model_hyperparameters"] = am
torch.save(ckpt, OUT)
print("repaired ->", OUT)
print("injected:", added)

# verify: re-import the repaired ckpt WITHOUT the override -> avg read from ckpt, no warning
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    obj2 = LAMMPS_MLIAP_MFF.from_checkpoint(checkpoint_path=OUT, element_types=ELEMS, device="cpu")
    avgwarn = [str(x.message) for x in w if "avg_num_neighbors" in str(x.message)]
print("reimport avg_num_neighbors =", obj2.wrapper.model.avg_num_neighbors)
print("avg fallback warning fired? ", bool(avgwarn))
