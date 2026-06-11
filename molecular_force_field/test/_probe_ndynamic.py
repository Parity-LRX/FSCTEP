"""Probe the workload for approach 3 (N-dynamic .pt2): make_fx-flatten the force graph, then ask
torch.export to make the atom-count N a dynamic Dim. The ConstraintViolationError (if any) names
the ops/locations that pin N -> that list IS the workload. Uses a small synthetic model (same arch;
specialization is code/shape-driven, not weights)."""
import os
import sys
import traceback
import torch
from torch.export import Dim

os.environ.setdefault("OMP_NUM_THREADS", "1")
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(_repo) not in sys.path:
    sys.path.insert(0, os.path.dirname(_repo))

from molecular_force_field.test.bench_ictd_fix_trainstep import build_model, make_fixed_graph
from molecular_force_field.training.makefx_compile import trace_and_compile_force
from molecular_force_field.test.bench_aoti_export import force_compute_fn_factory

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float32
torch.backends.cuda.matmul.allow_tf32 = False
model = build_model(channels=64, lmax=2, num_interaction=2, route="fusion",
                    product_backend="ictd-pure-u", dtype=dt, device=dev, correlation=2, attn_heads=1)
model.eval()
model.skip_input_validation = True
graph = make_fixed_graph(num_nodes=48, avg_degree=10, dtype=dt, device=dev)
ex = (graph[0],) + tuple(graph[1:])

gm = trace_and_compile_force(model, ex, training=False,
                             compute_fn=force_compute_fn_factory(model, training=False), do_compile=False)
print("make_fx flatten OK; now torch.export with N (and E) dynamic...")

Ndim = Dim("n_atoms", min=2)
Edim = Dim("n_edges", min=2)
# (pos[N,3], A[N], batch[N], edge_src[E], edge_dst[E], edge_shifts[E,3], cell[M,3,3])
dyn = ({0: Ndim}, {0: Ndim}, {0: Ndim}, {0: Edim}, {0: Edim}, {0: Edim}, None)
try:
    ep = torch.export.export(gm, tuple(ex), dynamic_shapes=dyn)
    print("\n=== N-DYNAMIC EXPORT OK (no op pins N!) ===")
except Exception as e:
    print("\n=== N-dynamic export FAILED:", type(e).__name__, "===")
    msg = str(e)
    print(msg[:6000])
    # count distinct source locations the error blames (rough workload proxy)
    import re
    locs = sorted(set(re.findall(r'[\w/]+\.py:\d+', msg)))
    print("\n--- distinct source locations named:", len(locs), "---")
    for l in locs[:40]:
        print("  ", l)
