"""make_fx-compile pipeline for FSCETP force-loss training (second-order backward).

Force-loss training minimizes ``L = a*||E - E*||^2 + b*||f - f*||^2`` with
``f = -dE/dx``; the optimizer step needs ``dL/dtheta``, which contains
``d(f)/dtheta = -d^2E/(dx dtheta)`` -- a full second derivative.  ``torch.compile``'s
AOTAutograd captures forward + first backward but cannot double-differentiate an
``autograd.grad(create_graph=True)`` hidden *inside* the compiled region, so a
direct ``torch.compile`` of the train step fails.

The make_fx route (reverse-engineered from DeepMD/SeZM; see
``HANDOFF_makefx_compile.md`` sections 3 and 8) sidesteps this:

1. ``make_fx`` traces ``forward + inner force-autograd`` *after* the first
   derivative has been materialised, producing one flat FX graph in which
   ``dE/dx`` is already a sequence of ordinary ops -- no hidden autograd call.
2. ``torch.compile(backend="inductor", dynamic=True)`` lowers that flat graph;
   because it no longer hides an autograd call, Inductor's ordinary backward
   differentiates the whole thing a second time for the optimizer step.

Everything else here exists to make that composition correct: stripping the
detach chains autograd inserts on saved activations (else the force-loss
gradient to theta is silently severed), rebuilding the FX graph to flush stale
node pointers, decomposing ops that lack a symbolic higher-order derivative
(SiLU's backward), and pinning the Inductor/Triton flags that have interacted
badly with make_fx + double-backward under dynamic shapes.

FSCETP differs from DeepMD in one structural way that *simplifies* the port:
FSCETP uses a flat node/edge representation (N nodes, E edges) with no
``nframes`` axis, so the symbolic dims are just N and E.  DeepMD's ``nf=5``
trace trick (repeating the batch to dodge collisions with reserved dims
1/2/3/9) is therefore unnecessary -- we trace with a real small graph whose N
and E avoid the reserved concrete dims (3, 9).

All behaviour here is opt-in; importing this module only sets the global
``optimize_ddp = False`` (safe: our compile boundary is always inner).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import torch
from torch.fx.experimental.proxy_tensor import make_fx

log = logging.getLogger(__name__)

# DDPOptimizer splits a DDP-wrapped model's graph at bucket boundaries; our
# compile region is *inner* (it wraps only the energy+force compute), and the
# split produces subgraphs whose outputs include symbolic ints, crashing
# AOTAutograd with "'int' object has no attribute 'meta'" (pytorch#134182).
# Disabling it globally is safe because we always own our own compile boundary.
import torch._dynamo.config as _dynamo_cfg

_dynamo_cfg.optimize_ddp = False


# Conservative Inductor/Triton options.  Each flag is pinned to a dynamic-shape
# bug seen in the DeepMD/SeZM make_fx work (HANDOFF section 3.3 / NOTE 6):
#   max_autotune=False           autotune rolls the search on every dynamic-shape
#                                recompile -> slower; deterministic kernels win.
#   shape_padding=True           pad symbolic shapes to SIMD-friendly sizes,
#                                killing tail-kernel codegen cost.
#   epilogue_fusion=False        inactive anyway with autotune off; and fused
#                                epilogues reorder saved tensors so the second
#                                backward can't recover them.
#   triton.cudagraphs=False      cudagraphs capture autograd metadata once;
#                                higher-order grads need fresh metadata per call.
#                                (=> make_fx-compile and inductor cudagraph are
#                                mutually exclusive; the manual whole-step
#                                CUDA-graph route is the other branch.)
#   max_fusion_size=8            cap fusion complexity so Triton IR gen doesn't
#                                time out on big edge-level reductions.
#   triton.persistent_reductions=False   avoid PassManager::run failed in
#                                make_ttgir on the dynamic edge graph.
#   triton.mix_order_reduction=False     workaround pytorch#174379/#178080/#179494
#                                (data-dependent symbolic shapes = our edge count).
_INDUCTOR_OPTS: dict[str, Any] = {
    "max_autotune": False,
    "shape_padding": True,
    "epilogue_fusion": False,
    "triton.cudagraphs": False,
    "max_fusion_size": 8,
    "triton.persistent_reductions": False,
    "triton.mix_order_reduction": False,
}


def _filter_inductor_opts(opts: dict[str, Any]) -> dict[str, Any]:
    """Drop option keys the running Inductor build does not recognise.

    The 7-flag set above was validated on torch 2.11; older/newer builds (e.g.
    the FSCETP 2.7.1 env) may not expose every key.  Filtering against the live
    config registry lets the same curated set run unchanged across versions
    instead of erroring on an unknown option.
    """
    try:
        from torch._inductor import config as inductor_config

        valid = inductor_config.get_config_copy()
        return {k: v for k, v in opts.items() if k.replace("-", "_") in valid}
    except Exception:
        # Registry not exposed on some builds; keep the curated set and let
        # torch.compile surface any genuine backend error.
        return dict(opts)


def _strip_saved_tensor_detach(gm: torch.fx.GraphModule) -> None:
    """Strip the ``aten.detach`` chains make_fx inserts for saved tensors.

    With ``create_graph=True``, the autograd engine wraps every saved forward
    activation in a double-detach chain (e.g. ``silu -> detach_A -> detach_B ->
    silu_backward``).  In eager autograd those detaches are informational; after
    tracing they become real ops that sever the gradient path from the force
    loss back to theta, so training silently emits zero parameter updates for
    the second-derivative term.

    The three categories are distinguished by graph topology alone -- no op-name
    matching -- so user-explicit ``.detach()`` calls (e.g. cached SO3 weights)
    survive:
      * chain inner: input is another detach;
      * dead node:   no downstream users;
      * chain head:  every user is a detach.
    Anything matching none of the three is treated as user intent and kept.

    Caller guards this with ``training`` -- eval never sets ``create_graph=True``
    so the chain is never inserted and removing detaches would be incorrect.
    """
    _DETACH = torch.ops.aten.detach.default

    def _is_detach(n: torch.fx.Node) -> bool:
        return n.op == "call_function" and n.target == _DETACH

    # Pass 1: classify every detach against the *original* graph.  Erasing
    # eagerly would let later classifications walk a mutated neighbourhood and
    # misjudge the boundaries (the double-detach pattern flips class within a
    # single erase).  Collect first, mutate second.
    to_remove: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if not _is_detach(node):
            continue
        input_node = node.args[0]
        users = list(node.users.keys())
        is_chain_inner = _is_detach(input_node)
        is_dead = len(users) == 0
        is_chain_head = len(users) > 0 and all(_is_detach(u) for u in users)
        if is_chain_inner or is_dead or is_chain_head:
            to_remove.append(node)

    # Pass 2: rewire + erase atomically after the full classification.
    for node in to_remove:
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()


def _rebuild_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Return a fresh GraphModule with a newly allocated node linked-list.

    ``Graph.erase_node`` (used by the detach strip) can leave stale C-level
    prev/next pointers on neighbouring Node objects on some torch builds
    (observed on 2.11+cu130).  When torch.compile later re-traces and walks
    ``graph.nodes`` to read ``nd.meta``, dereferencing a stale pointer
    segfaults.  A single ``node_copy`` pass into a clean Graph side-steps it
    entirely.  Always rebuilt -- a fresh graph is cheap, a segfault is fatal.
    """
    old_graph = gm.graph
    new_graph = torch.fx.Graph()
    val_map: dict[torch.fx.Node, torch.fx.Node] = {}
    for node in old_graph.nodes:
        val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    new_graph.lint()
    return torch.fx.GraphModule(gm, new_graph)


def _default_decompositions():
    """Ops without a symbolic higher-order derivative that Inductor would refuse
    to differentiate a second time.

    ``silu_backward`` is the one DeepMD hit, and FSCETP uses ``nn.SiLU``
    throughout, so it applies directly: lowering it to ``sigmoid + pointwise
    mul`` gives every piece a well-defined higher derivative.  If a future
    double-backward trace raises "no higher-order derivative / not
    differentiable" for another op, add its ``aten.*_backward`` here.
    """
    from torch._decomp import get_decompositions

    return get_decompositions([torch.ops.aten.silu_backward.default])


def make_force_compute_fn(model: torch.nn.Module, *, training: bool) -> Callable:
    """Wrap ``model.forward`` as a pure tensor function returning ``(energy, forces)``.

    The coordinate is rebound to a fresh leaf via ``detach().requires_grad_(True)``
    so the inner ``autograd.grad`` differentiates against a graph of known shape
    and ownership -- the precondition for make_fx symbolic tracing to capture
    ``dE/dx`` as ordinary FX nodes.  ``create_graph=training`` is the single
    toggle that keeps the first-derivative graph alive so the outer optimizer's
    ``.backward()`` can walk it into the parameters.
    """

    def compute_fn(pos: torch.Tensor, *rest: torch.Tensor):
        p = pos.detach().requires_grad_(True)
        out = model(p, *rest)
        e_atom = out[0] if isinstance(out, tuple) else out
        energy = e_atom.sum()
        grad = torch.autograd.grad(energy, p, create_graph=training)[0]
        return energy, -grad

    return compute_fn


def trace_and_compile_force(
    model: torch.nn.Module,
    example_inputs: Sequence[torch.Tensor],
    *,
    training: bool,
    compute_fn: Callable | None = None,
    decompositions=None,
    inductor_options: dict[str, Any] | None = None,
    do_compile: bool = True,
) -> Callable:
    """make_fx-trace the forward + force-autograd, flatten it, and compile.

    ``example_inputs`` is a real small graph ``(pos, *rest)`` matching the model
    signature; ``pos`` need not require grad (``compute_fn`` rebinds it).  With
    ``dynamic=True`` the single compiled product accepts other N/E at runtime.

    ``compute_fn`` is the pure tensor function to trace; when ``None`` it defaults
    to :func:`make_force_compute_fn` (returns ``(energy_sum, forces)``).  The
    trainer injects its own (returning ``(E_per_atom, dE/dx)``) so the compiled
    output drops straight into the eager loss code; whatever the factory builds,
    it must rebind the coordinate to a fresh leaf and call ``autograd.grad`` with
    ``create_graph=training`` so the flattened graph carries the 2nd-order chain.

    Returns the compiled callable ``f(pos, *rest)``.  With ``do_compile=False``
    returns the flattened (uncompiled) ``GraphModule`` -- useful for CPU
    correctness checks of the flatten+strip+rebuild stages without Inductor/GPU.
    """
    # Remove host syncs (.item()/torch.any input validation) so the trace contains
    # no data-dependent scalar reads -- otherwise make_fx hits
    # GuardOnDataDependentSymNode on the `A.max().item()` guard and the whole flat
    # graph is rejected.  Set UNCONDITIONALLY: the model reads this with
    # getattr(self, "skip_input_validation", False) and does not define the
    # attribute by default, so a hasattr-guarded set would never fire.  Only
    # disables guards -- numerics are unchanged -- and the choice is baked into the
    # trace, so the attribute is restored right after.
    prev_skip = getattr(model, "skip_input_validation", None)
    model.skip_input_validation = True

    try:
        if compute_fn is None:
            compute_fn = make_force_compute_fn(model, training=training)
        decomp = decompositions if decompositions is not None else _default_decompositions()
        # Warm up lazy device/dtype-keyed caches (CG tensors, projection groups, ICTD
        # U matrices) with one real eager call BEFORE tracing. These caches build on
        # first use; make_fx must trace with them already materialized, otherwise the
        # cache-miss BUILD path -- e.g. ictd_irreps._get_proj_group_list's
        # cg_list[int(p_idx)] list indexing -- runs under the proxy tracer and raises
        # IndexError. The bench harness only worked because it does an eager reference
        # pass on the same model first; the trainer hands us a cold model, so prime it.
        compute_fn(*example_inputs)
        # tracing_mode="symbolic": every shape becomes a sympy symbol so
        # dynamic=True compile works.  _allow_non_fake_inputs=True: feed real
        # tensors -- the edge/index ops resolve their control flow on concrete
        # values once, shapes go symbolic immediately after.
        traced = make_fx(
            compute_fn,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            decomposition_table=decomp,
        )(*example_inputs)

        if training:
            _strip_saved_tensor_detach(traced)
        traced = _rebuild_graph_module(traced)
    finally:
        if prev_skip is None:
            # The model never defined the attribute; drop the temporary we added.
            try:
                delattr(model, "skip_input_validation")
            except Exception:
                model.skip_input_validation = False
        else:
            model.skip_input_validation = prev_skip

    if not do_compile:
        return traced

    opts = _filter_inductor_opts(inductor_options if inductor_options is not None else _INDUCTOR_OPTS)
    return torch.compile(traced, backend="inductor", dynamic=True, options=opts)


class CompiledForceCache:
    """Multi-slot cache of compiled force-compute callables, keyed by input shape.

    Two reasons for the key:
      * ``training`` toggles ``create_graph`` -> changes graph topology.
      * **the leading dim of every input** (atom count N, edge count E, ...): make_fx
        traces with ``tracing_mode="symbolic"`` but in practice BAKES the concrete
        leading dims of this model into the flat graph (the many ``view(*x.shape[:-1],
        C, 2l+1)`` reshapes in the ICTD layers specialize on N), so a graph traced at
        N=701 raises on an N=695 batch. We therefore compile one graph per distinct
        shape signature. Real datasets with a handful of system sizes (e.g. slab =
        {695,701,703}) compile that many times, then hit cache forever. ``max_slots``
        caps it: a dataset with too many distinct sizes would blow GPU memory on
        per-size graphs, so we raise past the budget and the caller falls back to eager.
        (Pair with --pad-edges-to-max to hold E fixed so only N varies.)

    Held in a plain attribute container so the compiled wrappers (which carry
    duplicated flat param views) stay out of parameter discovery; callers wrapping in
    DDP/FSDP should install this outside the ``nn.Module`` tree (``object.__setattr__``).
    """

    def __init__(self, model: torch.nn.Module, *, max_slots: int = 8):
        self._model = model
        self._max_slots = int(max_slots)
        self._cache: dict[tuple, Callable] = {}
        # NOTE: a single-slot ("N-dynamic") cache does NOT help the TRAINING path. e3nn jit_script_fx=False
        # makes the make_fx flatten N-symbolic (which IS what unblocks the AOTI torch.export deploy), but
        # torch.compile's dynamo RE-specializes N when it retraces the flat GraphModule (mark_dynamic ->
        # ConstraintViolationError "size inferred constant", dynamic=True -> silent ~73s recompile per new
        # N). So the per-N recompile lives in dynamo, not here; keep the explicit per-shape slots so the
        # compile count is at least bounded and visible. (Deploy N-dynamic uses torch.export, not
        # torch.compile, which DOES honor explicit Dims.)

    def get(
        self,
        example_inputs: Sequence[torch.Tensor],
        *,
        training: bool,
        compute_fn_factory: Callable | None = None,
        **compile_kwargs: Any,
    ) -> Callable:
        shape_key = tuple(int(t.shape[0]) for t in example_inputs if torch.is_tensor(t))
        key = (bool(training),) + shape_key
        if key not in self._cache:
            if len(self._cache) >= self._max_slots:
                raise RuntimeError(
                    f"makefx_compile: exceeded max_slots={self._max_slots} distinct input-shape "
                    f"signatures (make_fx bakes leading dims into the traced graph, so each distinct "
                    f"atom/edge count needs its own compile + GPU memory). Seen {sorted(self._cache)}, "
                    f"new {key}. Too many sizes to be worthwhile -- falling back to eager. "
                    f"(Use --pad-edges-to-max to hold edge count fixed.)")
            log.info("makefx_compile: tracing+compiling force compute (training=%s, shape_key=%s, slot %d/%d)",
                     training, shape_key, len(self._cache) + 1, self._max_slots)
            cfn = compute_fn_factory(self._model, training=training) if compute_fn_factory is not None else None
            self._cache[key] = trace_and_compile_force(
                self._model, example_inputs, training=training, compute_fn=cfn, **compile_kwargs
            )
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()
