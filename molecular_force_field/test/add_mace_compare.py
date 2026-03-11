import re

with open('molecular_force_field/benchmark_max_atoms_1s.py', 'r') as f:
    text = f.read()

# Modify measure_inference_ms to accept is_mace
text = text.replace("def measure_inference_ms(layer, graph, warmup: int, repeat: int, device):", 
"def measure_inference_ms(layer, graph, warmup: int, repeat: int, device, is_mace=False):\n    if is_mace:\n        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph\n        num_nodes = pos.shape[0]\n        import e3nn.o3 as o3\n        mace_data = {\n            'positions': pos,\n            'node_attrs': torch.ones(num_nodes, 1, device=device, dtype=pos.dtype),\n            'edge_index': torch.vstack([edge_src, edge_dst]),\n            'shifts': edge_shifts,\n            'unit_shifts': edge_shifts,\n            'cell': cell,\n            'batch': batch,\n            'ptr': torch.tensor([0, num_nodes], device=device, dtype=torch.long)\n        }\n        graph = (mace_data,)\n")

# Add arg
text = text.replace('parser.add_argument("--compile", action="store_true"', 
'parser.add_argument("--mace", action="store_true", help="Compare with MACE model")\n    parser.add_argument("--compile", action="store_true"')

# In main: Replace layer instantiation and printing
old_init = "layer = PureCartesianICTDTransformerLayer(**cfg).to(device=device, dtype=dtype)"
new_init = """if args.mace:
        import numpy as np
        import mace.modules
        import e3nn.o3 as o3
        layer = mace.modules.MACE(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=args.lmax,
            interaction_cls=mace.modules.interaction_classes['RealAgnosticResidualInteractionBlock'],
            interaction_cls_first=mace.modules.interaction_classes['RealAgnosticInteractionBlock'],
            num_interactions=args.num_interaction,
            num_elements=1,
            hidden_irreps=o3.Irreps(f'{args.channels}x0e + {args.channels}x1o + {args.channels}x2e'),
            MLP_irreps=o3.Irreps('16x0e'),
            atomic_energies=np.zeros(1),
            avg_num_neighbors=args.avg_degree,
            atomic_numbers=[1],
            correlation=args.num_interaction,
            gate=torch.nn.functional.silu,
        ).to(device=device, dtype=dtype)
        print("Initialized MACE model.")
    else:
        layer = PureCartesianICTDTransformerLayer(**cfg).to(device=device, dtype=dtype)"""

text = text.replace(old_init, new_init)

# Fix measure_inference_ms calls to pass is_mace
text = text.replace("warmup=5, repeat=3, device=device", "warmup=5, repeat=3, device=device, is_mace=args.mace")
text = text.replace("warmup=args.warmup, repeat=args.repeat, device=device", "warmup=args.warmup, repeat=args.repeat, device=device, is_mace=args.mace")

with open('molecular_force_field/benchmark_max_atoms_1s.py', 'w') as f:
    f.write(text)

