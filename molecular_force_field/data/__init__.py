"""Data handling modules including datasets, collate functions, and preprocessing."""

try:
    from molecular_force_field.data.datasets import (
        CustomDataset,
        H5Dataset,
        OnTheFlyDataset,
        compute_graph_worker,
    )
    
    __all__ = [
        "CustomDataset",
        "H5Dataset",
        "OnTheFlyDataset",
        "compute_graph_worker",
    ]
except ImportError:
    __all__ = []