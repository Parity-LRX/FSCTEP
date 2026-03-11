"""
Molecular Force Field - A Python library for molecular modeling with E3NN-based neural networks.
"""

__version__ = "0.1.0"

import torch
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])

# Import main classes and functions (lazy import to avoid circular dependencies)
try:
    from molecular_force_field.models import (
        E3Conv,
        E3Conv2,
        E3_TransformerLayer_multi,
        MainNet,
        MainNet2,
        RMSELoss,
        RobustScalarWeightedSum,
    )

    from molecular_force_field.data import (
        CustomDataset,
        H5Dataset,
        OnTheFlyDataset,
    )
    
    __all__ = [
        "E3Conv",
        "E3Conv2",
        "E3_TransformerLayer_multi",
        "MainNet",
        "MainNet2",
        "RMSELoss",
        "RobustScalarWeightedSum",
        "CustomDataset",
        "H5Dataset",
        "OnTheFlyDataset",
    ]
except ImportError:
    # If dependencies are not installed, just define version
    __all__ = []