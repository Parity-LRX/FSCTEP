"""Model definitions for the molecular force field library."""

from molecular_force_field.models.e3nn_layers import (
    E3Conv,
    E3Conv2,
    E3_TransformerLayer_multi,
)
from molecular_force_field.models.mlp import MainNet, MainNet2
from molecular_force_field.models.losses import RMSELoss
from molecular_force_field.models.cartesian_e3_layers import (
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    CartesianFullyConnectedTensorProduct,
    CartesianFullTensorProduct,
    EquivariantTensorProduct,
)

# Backward-compatible conv symbol exports:
# Some versions of this repo used names without the "Sparse" infix.
try:
    from molecular_force_field.models.cartesian_e3_layers import (
        CartesianE3Conv,
        CartesianE3Conv2,
        CartesianE3ConvLoose,
        CartesianE3Conv2Loose,
    )
except ImportError:  # pragma: no cover
    from molecular_force_field.models.cartesian_e3_layers import (  # type: ignore
        CartesianE3ConvSparse as CartesianE3Conv,
        CartesianE3Conv2Sparse as CartesianE3Conv2,
        CartesianE3ConvSparseLoose as CartesianE3ConvLoose,
        CartesianE3Conv2SparseLoose as CartesianE3Conv2Loose,
    )
from molecular_force_field.models.pure_cartesian_layers import (
    PureCartesianTransformerLayer,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PureCartesianICTDTransformerLayer,
)
from molecular_force_field.models.pure_cartesian_sparse_layers import (
    PureCartesianSparseTransformerLayer,
)
from molecular_force_field.models.ictd_fast import (
    FastSymmetricSTF,
    FastSymmetricTraceChain,
    decompose_rank2_generic,
)

__all__ = [
    "E3Conv",
    "E3Conv2",
    "E3_TransformerLayer_multi",
    "CartesianTransformerLayer",
    "CartesianTransformerLayerLoose",
    "CartesianE3Conv",
    "CartesianE3Conv2",
    "CartesianE3ConvLoose",
    "CartesianE3Conv2Loose",
    "CartesianFullyConnectedTensorProduct",
    "CartesianFullTensorProduct",
    "EquivariantTensorProduct",
    "PureCartesianTransformerLayer",
    "PureCartesianICTDTransformerLayer",
    "PureCartesianSparseTransformerLayer",
    "FastSymmetricSTF",
    "FastSymmetricTraceChain",
    "decompose_rank2_generic",
    "MainNet",
    "MainNet2",
    "RMSELoss",
]