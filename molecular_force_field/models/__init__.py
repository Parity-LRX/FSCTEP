"""Model definitions for the molecular force field library."""

from molecular_force_field.models.e3nn_layers import (
    E3Conv,
    E3Conv2,
    E3_TransformerLayer_multi,
)
from molecular_force_field.models.mlp import MainNet, MainNet2
from molecular_force_field.models.losses import RMSELoss, weighted_mse_loss_stats
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
from molecular_force_field.models.pure_cartesian_ictd_fix import (
    PureCartesianICTDFix,
)
try:
    from molecular_force_field.models.pure_cartesian_ictd_fix_so2 import (
        PureCartesianICTDFixSO2,
    )
except ImportError:  # pragma: no cover
    PureCartesianICTDFixSO2 = None  # type: ignore[assignment]
try:
    from molecular_force_field.models.pure_cartesian_ictd_escn_so2 import (
        PureCartesianICTDESCNSO2,
    )
except ImportError:  # pragma: no cover
    PureCartesianICTDESCNSO2 = None  # type: ignore[assignment]
from molecular_force_field.models.spherical_fix import (
    SphericalFix,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_nocross import (
    PureCartesianICTDTransformerLayerNoCrossFusion,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_noshallowreadout import (
    PureCartesianICTDTransformerLayerNoShallowReadout,
)
try:
    from molecular_force_field.models.pure_cartesian_ictd_layers_full_o3 import (
        PureCartesianICTDO3TransformerLayer,
        PureCartesianICTDFullO3TransformerLayer,
    )
except ImportError:  # pragma: no cover
    PureCartesianICTDO3TransformerLayer = None  # type: ignore[assignment]
    PureCartesianICTDFullO3TransformerLayer = None  # type: ignore[assignment]
try:
    from molecular_force_field.models.pure_cartesian_ictd_layers_o3 import (
        PureCartesianICTDSaveO3TransformerLayer,
    )
except ImportError:  # pragma: no cover
    PureCartesianICTDSaveO3TransformerLayer = None  # type: ignore[assignment]
try:
    from molecular_force_field.models.pure_cartesian_ictd_layers_so2 import (
        PureCartesianICTDSO2TransformerLayer,
    )
except ImportError:  # pragma: no cover
    PureCartesianICTDSO2TransformerLayer = None  # type: ignore[assignment]
try:
    from molecular_force_field.models.pure_cartesian_ictd_layers_o2 import (
        PureCartesianICTDO2TransformerLayer,
    )
except ImportError:  # pragma: no cover
    PureCartesianICTDO2TransformerLayer = None  # type: ignore[assignment]
try:
    from molecular_force_field.models.pure_cartesian_ictd_layers_node_so2 import (
        PureCartesianICTDNodeSO2TransformerLayer,
    )
except ImportError:  # pragma: no cover
    PureCartesianICTDNodeSO2TransformerLayer = None  # type: ignore[assignment]
from molecular_force_field.models.pure_cartesian_sparse_layers import (
    PureCartesianSparseTransformerLayer,
)
from molecular_force_field.models.pure_cartesian_sparse_layers_save import (
    PureCartesianSparseTransformerLayerSave,
)
from molecular_force_field.models.ictd_fast import (
    FastSymmetricSTF,
    FastSymmetricTraceChain,
    decompose_rank2_generic,
)
from molecular_force_field.models.zbl import (
    ZBLConfig,
    ZBLRepulsionWrapper,
    compute_zbl_pair_energy,
    maybe_wrap_model_with_zbl,
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
    "PureCartesianICTDFix",
    "SphericalFix",
    "PureCartesianICTDTransformerLayerNoCrossFusion",
    "PureCartesianICTDTransformerLayerNoShallowReadout",
    "PureCartesianSparseTransformerLayer",
    "PureCartesianSparseTransformerLayerSave",
    "weighted_mse_loss_stats",
    "FastSymmetricSTF",
    "FastSymmetricTraceChain",
    "decompose_rank2_generic",
    "ZBLConfig",
    "ZBLRepulsionWrapper",
    "compute_zbl_pair_energy",
    "maybe_wrap_model_with_zbl",
    "MainNet",
    "MainNet2",
    "RMSELoss",
]

if PureCartesianICTDO3TransformerLayer is not None:
    __all__.append("PureCartesianICTDO3TransformerLayer")
if PureCartesianICTDFullO3TransformerLayer is not None:
    __all__.append("PureCartesianICTDFullO3TransformerLayer")
if PureCartesianICTDSaveO3TransformerLayer is not None:
    __all__.append("PureCartesianICTDSaveO3TransformerLayer")
if PureCartesianICTDSO2TransformerLayer is not None:
    __all__.append("PureCartesianICTDSO2TransformerLayer")
if PureCartesianICTDO2TransformerLayer is not None:
    __all__.append("PureCartesianICTDO2TransformerLayer")
if PureCartesianICTDNodeSO2TransformerLayer is not None:
    __all__.append("PureCartesianICTDNodeSO2TransformerLayer")
if PureCartesianICTDFixSO2 is not None:
    __all__.append("PureCartesianICTDFixSO2")
if PureCartesianICTDESCNSO2 is not None:
    __all__.append("PureCartesianICTDESCNSO2")
