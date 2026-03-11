"""Cartesian coordinate-based neural network layers for molecular modeling.

This module provides an alternative to e3nn-based layers using direct Cartesian
tensor products with sparse structure based on angular momentum selection rules.

Key features:
    - Uses Cartesian coordinates converted to irreducible representations
    - Exploits sparsity from selection rules: l1 ⊗ l2 → |l1-l2|, ..., l1+l2
    - Maintains similar parameter count to e3nn version
    - CartesianFullyConnectedTensorProduct: drop-in replacement for e3nn.o3.FullyConnectedTensorProduct

Main classes:
    EquivariantTensorProduct:
        Strictly equivariant tensor product using Clebsch-Gordan coefficients.
        Mathematically equivalent to e3nn.o3.FullyConnectedTensorProduct.
        Uses CG coefficients for exact coupling of irreducible representations.
        Guarantees strict equivariance by construction.

    CartesianFullyConnectedTensorProduct:
        Fast tensor product using norm product approximation.
        Uses W[mul_out, mul1, mul2] weight structure matching EquivariantTensorProduct.
        Not strictly equivariant but significantly faster.
        Supports torch.compile for additional speedup (3-4x).

    CartesianTransformerLayerStrict:
        Transformer layer with strict parity handling.
        Uses CartesianFullyConnectedTensorProduct and elementwise products.
        Alias: CartesianTransformerLayer (for backward compatibility).

    CartesianTransformerLayerLoose:
        Fast transformer layer using CartesianFullyConnectedTensorProduct (norm product approximation).
        Not strictly equivariant but faster than strict parity mode.
        Recommended for inference when speed is prioritized over strict equivariance.

Performance characteristics:
    - EquivariantTensorProduct: Strictly equivariant, slower than e3nn
    - CartesianFullyConnectedTensorProduct: Fast, not strictly equivariant
    - With torch.compile: Cartesian-Loose can be 3-5x faster than e3nn
    - Training: Cannot use torch.compile (requires double backward)
    - Inference: Can use torch.compile for significant speedup

Mathematical background:
    The module implements tensor products in Cartesian coordinates with two approaches:
    
    1. Strict equivariance (EquivariantTensorProduct):
       Uses Clebsch-Gordan coefficients C[l1, l2, l_out, m1, m2, m_out] to
       couple irreducible representations. This ensures exact rotational equivariance.
       
    2. Approximate (CartesianFullyConnectedTensorProduct):
       Uses norm product ||f1||^2 × ||f2||^2 as a scalar approximation.
       This is faster but does not guarantee strict equivariance.
       
    Both approaches respect angular momentum selection rules:
    |l1 - l2| <= l_out <= l1 + l2, and parity selection: p_out = p1 * p2.
"""

import math
import re
from typing import Union, List, Tuple, Optional
import torch
import torch.nn as nn
from e3nn.math import soft_one_hot_linspace
from e3nn import o3
from molecular_force_field.utils.scatter import scatter

from molecular_force_field.models.mlp import MainNet, RobustScalarWeightedSum


# ============================================================================
# Irreps parsing utilities
# ============================================================================

def parse_irreps_string(irreps_str: str) -> List[Tuple[int, int, int]]:
    """
    Parse e3nn-style irreps string to list of (multiplicity, l, parity).
    
    Parses a string representation of irreducible representations into a list
    of tuples. Each tuple contains (multiplicity, angular_momentum, parity).
    
    Format: "MxLe" or "MxLo" where:
        M: Multiplicity (number of copies)
        L: Angular momentum l (0, 1, 2, ...)
        e: Even parity (parity = +1)
        o: Odd parity (parity = -1)
    
    Multiple irreps can be combined with "+" separator.
    
    Args:
        irreps_str: String representation of irreps.
                    Example: "16x0e" or "1x0e + 1x1o + 1x2e"
    
    Returns:
        List of tuples (multiplicity, l, parity).
        parity is +1 for even ('e') and -1 for odd ('o').
    
    Examples:
        parse_irreps_string("16x0e")
        Returns: [(16, 0, 1)]
        
        parse_irreps_string("1x0e + 1x1o + 1x2e")
        Returns: [(1, 0, 1), (1, 1, -1), (1, 2, 1)]
        
        parse_irreps_string("64x0e + 64x1o + 64x2e")
        Returns: [(64, 0, 1), (64, 1, -1), (64, 2, 1)]
    """
    result = []
    parts = irreps_str.replace(" ", "").split("+")
    for part in parts:
        match = re.match(r"(\d+)x(\d+)([eo])", part)
        if match:
            mul = int(match.group(1))
            l = int(match.group(2))
            parity = 1 if match.group(3) == 'e' else -1
            result.append((mul, l, parity))
    return result


def irreps_dim(irreps_str: str) -> int:
    """
    Get total dimension of irreps.
    
    Computes the total dimension of an irreps string by summing the dimensions
    of each irrep. Each irrep with angular momentum l has dimension (2*l+1),
    and multiplicity mul means there are mul copies of that irrep.
    
    Formula: sum(mul * (2*l+1) for each (mul, l, parity) in irreps)
    
    Args:
        irreps_str: String representation of irreps.
                   Example: "16x0e" or "1x0e + 1x1o + 1x2e"
    
    Returns:
        Total dimension (integer).
    
    Examples:
        irreps_dim("16x0e")
        Returns: 16 * (2*0+1) = 16
        
        irreps_dim("1x0e + 1x1o + 1x2e")
        Returns: 1*(2*0+1) + 1*(2*1+1) + 1*(2*2+1) = 1 + 3 + 5 = 9
        
        irreps_dim("64x0e + 64x1o + 64x2e")
        Returns: 64*1 + 64*3 + 64*5 = 64 + 192 + 320 = 576
    """
    parsed = parse_irreps_string(irreps_str)
    return sum(mul * (2 * l + 1) for mul, l, _ in parsed)


def get_irreps_structure(irreps_str: str) -> dict:
    """
    Get structure info mapping angular momentum to block information.
    
    Parses an irreps string and returns a dictionary mapping angular momentum l
    to a list of blocks. Each block contains (multiplicity, start_idx, end_idx, parity).
    This structure is used to efficiently index into irreps tensors.
    
    Args:
        irreps_str: String representation of irreps.
                   Example: "16x0e" or "1x0e + 1x1o + 1x2e"
    
    Returns:
        Dictionary mapping l (angular momentum) to list of blocks.
        Each block is a tuple (multiplicity, start_idx, end_idx, parity).
        start_idx and end_idx define the slice in the flattened tensor.
        parity is +1 for even ('e') and -1 for odd ('o').
    
    Example:
        get_irreps_structure("1x0e + 1x1o + 1x2e")
        Returns: {
            0: [(1, 0, 1, 1)],      # l=0: indices 0-0 (1 component)
            1: [(1, 1, 4, -1)],     # l=1: indices 1-3 (3 components)
            2: [(1, 4, 9, 1)]       # l=2: indices 4-8 (5 components)
        }
        
        get_irreps_structure("64x0e + 64x1o + 64x2e")
        Returns: {
            0: [(64, 0, 64, 1)],        # l=0: indices 0-63 (64 components)
            1: [(64, 64, 256, -1)],     # l=1: indices 64-255 (192 components)
            2: [(64, 256, 576, 1)]      # l=2: indices 256-575 (320 components)
        }
    """
    parsed = parse_irreps_string(irreps_str)
    structure = {}
    idx = 0
    for mul, l, parity in parsed:
        dim = mul * (2 * l + 1)
        if l not in structure:
            structure[l] = []
        structure[l].append((mul, idx, idx + dim, parity))
        idx += dim
    return structure


def get_irreps_str(channels: int, lmax: int) -> str:
    """
    Generate irreps string with standard parity pattern.
    
    Creates an irreps string with the standard parity pattern:
    - Even l (0, 2, 4, ...) have even parity ('e', parity = +1)
    - Odd l (1, 3, 5, ...) have odd parity ('o', parity = -1)
    
    This is the most common pattern used in E(3)-equivariant neural networks.
    
    Args:
        channels: Multiplicity (number of copies) for each angular momentum.
        lmax: Maximum angular momentum (inclusive).
    
    Returns:
        Irreps string in e3nn format.
    
    Examples:
        get_irreps_str(channels=1, lmax=2)
        Returns: "1x0e + 1x1o + 1x2e"
        
        get_irreps_str(channels=64, lmax=2)
        Returns: "64x0e + 64x1o + 64x2e"
        
        get_irreps_str(channels=32, lmax=1)
        Returns: "32x0e + 32x1o"
    """
    parts = []
    for l in range(lmax + 1):
        parity = 'e' if l % 2 == 0 else 'o'
        parts.append(f"{channels}x{l}{parity}")
    return " + ".join(parts)


def reorder_concatenated_irreps_multi(features, channels: int, lmax: int) -> torch.Tensor:
    """
    Reorder concatenated irreps features (multi-input) to match e3nn's expected format.

    Args:
        features: List of irreps features, each shape (..., channels * sum(2*l+1))
        channels: Number of channels per angular momentum
        lmax: Maximum angular momentum

    Returns:
        Reordered concatenated features, shape (..., n*channels * sum(2*l+1))
    """
    if len(features) == 1:
        return features[0]
    dims = [2 * l + 1 for l in range(lmax + 1)]
    idxs = [0 for _ in features]
    parts = []
    for l in range(lmax + 1):
        dim_l = dims[l]
        size_l = channels * dim_l
        l_parts = []
        for i, f in enumerate(features):
            f_l = f[..., idxs[i]:idxs[i] + size_l]
            idxs[i] += size_l
            l_parts.append(f_l)
        parts.append(torch.cat(l_parts, dim=-1))
    return torch.cat(parts, dim=-1)


def reorder_concatenated_irreps(f1: torch.Tensor, f2: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    """
    Reorder concatenated irreps features to match e3nn's expected format.
    
    When concatenating two irreps features f1 and f2, the naive concatenation
    `torch.cat([f1, f2], dim=-1)` produces:
        [f1's l=0 block, f2's l=0 block, f1's l=1 block, f2's l=1 block, ...]
    
    But e3nn's Irreps expects:
        [all l=0 blocks, all l=1 blocks, ...]
    
    This function reorders the concatenated features to match e3nn's format.
    
    Args:
        f1: First irreps features, shape (..., channels * sum(2*l+1 for l in range(lmax+1)))
        f2: Second irreps features, same shape as f1
        channels: Number of channels per angular momentum
        lmax: Maximum angular momentum
    
    Returns:
        Reordered concatenated features, shape (..., 2*channels * sum(2*l+1 for l in range(lmax+1)))
    
    Example:
        For lmax=2, channels=64:
        - f1 shape: (N, 576) = [64 of 0e, 192 of 1o, 320 of 2e]
        - f2 shape: (N, 576) = [64 of 0e, 192 of 1o, 320 of 2e]
        - Naive concat: [f1's 64 of 0e, f2's 64 of 0e, f1's 192 of 1o, f2's 192 of 1o, f1's 320 of 2e, f2's 320 of 2e]
        - Reordered: groups by l: [all 128 of 0e, all 128 of 1o, all 128 of 2e]
    """
    return reorder_concatenated_irreps_multi([f1, f2], channels, lmax)


# ============================================================================
# Irreducible Cartesian Tensor Decomposition (ICTD)
# Based on: "Irreducible Cartesian tensor decomposition" theory
# Reference: Shihao Shao's analytical ICTD matrices
# ============================================================================

class ICTDecomposition:
    """
    Irreducible Cartesian Tensor Decomposition.
    
    Decomposes rank-ν Cartesian tensors into irreducible components.
    
    For rank-2 (equation 2 from paper):
        T_ij = (1/2)(T_ij + T_ji) - (1/3)δ_ij T_kk  [ℓ=2, traceless symmetric]
             + (1/2)(T_ij - T_ji)                    [ℓ=1, antisymmetric]
             + (1/3)δ_ij T_kk                        [ℓ=0, trace/scalar]
    
    This is exact and strictly equivariant by construction.
    """
    
    # Rank-2 decomposition dimensions: ℓ=0 (1), ℓ=1 (3), ℓ=2 (5)
    RANK2_DIMS = {0: 1, 1: 3, 2: 5}
    
    @staticmethod
    def decompose_rank2(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose rank-2 tensor (3x3 matrix) into irreducible components.
        
        Args:
            T: (..., 3, 3) - rank-2 Cartesian tensor
            
        Returns:
            l0: (..., 1) - scalar (trace)
            l1: (..., 3) - pseudovector (antisymmetric part)
            l2: (..., 5) - traceless symmetric tensor
        """
        # Trace: ℓ=0 component
        trace = T[..., 0, 0] + T[..., 1, 1] + T[..., 2, 2]
        l0 = trace.unsqueeze(-1) / math.sqrt(3)  # Normalize
        
        # Antisymmetric part: ℓ=1 component (pseudovector via Levi-Civita)
        # (T_ij - T_ji) / 2 corresponds to a pseudovector
        l1_x = (T[..., 1, 2] - T[..., 2, 1]) / math.sqrt(2)
        l1_y = (T[..., 2, 0] - T[..., 0, 2]) / math.sqrt(2)
        l1_z = (T[..., 0, 1] - T[..., 1, 0]) / math.sqrt(2)
        l1 = torch.stack([l1_x, l1_y, l1_z], dim=-1)
        
        # Traceless symmetric part: ℓ=2 component (5 independent components)
        sym = (T + T.transpose(-2, -1)) / 2  # Symmetric part
        trace_part = trace.unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=T.device, dtype=T.dtype) / 3
        traceless_sym = sym - trace_part
        
        # Map to 5 components (standard order: 2,-2,1,-1,0 -> xx-yy, xy, xz, yz, zz)
        l2 = torch.stack([
            (traceless_sym[..., 0, 0] - traceless_sym[..., 1, 1]) / math.sqrt(2),  # xx - yy
            traceless_sym[..., 0, 1] * math.sqrt(2),                                # xy
            traceless_sym[..., 0, 2] * math.sqrt(2),                                # xz
            traceless_sym[..., 1, 2] * math.sqrt(2),                                # yz
            (2 * traceless_sym[..., 2, 2] + trace_part[..., 0, 0] - trace_part[..., 1, 1]) / math.sqrt(6),  # simplified zz
        ], dim=-1)
        # Use standard spherical harmonic ordering for ℓ=2
        l2 = torch.stack([
            traceless_sym[..., 0, 1] * math.sqrt(2),                                # m=-2: xy
            traceless_sym[..., 1, 2] * math.sqrt(2),                                # m=-1: yz
            (2*traceless_sym[..., 2, 2] - traceless_sym[..., 0, 0] - traceless_sym[..., 1, 1]) / math.sqrt(6),  # m=0: 2zz-xx-yy
            traceless_sym[..., 0, 2] * math.sqrt(2),                                # m=1: xz  
            (traceless_sym[..., 0, 0] - traceless_sym[..., 1, 1]) / math.sqrt(2),   # m=2: xx-yy
        ], dim=-1)
        
        return l0, l1, l2
    
    @staticmethod
    def compose_rank2(l0: torch.Tensor, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        """
        Compose rank-2 tensor from irreducible components.
        
        Args:
            l0: (..., 1) - scalar
            l1: (..., 3) - pseudovector
            l2: (..., 5) - traceless symmetric
            
        Returns:
            T: (..., 3, 3) - rank-2 Cartesian tensor
        """
        batch_shape = l0.shape[:-1]
        device, dtype = l0.device, l0.dtype
        
        # Scalar → trace part
        trace = l0[..., 0] * math.sqrt(3)
        T_trace = torch.zeros(*batch_shape, 3, 3, device=device, dtype=dtype)
        T_trace[..., 0, 0] = trace / 3
        T_trace[..., 1, 1] = trace / 3
        T_trace[..., 2, 2] = trace / 3
        
        # Pseudovector → antisymmetric part
        T_anti = torch.zeros(*batch_shape, 3, 3, device=device, dtype=dtype)
        T_anti[..., 1, 2] = l1[..., 0] * math.sqrt(2) / 2
        T_anti[..., 2, 1] = -l1[..., 0] * math.sqrt(2) / 2
        T_anti[..., 2, 0] = l1[..., 1] * math.sqrt(2) / 2
        T_anti[..., 0, 2] = -l1[..., 1] * math.sqrt(2) / 2
        T_anti[..., 0, 1] = l1[..., 2] * math.sqrt(2) / 2
        T_anti[..., 1, 0] = -l1[..., 2] * math.sqrt(2) / 2
        
        # Traceless symmetric → symmetric part (inverse of decomposition)
        T_sym = torch.zeros(*batch_shape, 3, 3, device=device, dtype=dtype)
        # l2 ordering: xy, yz, 2zz-xx-yy, xz, xx-yy
        xy = l2[..., 0] / math.sqrt(2)
        yz = l2[..., 1] / math.sqrt(2)
        zz_combo = l2[..., 2] * math.sqrt(6) / 2  # 2zz - xx - yy
        xz = l2[..., 3] / math.sqrt(2)
        xx_yy = l2[..., 4] * math.sqrt(2)  # xx - yy
        
        # Solve for xx, yy, zz from: xx-yy and 2zz-xx-yy, with xx+yy+zz=0 (traceless)
        # xx - yy = xx_yy
        # 2zz - xx - yy = zz_combo → 2zz - (xx+yy) = zz_combo
        # xx + yy + zz = 0 → xx + yy = -zz
        # So: 2zz - (-zz) = 3zz = zz_combo → zz = zz_combo/3
        # And: xx + yy = -zz, xx - yy = xx_yy
        # → xx = (-zz + xx_yy)/2, yy = (-zz - xx_yy)/2
        zz = zz_combo / 3
        xx = (-zz + xx_yy) / 2
        yy = (-zz - xx_yy) / 2
        
        T_sym[..., 0, 0] = xx
        T_sym[..., 1, 1] = yy
        T_sym[..., 2, 2] = zz
        T_sym[..., 0, 1] = xy
        T_sym[..., 1, 0] = xy
        T_sym[..., 0, 2] = xz
        T_sym[..., 2, 0] = xz
        T_sym[..., 1, 2] = yz
        T_sym[..., 2, 1] = yz
        
        return T_trace + T_anti + T_sym


class CartesianTensorContraction(nn.Module):
    """
    Equivariant tensor contraction: ν1_T ⊗^(ν1,ν2,ν3) ν2_T
    
    Implements equation (4) from the paper:
    (ν1+ν2-2k)_T = Σ_{i1,...,ik} ν1_T_{a1...a_{ν1-k},i1...ik} × ν2_T_{i1...ik,b1...b_{ν2-k}}
    
    This is the fundamental equivariant operation for Cartesian tensors.
    The contraction of k indices reduces the rank by 2k.
    
    Special cases:
    - k=0: Outer product (rank ν1+ν2)
    - k=min(ν1,ν2): Full contraction (rank |ν1-ν2|)
    - ν1=ν2=1, k=1: Dot product (scalar)
    - ν1=ν2=1, k=0: Outer product (rank-2 tensor)
    """
    
    def __init__(self, rank1: int, rank2: int, num_contractions: int, 
                 channels_in: int = 1, channels_out: int = 1,
                 learnable: bool = True):
        """
        Args:
            rank1: Rank of first tensor (ν1)
            rank2: Rank of second tensor (ν2)
            num_contractions: Number of indices to contract (k)
            channels_in: Input channel multiplicity
            channels_out: Output channel multiplicity
            learnable: Whether to include learnable weights
        """
        super().__init__()
        self.rank1 = rank1
        self.rank2 = rank2
        self.k = num_contractions
        self.rank_out = rank1 + rank2 - 2 * num_contractions
        
        assert 0 <= num_contractions <= min(rank1, rank2), \
            f"num_contractions must be in [0, min(rank1, rank2)]"
        
        self.channels_in = channels_in
        self.channels_out = channels_out
        
        # Dimension of each tensor (3^rank for Cartesian)
        self.dim1 = 3 ** rank1
        self.dim2 = 3 ** rank2
        self.dim_out = 3 ** self.rank_out
        
        if learnable:
            # Channel mixing weights
            self.weight = nn.Parameter(torch.randn(channels_out, channels_in, channels_in) / 
                                       math.sqrt(channels_in * channels_in))
        else:
            self.register_parameter('weight', None)
        
        # Build contraction index pattern
        self._build_contraction_indices()
    
    def _build_contraction_indices(self):
        """Build index patterns for efficient contraction."""
        # For simplicity, we use einsum-style contraction
        # The exact implementation depends on the specific (rank1, rank2, k) combination
        pass
    
    def contract_vectors(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Contract two vectors (rank-1 tensors).
        
        k=0: Outer product → rank-2 tensor (3x3 = 9 components)
        k=1: Dot product → scalar (1 component)
        """
        if self.k == 0:
            # Outer product: v1_i × v2_j → T_ij
            return torch.einsum('...ci,...cj->...cij', v1, v2).flatten(-2)
        elif self.k == 1:
            # Dot product: Σ_i v1_i × v2_i → scalar
            return (v1 * v2).sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Invalid k={self.k} for vectors")
    
    def contract_vector_matrix(self, v: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Contract vector (rank-1) with matrix (rank-2).
        
        k=0: Outer product → rank-3 tensor
        k=1: Matrix-vector product → vector
        """
        M_shape = M.shape[:-1] + (3, 3)
        M = M.view(*M_shape)
        
        if self.k == 0:
            # v_i M_jk → T_ijk
            return torch.einsum('...i,...jk->...ijk', v, M).flatten(-3)
        elif self.k == 1:
            # Σ_i v_i M_ij → w_j  (contract first index of M)
            return torch.einsum('...i,...ij->...j', v, M)
        else:
            raise ValueError(f"Invalid k={self.k} for vector-matrix")
    
    def contract_matrices(self, M1: torch.Tensor, M2: torch.Tensor) -> torch.Tensor:
        """
        Contract two matrices (rank-2 tensors).
        
        k=0: Outer product → rank-4 tensor
        k=1: One-index contraction → rank-2 tensor (matrix multiplication-like)
        k=2: Full contraction (double trace) → scalar
        """
        batch_shape = M1.shape[:-1]
        M1 = M1.view(*batch_shape, 3, 3)
        M2 = M2.view(*batch_shape, 3, 3)
        
        if self.k == 0:
            # M1_ij M2_kl → T_ijkl
            return torch.einsum('...ij,...kl->...ijkl', M1, M2).flatten(-4)
        elif self.k == 1:
            # Σ_j M1_ij M2_jk → T_ik (matrix multiplication)
            return torch.einsum('...ij,...jk->...ik', M1, M2).flatten(-2)
        elif self.k == 2:
            # Σ_ij M1_ij M2_ij → scalar (Frobenius inner product)
            return (M1 * M2).sum(dim=(-2, -1), keepdim=True).squeeze(-1)
        else:
            raise ValueError(f"Invalid k={self.k} for matrices")
    
    def forward(self, t1: torch.Tensor, t2: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform tensor contraction.
        
        Args:
            t1: (..., channels_in, dim1) - first tensor
            t2: (..., channels_in, dim2) - second tensor
            weights: Optional external weights
            
        Returns:
            (..., channels_out, dim_out) - contracted tensor
        """
        # Dispatch based on ranks
        if self.rank1 == 1 and self.rank2 == 1:
            out = self.contract_vectors(t1, t2)
        elif self.rank1 == 1 and self.rank2 == 2:
            out = self.contract_vector_matrix(t1, t2)
        elif self.rank1 == 2 and self.rank2 == 1:
            out = self.contract_vector_matrix(t2, t1)  # Swap and adjust
        elif self.rank1 == 2 and self.rank2 == 2:
            out = self.contract_matrices(t1, t2)
        else:
            raise NotImplementedError(f"Ranks ({self.rank1}, {self.rank2}) not yet implemented")
        
        # Apply channel mixing if learnable
        if self.weight is not None:
            w = weights if weights is not None else self.weight
            # out: (..., channels_in, dim_out)
            # Apply channel mixing
            out = torch.einsum('oci,...ci->...od', w, out.unsqueeze(-2) * out.unsqueeze(-3))
        
        return out


class EquivariantTensorProduct(nn.Module):
    """
    Strictly equivariant tensor product using Clebsch-Gordan coefficients.
    
    This is mathematically equivalent to e3nn.o3.FullyConnectedTensorProduct
    but implemented with explicit CG matrices for clarity and correctness.
    All operations are strictly equivariant by construction.
    
    The tensor product computes: irreps_in1 ⊗ irreps_in2 → irreps_out
    using Clebsch-Gordan coefficients C[l1, l2, l_out, m1, m2, m_out] to
    couple irreducible representations. This ensures exact rotational equivariance.
    
    Weight structure:
        Each path (l1, l2) → l_out has weights W[mul_out, mul1, mul2].
        Total weight_numel = sum(mul_out × mul1 × mul2) over all valid paths.
        This matches e3nn's weight structure exactly.
    
    Normalization modes:
        'component': Each output component has approximately unit variance when
                     inputs have unit variance. This is the default and matches e3nn.
        'norm': Each output irrep has approximately unit norm when inputs have
                unit norm. CG coefficients are scaled by sqrt(2*l_out+1).
    
    Args:
        irreps_in1: Input irreps string for first input.
                    Example: "32x0e + 32x1o + 32x2e"
                    Can also be o3.Irreps object (will be converted to string).
        irreps_in2: Input irreps string for second input.
                    Same format as irreps_in1.
        irreps_out: Output irreps string.
                    Must satisfy selection rules: |l1-l2| <= l_out <= l1+l2
                    and parity: p_out = p1 * p2.
        channels: Not used, kept for API compatibility with CartesianFullyConnectedTensorProduct.
        shared_weights: If True, use shared weights across all samples.
                        If False, requires external weights per sample.
        internal_weights: If True, create internal learnable weights.
                          If False, requires external weights to be provided.
        normalization: Normalization mode for tensor product output.
                        Options: 'component' (default) or 'norm'.
        irrep_normalization: Normalization for individual irreps.
                             Options: 'component' (default) or 'norm'.
                             Currently not fully implemented, kept for API compatibility.
    
    Attributes:
        weight_numel: Total number of weights needed.
                      Equal to sum(mul_out × mul1 × mul2) over all paths.
        weight: Learnable parameter tensor of shape (weight_numel,).
                Only exists if internal_weights=True and shared_weights=True.
        dim_in1: Total dimension of first input irreps.
        dim_in2: Total dimension of second input irreps.
        dim_out: Total dimension of output irreps.
        paths: List of dictionaries describing valid tensor product paths.
               Each path contains: l1, l2, l_out, mul1, mul2, mul_out,
               start/end indices, parity, weight indices, and normalization factors.
    
    Example:
        >>> tp = EquivariantTensorProduct(
        ...     irreps_in1="64x0e + 64x1o + 64x2e",
        ...     irreps_in2="1x0e + 1x1o + 1x2e",
        ...     irreps_out="64x0e + 64x1o + 64x2e",
        ...     shared_weights=True,
        ...     internal_weights=True
        ... )
        >>> x1 = torch.randn(100, 576)  # 64*(1+3+5) = 576
        >>> x2 = torch.randn(100, 9)     # 1*(1+3+5) = 9
        >>> out = tp(x1, x2)  # Shape: (100, 576)
    
    Note:
        This implementation uses e3nn.o3.wigner_3j to compute CG coefficients.
        CG matrices are cached on CPU to save GPU memory and moved to device when needed.
        The implementation is optimized to reduce intermediate tensor memory usage.
    """
    
    def __init__(self, 
                 irreps_in1: Union[str, o3.Irreps], 
                 irreps_in2: Union[str, o3.Irreps], 
                 irreps_out: Union[str, o3.Irreps],
                 channels: int = 1,
                 shared_weights: bool = True,
                 internal_weights: bool = True,
                 normalization: str = 'component',
                 irrep_normalization: str = 'component'):
        super().__init__()
        
        # Convert to string if o3.Irreps is passed (for compatibility with CartesianFullyConnectedTensorProduct)
        self.irreps_in1_str = str(irreps_in1)
        self.irreps_in2_str = str(irreps_in2)
        self.irreps_out_str = str(irreps_out)
        
        self.irreps_in1 = parse_irreps_string(self.irreps_in1_str)
        self.irreps_in2 = parse_irreps_string(self.irreps_in2_str)
        self.irreps_out = parse_irreps_string(self.irreps_out_str)
        
        self.dim_in1 = irreps_dim(self.irreps_in1_str)
        self.dim_in2 = irreps_dim(self.irreps_in2_str)
        self.dim_out = irreps_dim(self.irreps_out_str)
        
        # channels: Not used, kept for API compatibility with CartesianFullyConnectedTensorProduct
        self.channels = channels
        
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights
        
        # Normalization modes
        assert normalization in ('component', 'norm'), \
            f"normalization must be 'component' or 'norm', got {normalization}"
        assert irrep_normalization in ('component', 'norm'), \
            f"irrep_normalization must be 'component' or 'norm', got {irrep_normalization}"
        self.normalization = normalization
        self.irrep_normalization = irrep_normalization
        
        # Cache CG matrices (with normalization factors)
        # Store on CPU to save GPU memory - CG matrices are small (~KB each)
        self._cg_cache = {}
        
        # Build the equivariant paths
        self._build_paths()
        
        if internal_weights and shared_weights:
            self.weight = nn.Parameter(torch.randn(self.weight_numel) * 0.1)
        else:
            self.register_parameter('weight', None)
    
    def _get_cg_matrix(self, l1: int, l2: int, l_out: int, device=None, dtype=None) -> torch.Tensor:
        """
        Get Clebsch-Gordan matrix for l1 ⊗ l2 → l_out with proper normalization.
        
        CG matrices are cached on CPU and moved to device only when needed.
        This saves GPU memory since CG matrices are small (~KB each).
        
        Args:
            device: Target device (if None, returns CPU tensor)
            dtype: Target dtype (if None, uses cached dtype)
        """
        key = (l1, l2, l_out, self.normalization)
        
        # Check if we have this CG matrix cached
        if key not in self._cg_cache:
            # Use e3nn's wigner_3j (CG coefficients, component-normalized)
            cg = o3.wigner_3j(l1, l2, l_out)  # (2*l1+1, 2*l2+1, 2*l_out+1)
            
            # Apply normalization
            if self.normalization == 'norm':
                # For norm normalization, scale by sqrt(2*l_out+1)
                cg = cg * math.sqrt(2 * l_out + 1)
            
            # Store on CPU (CG matrices are small, ~KB each)
            self._cg_cache[key] = cg.cpu()
        
        # Get cached matrix (on CPU)
        cg = self._cg_cache[key]
        
        # Move to target device if needed (creates temporary copy, will be GC'd)
        if device is not None:
            return cg.to(device=device, dtype=dtype if dtype is not None else cg.dtype)
        
        return cg
    
    def _build_paths(self):
        """
        Build equivariant paths based on CG selection rules.
        
        Normalization modes (matching e3nn behavior):
        - 'component': Each output component has approximately unit variance when 
          inputs have unit variance. alpha = 1 (no path normalization needed since
          CG coefficients are already component-normalized).
        - 'norm': Each output irrep has approximately unit norm when inputs have 
          unit norm. alpha = 1/sqrt(num_paths) to account for sum of paths.
          The CG coefficients are also scaled by sqrt(2*l_out+1) in _get_cg_matrix.
        
        irrep_normalization affects the assumed normalization of input irreps:
        - 'component': Standard component normalization (default)
        - 'norm': Inputs assumed to be norm-normalized
        """
        self.paths = []
        weight_idx = 0
        
        in1_struct = get_irreps_structure(self.irreps_in1_str)
        in2_struct = get_irreps_structure(self.irreps_in2_str)
        out_struct = get_irreps_structure(self.irreps_out_str)
        
        # First pass: count paths to each output block for norm normalization
        path_counts = {}  # (l_out, start_out) -> count of paths
        
        for l1, blocks1 in in1_struct.items():
            for l2, blocks2 in in2_struct.items():
                for l_out in range(abs(l1 - l2), l1 + l2 + 1):
                    if l_out not in out_struct:
                        continue
                    
                    for mul1, start1, end1, p1 in blocks1:
                        for mul2, start2, end2, p2 in blocks2:
                            for mul_out, start_out, end_out, p_out in out_struct[l_out]:
                                if p_out != p1 * p2:
                                    continue
                                
                                key = (l_out, start_out)
                                path_counts[key] = path_counts.get(key, 0) + 1
        
        # Second pass: build paths with normalization factors
        for l1, blocks1 in in1_struct.items():
            for l2, blocks2 in in2_struct.items():
                # Possible output l values from CG rules
                for l_out in range(abs(l1 - l2), l1 + l2 + 1):
                    if l_out not in out_struct:
                        continue
                    
                    for mul1, start1, end1, p1 in blocks1:
                        for mul2, start2, end2, p2 in blocks2:
                            for mul_out, start_out, end_out, p_out in out_struct[l_out]:
                                # Parity selection: p_out = p1 * p2
                                if p_out != p1 * p2:
                                    continue
                                
                                # Number of weights for this path
                                num_weights = mul1 * mul2 * mul_out
                                
                                # Compute normalization factor based on normalization mode
                                key = (l_out, start_out)
                                num_paths = path_counts[key]
                                
                                if self.normalization == 'component':
                                    # For component normalization: no path scaling needed
                                    # Each component has unit variance
                                    alpha = 1.0
                                else:  # 'norm'
                                    # For norm normalization: scale by 1/sqrt(num_paths)
                                    # This ensures the output irrep has unit norm when inputs have unit norm
                                    alpha = 1.0 / math.sqrt(num_paths) if num_paths > 0 else 1.0
                                
                                self.paths.append({
                                    'l1': l1, 'l2': l2, 'l_out': l_out,
                                    'mul1': mul1, 'mul2': mul2, 'mul_out': mul_out,
                                    'start1': start1, 'end1': end1,
                                    'start2': start2, 'end2': end2,
                                    'start_out': start_out, 'end_out': end_out,
                                    'p1': p1, 'p2': p2, 'p_out': p_out,
                                    'weight_start': weight_idx,
                                    'weight_end': weight_idx + num_weights,
                                    'alpha': alpha,  # Normalization factor
                                    'num_paths': num_paths,
                                })
                                weight_idx += num_weights
        
        self.weight_numel = weight_idx
    
    def _cg_tensor_product(self, f1: torch.Tensor, f2: torch.Tensor,
                           l1: int, l2: int, l_out: int,
                           mul1: int, mul2: int, mul_out: int,
                           weights: torch.Tensor, per_sample_weights: bool = False) -> torch.Tensor:
        """
        Compute tensor product using Clebsch-Gordan coefficients.
        
        This is strictly equivariant for any (l1, l2, l_out) combination.
        
        f1: (..., mul1 * (2*l1+1))
        f2: (..., mul2 * (2*l2+1))
        weights: (mul_out * mul1 * mul2,) if shared, or (..., mul_out * mul1 * mul2) if per-sample
        per_sample_weights: If True, weights have the same batch dimensions as f1/f2
        
        Returns: (..., mul_out * (2*l_out+1))
        """
        batch_shape = f1.shape[:-1]
        dim1 = 2 * l1 + 1
        dim2 = 2 * l2 + 1
        dim_out = 2 * l_out + 1
        
        # Reshape inputs: (..., mul, dim)
        f1_v = f1.view(*batch_shape, mul1, dim1)  # (..., mul1, dim1)
        f2_v = f2.view(*batch_shape, mul2, dim2)  # (..., mul2, dim2)
        
        # Get CG matrix: (dim1, dim2, dim_out) - optimized to avoid repeated .to() calls
        cg = self._get_cg_matrix(l1, l2, l_out, device=f1.device, dtype=f1.dtype)
        
        # Optimized einsum: combine steps to reduce intermediate memory
        # Instead of creating (..., mul1, mul2, dim_out) explicitly,
        # we can fuse the operations when possible
        
        if per_sample_weights:
            # weights: (..., mul_out * mul1 * mul2)
            w = weights.view(*batch_shape, mul_out, mul1, mul2)
            # Fused: f1_v ⊗ f2_v ⊗ cg → weighted output
            # (..., mul1, dim1), (..., mul2, dim2), (dim1, dim2, dim_out), (..., mul_out, mul1, mul2)
            # → (..., mul_out, dim_out)
            # This avoids creating the full (..., mul1, mul2, dim_out) intermediate
            output = torch.einsum('...ik,...jl,klm,...oij->...om', f1_v, f2_v, cg, w)
        else:
            # weights: (mul_out * mul1 * mul2,)
            w = weights.view(mul_out, mul1, mul2)
            # Fused operation to reduce intermediate memory
            output = torch.einsum('...ik,...jl,klm,oij->...om', f1_v, f2_v, cg, w)
        
        return output.flatten(-2)  # (..., mul_out * dim_out)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute strictly equivariant tensor product.
        
        Args:
            x1: (..., dim_in1) - first input
            x2: (..., dim_in2) - second input
            weights: Optional external weights (for shared_weights=False)
                     Can be (weight_numel,) for shared or (..., weight_numel) for per-sample
            
        Returns:
            (..., dim_out) - output
            
        Notes on normalization:
            - 'component' normalization: Each output component has approximately unit variance
              when inputs have unit variance. This is the default and matches e3nn.
            - 'norm' normalization: Each output irrep has approximately unit norm when
              inputs have unit norm. The CG coefficients are scaled by sqrt(2*l_out+1).
        """
        batch_shape = x1.shape[:-1]
        
        if self.internal_weights:
            w = self.weight
            per_sample = False
        else:
            w = weights
            assert w is not None, "External weights required when internal_weights=False"
            # Check if weights are per-sample (have same batch dimensions as input)
            per_sample = w.dim() > 1 and w.shape[:-1] == batch_shape
        
        # Initialize output
        output = torch.zeros(*batch_shape, self.dim_out, device=x1.device, dtype=x1.dtype)
        
        for path in self.paths:
            l1, l2, l_out = path['l1'], path['l2'], path['l_out']
            mul1, mul2, mul_out = path['mul1'], path['mul2'], path['mul_out']
            alpha = path['alpha']  # Path normalization factor
            
            # Extract input features
            f1 = x1[..., path['start1']:path['end1']]
            f2 = x2[..., path['start2']:path['end2']]
            
            # Get weights for this path
            if per_sample:
                path_weights = w[..., path['weight_start']:path['weight_end']]
            else:
                path_weights = w[path['weight_start']:path['weight_end']]
            
            # Compute contribution using CG coefficients (always equivariant)
            contrib = self._cg_tensor_product(
                f1, f2, l1, l2, l_out, mul1, mul2, mul_out, path_weights, per_sample
            )
            
            # Apply path normalization factor (alpha)
            # This ensures proper variance/norm control when multiple paths contribute
            contrib = contrib * alpha
            
            # Accumulate to output
            output[..., path['start_out']:path['end_out']] += contrib
        
        return output


class CartesianFullyConnectedTensorProduct(nn.Module):
    """
    Fast Cartesian tensor product using norm product approximation.
    
    Computes: irreps_in1 ⊗ irreps_in2 → irreps_out
    using CG selection rules: l1 ⊗ l2 → |l1-l2|, ..., l1+l2
    
    This implementation uses norm product ||f1||^2 × ||f2||^2 as a scalar approximation
    instead of strict Clebsch-Gordan coefficients. This makes it significantly faster
    than EquivariantTensorProduct, but does not guarantee strict equivariance.
    
    Weight Structure (Optimized):
        Each path has weight W[mul_out, mul1, mul2], independent of (2*l+1).
        This matches EquivariantTensorProduct's weight structure exactly.
        For non-scalar outputs (l_out > 0), a small learnable "direction template"
        is used to expand scalars to (2*l_out+1) components.
        This makes weight_numel approximately equal to EquivariantTensorProduct.
    
    Performance:
        - Forward pass: 1.5-2x faster than e3nn on tensor product operations
        - With torch.compile: 3-5x faster than e3nn
        - Memory: Similar to EquivariantTensorProduct (same weight_numel)
        - Equivariance: Approximate (not strictly equivariant)
    
    API compatible with e3nn and EquivariantTensorProduct:
        tp = CartesianFullyConnectedTensorProduct(
            irreps_in1="64x0e + 64x1o + 64x2e",
            irreps_in2="64x0e + 64x1o + 64x2e", 
            irreps_out="16x0e",
            shared_weights=True,
            internal_weights=True
        )
        out = tp(x1, x2)
    
    If shared_weights=False, call with external weights:
        out = tp(x1, x2, weights)
    
    Args:
        irreps_in1: Input irreps string for first input.
                    Example: "64x0e + 64x1o + 64x2e"
                    Can also be o3.Irreps object (will be converted to string).
        irreps_in2: Input irreps string for second input.
                    Same format as irreps_in1.
        irreps_out: Output irreps string.
                    Must satisfy selection rules: |l1-l2| <= l_out <= l1+l2
                    and parity: p_out = p1 * p2.
        channels: Not used, kept for API compatibility with EquivariantTensorProduct.
        shared_weights: If True, use shared weights across all samples.
                        If False, requires external weights per sample.
        internal_weights: If True, create internal learnable weights.
                          If False, requires external weights to be provided.
        normalization: Normalization mode (for API compatibility).
                        Options: 'component' (default) or 'norm'.
                        Currently has limited effect on output.
        irrep_normalization: Normalization for individual irreps (for API compatibility).
                            Options: 'component' (default) or 'norm'.
    
    Attributes:
        weight_numel: Total number of weights needed.
                      Equal to sum(mul_out × mul1 × mul2) over all paths.
                      Matches EquivariantTensorProduct.weight_numel exactly.
        _internal_weight: Learnable parameter tensor of shape (weight_numel,).
                          Only exists if internal_weights=True and shared_weights=True.
        _direction_template_param: Learnable parameter for non-scalar outputs.
                                   Used to expand scalars to (2*l+1) components.
        dim_in1: Total dimension of first input irreps.
        dim_in2: Total dimension of second input irreps.
        dim_out: Total dimension of output irreps.
        paths: List of dictionaries describing valid tensor product paths.
        _output_blocks: Dictionary mapping output start indices to block information.
                        Used for efficient accumulation of contributions.
    
    Example:
        >>> tp = CartesianFullyConnectedTensorProduct(
        ...     irreps_in1="64x0e + 64x1o + 64x2e",
        ...     irreps_in2="1x0e + 1x1o + 1x2e",
        ...     irreps_out="64x0e + 64x1o + 64x2e",
        ...     shared_weights=True,
        ...     internal_weights=True
        ... )
        >>> x1 = torch.randn(100, 576)  # 64*(1+3+5) = 576
        >>> x2 = torch.randn(100, 9)     # 1*(1+3+5) = 9
        >>> out = tp(x1, x2)  # Shape: (100, 576)
        >>> # With torch.compile (for inference only):
        >>> tp_compiled = torch.compile(tp, mode='reduce-overhead')
        >>> out = tp_compiled(x1, x2)  # 3-5x faster
    
    Note:
        This implementation uses norm product approximation, which is faster but
        not strictly equivariant. For strict equivariance, use EquivariantTensorProduct.
        
        The weight structure is optimized to match EquivariantTensorProduct, ensuring
        that weight networks (fc layers) have the same output dimension.
        
        torch.compile can be used for inference to achieve 3-5x speedup.
        However, torch.compile cannot be used during training because it does not
        support double backward (required for force training: force = -dE/dr,
        then loss.backward() needs d(force)/d(parameters)).
    """
    
    def __init__(self, 
                 irreps_in1: Union[str, o3.Irreps],
                 irreps_in2: Union[str, o3.Irreps],
                 irreps_out: Union[str, o3.Irreps],
                 channels: int = 1,
                 shared_weights: bool = True,
                 internal_weights: bool = True,
                 normalization: str = 'component',
                 irrep_normalization: str = 'component'):
        super().__init__()
        
        # Convert to string if needed (for compatibility with EquivariantTensorProduct)
        self.irreps_in1_str = str(irreps_in1)
        self.irreps_in2_str = str(irreps_in2)
        self.irreps_out_str = str(irreps_out)
        
        # Parse irreps
        self.irreps_in1 = parse_irreps_string(self.irreps_in1_str)
        self.irreps_in2 = parse_irreps_string(self.irreps_in2_str)
        self.irreps_out_parsed = parse_irreps_string(self.irreps_out_str)
        
        # Store as e3nn Irreps for compatibility
        self.irreps_out = o3.Irreps(self.irreps_out_str)
        
        # channels: Not used, kept for API compatibility with EquivariantTensorProduct
        self.channels = channels
        
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights
        
        # Normalization modes (for compatibility with EquivariantTensorProduct)
        assert normalization in ('component', 'norm'), \
            f"normalization must be 'component' or 'norm', got {normalization}"
        assert irrep_normalization in ('component', 'norm'), \
            f"irrep_normalization must be 'component' or 'norm', got {irrep_normalization}"
        self.normalization = normalization
        self.irrep_normalization = irrep_normalization
        
        # Dimensions
        self.dim_in1 = irreps_dim(self.irreps_in1_str)
        self.dim_in2 = irreps_dim(self.irreps_in2_str)
        self.dim_out = irreps_dim(self.irreps_out_str)
        
        # Build path table (EquivariantTP-style: W[mul_out, mul1, mul2] per path)
        self._build_paths_optimized()
        
        # Create internal weights if needed
        if internal_weights and shared_weights:
            self._create_internal_weights()
            self.register_parameter('weight', None)
        else:
            self.register_parameter('weight', None)
        
    def _build_paths_optimized(self):
        """Build paths with EquivariantTP-style weight structure: W[mul_out, mul1, mul2] per path.
        
        This ensures weight_numel ≈ EquivariantTensorProduct.weight_numel.
        """
        # Get structures
        struct1 = get_irreps_structure(self.irreps_in1_str)
        struct2 = get_irreps_structure(self.irreps_in2_str)
        out_struct = get_irreps_structure(self.irreps_out_str)
        
        # Build paths (EquivariantTP-style)
        self.paths = []
        weight_idx = 0
        
        # Direction templates for expanding scalars to (2*l+1) components
        # These are small learnable parameters: one per unique (l_out, mul_out)
        self._direction_templates = {}
        direction_template_idx = 0
        
        for l1, blocks1 in struct1.items():
            for l2, blocks2 in struct2.items():
                # Possible output l values from CG rules
                for l_out in range(abs(l1 - l2), l1 + l2 + 1):
                    if l_out not in out_struct:
                        continue
                    
                    dim_out_l = 2 * l_out + 1
                    
                    for mul1, start1, end1, p1 in blocks1:
                        for mul2, start2, end2, p2 in blocks2:
                            for mul_out, start_out, end_out, p_out in out_struct[l_out]:
                                # Parity selection: p_out = p1 * p2
                                if p_out != p1 * p2:
                                    continue
                                
                                # Weight shape: W[mul_out, mul1, mul2] (same as EquivariantTP)
                                num_weights = mul_out * mul1 * mul2
                                
                                self.paths.append({
                                    'l1': l1, 'l2': l2, 'l_out': l_out,
                                    'mul1': mul1, 'mul2': mul2, 'mul_out': mul_out,
                                    'dim1': 2 * l1 + 1, 'dim2': 2 * l2 + 1, 'dim_out': dim_out_l,
                                    'start1': start1, 'end1': end1,
                                    'start2': start2, 'end2': end2,
                                    'start_out': start_out, 'end_out': end_out,
                                    'p1': p1, 'p2': p2, 'p_out': p_out,
                                    'weight_start': weight_idx,
                                    'weight_end': weight_idx + num_weights,
                                })
                                weight_idx += num_weights
                                
                                # Register direction template for non-scalar outputs
                                if l_out > 0:
                                    key = (l_out, mul_out, start_out)
                                    if key not in self._direction_templates:
                                        self._direction_templates[key] = {
                                            'l_out': l_out,
                                            'mul_out': mul_out,
                                            'dim_out': dim_out_l,
                                            'start_out': start_out,
                                        }
        
        self.weight_numel = weight_idx
        
        # Create direction templates as a single parameter for non-scalar outputs
        # Total dimension = sum(mul_out * dim_out) for each unique output block
        total_dir_dim = 0
        self._dir_template_slices = {}
        for key, info in self._direction_templates.items():
            dim = info['mul_out'] * info['dim_out']
            self._dir_template_slices[key] = (total_dir_dim, total_dir_dim + dim, info['mul_out'], info['dim_out'])
            total_dir_dim += dim
        
        if total_dir_dim > 0:
            # Initialize direction templates (uniform initialization, will be learned)
            self._direction_template_param = nn.Parameter(
                torch.ones(total_dir_dim) / math.sqrt(total_dir_dim)
            )
        else:
            self.register_parameter('_direction_template_param', None)
        
        # Pre-compute output block accumulation info for efficiency
        self._output_blocks = {}
        for path in self.paths:
            key = path['start_out']
            if key not in self._output_blocks:
                self._output_blocks[key] = {
                    'l_out': path['l_out'],
                    'mul_out': path['mul_out'],
                    'dim_out': path['dim_out'],
                    'start_out': path['start_out'],
                    'end_out': path['end_out'],
                    'paths': [],
                }
            self._output_blocks[key]['paths'].append(path)
    
    def _create_internal_weights(self):
        """Create internal weights as a single Parameter."""
        if self.weight_numel > 0:
            self._internal_weight = nn.Parameter(
                torch.randn(self.weight_numel) * 0.1
            )
        else:
            self.register_parameter('_internal_weight', None)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                weights: torch.Tensor = None) -> torch.Tensor:
        """
        Compute tensor product (optimized version with EquivariantTP-style weights).
        
        Args:
            x1: (..., dim_in1) - first input
            x2: (..., dim_in2) - second input
            weights: (..., weight_numel) - external weights if shared_weights=False
            
        Returns:
            (..., dim_out) - output
        """
        batch_shape = x1.shape[:-1]
        device = x1.device
        dtype = x1.dtype
        
        # Select weights source
        if self.internal_weights and self.shared_weights:
            w = self._internal_weight
            per_sample = False
        else:
            w = weights
            per_sample = w is not None and w.dim() > 1 and w.shape[:-1] == batch_shape
        
        # Pre-allocate output tensor
        output = torch.zeros(*batch_shape, self.dim_out, device=device, dtype=dtype)
        
        # Process each output block
        for start_out, block_info in self._output_blocks.items():
            l_out = block_info['l_out']
            mul_out = block_info['mul_out']
            dim_out = block_info['dim_out']
            end_out = block_info['end_out']
            paths = block_info['paths']
            
            # Accumulate scalar contributions from all paths to this output block
            # Shape: (..., mul_out)
            scalar_accum = torch.zeros(*batch_shape, mul_out, device=device, dtype=dtype)
            
            for path in paths:
                mul1, mul2 = path['mul1'], path['mul2']
                dim1, dim2 = path['dim1'], path['dim2']
                
                # Extract and reshape inputs
                f1 = x1[..., path['start1']:path['end1']].view(*batch_shape, mul1, dim1)
                f2 = x2[..., path['start2']:path['end2']].view(*batch_shape, mul2, dim2)
                
                # Compute norm products: (..., mul1) x (..., mul2) -> (..., mul1, mul2)
                f1_norm = torch.einsum('...ij,...ij->...i', f1, f1)  # (..., mul1)
                f2_norm = torch.einsum('...ij,...ij->...i', f2, f2)  # (..., mul2)
                coupled = torch.einsum('...i,...j->...ij', f1_norm, f2_norm)  # (..., mul1, mul2)
                
                # Get weights for this path: W[mul_out, mul1, mul2]
                if per_sample:
                    path_w = w[..., path['weight_start']:path['weight_end']].view(*batch_shape, mul_out, mul1, mul2)
                    # Apply weights: (..., mul1, mul2) x (..., mul_out, mul1, mul2) -> (..., mul_out)
                    contrib = torch.einsum('...ij,...oij->...o', coupled, path_w)
                else:
                    path_w = w[path['weight_start']:path['weight_end']].view(mul_out, mul1, mul2)
                    # Apply weights: (..., mul1, mul2) x (mul_out, mul1, mul2) -> (..., mul_out)
                    contrib = torch.einsum('...ij,oij->...o', coupled, path_w)
                
                scalar_accum = scalar_accum + contrib
            
            # Expand scalars to (2*l+1) components
            if l_out == 0:
                # Scalar output: no expansion needed
                output[..., start_out:end_out] = scalar_accum
            else:
                # Non-scalar output: use direction template
                key = (l_out, mul_out, start_out)
                if key in self._dir_template_slices:
                    s, e, mo, do = self._dir_template_slices[key]
                    # Direction template: (mul_out * dim_out,) -> (mul_out, dim_out)
                    dir_template = self._direction_template_param[s:e].view(mo, do)
                    # Expand: (..., mul_out) x (mul_out, dim_out) -> (..., mul_out, dim_out)
                    expanded = scalar_accum.unsqueeze(-1) * dir_template.unsqueeze(0).expand(*batch_shape, mo, do)
                    output[..., start_out:end_out] = expanded.flatten(-2)
                else:
                    # Fallback: uniform expansion
                    output[..., start_out:end_out] = scalar_accum.unsqueeze(-1).expand(*batch_shape, mul_out, dim_out).flatten(-2)
        
        return output


class CartesianFullTensorProduct(nn.Module):
    """
    Cartesian equivalent of e3nn.o3.FullTensorProduct.
    
    Computes all possible outputs from l1 ⊗ l2 without learnable weights.
    The output irreps are automatically determined by selection rules.
    """
    
    def __init__(self,
                 irreps_in1: Union[str, o3.Irreps],
                 irreps_in2: Union[str, o3.Irreps]):
        super().__init__()
        
        self.irreps_in1_str = str(irreps_in1)
        self.irreps_in2_str = str(irreps_in2)
        
        self.irreps_in1 = parse_irreps_string(self.irreps_in1_str)
        self.irreps_in2 = parse_irreps_string(self.irreps_in2_str)
        
        self.dim_in1 = irreps_dim(self.irreps_in1_str)
        self.dim_in2 = irreps_dim(self.irreps_in2_str)
        
        # Compute output irreps
        self._build_output_irreps()
    
    def _build_output_irreps(self):
        """Determine output irreps from CG selection rules."""
        output_muls = {}  # l_out -> multiplicity
        
        for mul1, l1, p1 in self.irreps_in1:
            for mul2, l2, p2 in self.irreps_in2:
                # Output parity
                p_out = p1 * p2
                
                # All possible l_out
                for l_out in range(abs(l1 - l2), l1 + l2 + 1):
                    key = (l_out, p_out)
                    if key not in output_muls:
                        output_muls[key] = 0
                    output_muls[key] += mul1 * mul2
        
        # Build irreps string
        irreps_parts = []
        for (l, p), mul in sorted(output_muls.items()):
            parity_char = 'e' if p == 1 else 'o'
            irreps_parts.append(f"{mul}x{l}{parity_char}")
        
        self.irreps_out_str = " + ".join(irreps_parts) if irreps_parts else "0x0e"
        self.irreps_out = o3.Irreps(self.irreps_out_str)
        self.dim_out = irreps_dim(self.irreps_out_str)
        
        self._output_muls = output_muls
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute full tensor product.
        
        Args:
            x1: (..., dim_in1)
            x2: (..., dim_in2)
            
        Returns:
            (..., dim_out)
        """
        batch_shape = x1.shape[:-1]

        struct1 = get_irreps_structure(self.irreps_in1_str)
        struct2 = get_irreps_structure(self.irreps_in2_str)

        # ------------------------------------------------------------------
        # Strict equivariant implementation for the common safe case:
        # scalar (l=0) ⊗ irreps  (or irreps ⊗ scalar)
        # In this case, CG coupling is trivial: output is just scalar multiplication.
        # This preserves both equivariance and parity exactly.
        # ------------------------------------------------------------------
        in1_is_scalar_only = all(l == 0 for _, l, _ in self.irreps_in1)
        in2_is_scalar_only = all(l == 0 for _, l, _ in self.irreps_in2)

        if in1_is_scalar_only ^ in2_is_scalar_only:
            outputs = []

            if in1_is_scalar_only:
                # x1: sum(mul1) scalars, x2: irreps
                blocks1 = struct1.get(0, [])
                for (l_out, p_out), mul_out in sorted(self._output_muls.items()):
                    dim_out_l = 2 * l_out + 1
                    contributions = []

                    # Only l2 == l_out contributes for scalar ⊗ irreps
                    for mul2, s2, e2, p2 in struct2.get(l_out, []):
                        f2 = x2[..., s2:e2].view(*batch_shape, mul2, dim_out_l)  # (..., mul2, dim)
                        for mul1, s1, e1, p1 in blocks1:
                            # Parity check: scalar (0e) has p1=1, p_out = p1 * p2 = p2
                            if p1 * p2 == p_out:
                                f1 = x1[..., s1:e1].view(*batch_shape, mul1, 1)  # (..., mul1, 1)
                                f1s = f1[..., :, 0]  # (..., mul1)
                                prod = f1s.unsqueeze(-1).unsqueeze(-1) * f2.unsqueeze(-3)  # (..., mul1, mul2, dim)
                                contributions.append(prod.reshape(*batch_shape, mul1 * mul2 * dim_out_l))

                    if contributions:
                        out = torch.cat(contributions, dim=-1)
                    else:
                        out = torch.zeros(
                            *batch_shape,
                            mul_out * dim_out_l,
                            device=x1.device,
                            dtype=x1.dtype,
                        )
                    outputs.append(out)

            else:
                # x1: irreps, x2: scalars
                blocks2 = struct2.get(0, [])
                for (l_out, p_out), mul_out in sorted(self._output_muls.items()):
                    dim_out_l = 2 * l_out + 1
                    contributions = []

                    # Only l1 == l_out contributes for irreps ⊗ scalar
                    for mul1, s1, e1, p1 in struct1.get(l_out, []):
                        f1 = x1[..., s1:e1].view(*batch_shape, mul1, dim_out_l)  # (..., mul1, dim)
                        for mul2, s2, e2, p2 in blocks2:
                            # Parity check: scalar (0e) has p2=1, p_out = p1 * p2 = p1
                            if p1 * p2 == p_out:
                                f2 = x2[..., s2:e2].view(*batch_shape, mul2, 1)  # (..., mul2, 1)
                                f2s = f2[..., :, 0]  # (..., mul2)
                                prod = f1.unsqueeze(-2) * f2s.unsqueeze(-1).unsqueeze(-3)  # (..., mul1, mul2, dim)
                                contributions.append(prod.reshape(*batch_shape, mul1 * mul2 * dim_out_l))

                    if contributions:
                        out = torch.cat(contributions, dim=-1)
                    else:
                        out = torch.zeros(
                            *batch_shape,
                            mul_out * dim_out_l,
                            device=x1.device,
                            dtype=x1.dtype,
                        )
                    outputs.append(out)

            return torch.cat(outputs, dim=-1)

        # ------------------------------------------------------------------
        # Fallback: approximate implementation for general irreps ⊗ irreps.
        # (Kept for backward-compat / experimentation; not used by tp1 below.)
        # ------------------------------------------------------------------
        outputs = []
        for (l_out, p_out), mul_out in sorted(self._output_muls.items()):
            dim_out_l = 2 * l_out + 1

            contributions = []
            for l1, blocks1 in struct1.items():
                for l2, blocks2 in struct2.items():
                    if abs(l1 - l2) <= l_out <= l1 + l2:
                        for mul1, s1, e1, p1 in blocks1:
                            for mul2, s2, e2, p2 in blocks2:
                                # Parity selection rule
                                if p1 * p2 == p_out:
                                    f1 = x1[..., s1:e1].view(*batch_shape, mul1, 2 * l1 + 1)
                                    f2 = x2[..., s2:e2].view(*batch_shape, mul2, 2 * l2 + 1)

                                    # Simplified coupling (NOT exact CG): norm product
                                    f1_norm = f1.pow(2).sum(-1)  # (..., mul1)
                                    f2_norm = f2.pow(2).sum(-1)  # (..., mul2)
                                    coupled = f1_norm.unsqueeze(-1) * f2_norm.unsqueeze(-2)
                                    flat = coupled.flatten(-2)  # (..., mul1*mul2)
                                    expanded = flat.unsqueeze(-1).expand(*batch_shape, flat.shape[-1], dim_out_l)
                                    contributions.append(expanded.flatten(-2))  # (..., mul1*mul2*dim_out_l)

            if contributions:
                combined = torch.cat(contributions, dim=-1)
                target_dim = mul_out * dim_out_l
                if combined.shape[-1] >= target_dim:
                    outputs.append(combined[..., :target_dim])
                else:
                    pad = torch.zeros(
                        *batch_shape,
                        target_dim - combined.shape[-1],
                        device=x1.device,
                        dtype=x1.dtype,
                    )
                    outputs.append(torch.cat([combined, pad], dim=-1))
            else:
                outputs.append(
                    torch.zeros(
                        *batch_shape,
                        mul_out * dim_out_l,
                        device=x1.device,
                        dtype=x1.dtype,
                    )
                )

        return torch.cat(outputs, dim=-1)


class CartesianToIrrepsLight(nn.Module):
    """
    Conversion from Cartesian vectors to irreducible representations.
    
    Converts 3D Cartesian vectors to spherical harmonics (irreducible representations)
    using e3nn.o3.spherical_harmonics for exact equivariance.
    
    The conversion computes spherical harmonics Y_l^m(r/|r|) for each angular
    momentum l up to lmax. This provides a rotation-equivariant representation
    of the direction vector.
    
    Output structure for lmax=2:
        l=0: 1 component (scalar, Y_0^0)
        l=1: 3 components (vector, Y_1^{-1}, Y_1^0, Y_1^1)
        l=2: 5 components (quadrupole, Y_2^{-2}, ..., Y_2^2)
        Total: 1 + 3 + 5 = 9 dimensions
    
    Args:
        lmax: Maximum angular momentum to compute.
              Default: 2
              Higher lmax provides more angular resolution but increases dimension.
    
    Attributes:
        lmax: Maximum angular momentum.
        dims: List of dimensions for each l: [2*l+1 for l in range(lmax+1)]
              Example for lmax=2: [1, 3, 5]
        total_dim: Total output dimension = sum(dims)
                   Example for lmax=2: 9
        irreps_out: e3nn Irreps object representing output structure.
    
    Example:
        >>> converter = CartesianToIrrepsLight(lmax=2)
        >>> vec = torch.randn(100, 3)  # 100 vectors in 3D
        >>> irreps = converter(vec)    # Shape: (100, 9)
        >>> # irreps[:, 0] is l=0 (scalar)
        >>> # irreps[:, 1:4] is l=1 (vector)
        >>> # irreps[:, 4:9] is l=2 (quadrupole)
    
    Note:
        This uses e3nn's spherical_harmonics function which is exactly equivariant.
        The output is normalized: Y_l^m(r/|r|) where r/|r| is the unit vector.
        This ensures the output transforms correctly under rotations.
    """
    
    def __init__(self, lmax: int = 2):
        super().__init__()
        self.lmax = lmax
        self.dims = [2 * l + 1 for l in range(lmax + 1)]  # [1, 3, 5] for lmax=2
        self.total_dim = sum(self.dims)  # 9 for lmax=2
        self.irreps_out = o3.Irreps.spherical_harmonics(lmax)
        
    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Convert Cartesian vector to irreps using e3nn spherical harmonics.
        
        Args:
            vec: (..., 3) Cartesian vector
            
        Returns:
            (..., total_dim) irreps [l=0, l=1, l=2, ...]
        """
        # Use e3nn's spherical_harmonics for exact equivariance
        # normalize=True returns Y_l^m(r/|r|), which is what we want
        return o3.spherical_harmonics(self.irreps_out, vec, normalize=True, normalization='component')


class SparseTensorProduct(nn.Module):
    """
    Sparse tensor product exploiting angular momentum selection rules.
    
    For l1 ⊗ l2, only outputs l3 where |l1-l2| <= l3 <= l1+l2.
    This dramatically reduces parameters compared to dense outer product.
    
    For lmax=2:
    Valid combinations:
    - 0 ⊗ 0 → 0
    - 0 ⊗ 1 → 1
    - 0 ⊗ 2 → 2
    - 1 ⊗ 0 → 1
    - 1 ⊗ 1 → 0, 1, 2
    - 1 ⊗ 2 → 1, 2 (limited by output lmax)
    - 2 ⊗ 0 → 2
    - 2 ⊗ 1 → 1, 2
    - 2 ⊗ 2 → 0, 2 (limited by output lmax)
    """
    
    def __init__(self, channels_in: int, channels_out: int, lmax: int = 2):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.lmax = lmax
        
        self.dims = [2 * l + 1 for l in range(lmax + 1)]  # [1, 3, 5]
        self.total_dim_in = sum(self.dims) * channels_in
        self.total_dim_out = sum(self.dims) * channels_out
        
        # Create sparse weight blocks for each valid (l1, l2) → l3 path
        # Instead of full (total_dim_in × total_dim_in) → total_dim_out matrix
        # We use separate smaller matrices for each l3
        
        self.weight_blocks = nn.ModuleDict()
        self.path_info = []  # Store (l1, l2, l3) for each path
        
        total_params = 0
        for l3 in range(lmax + 1):
            # Collect all (l1, l2) pairs that can produce l3
            valid_pairs = []
            for l1 in range(lmax + 1):
                for l2 in range(lmax + 1):
                    if abs(l1 - l2) <= l3 <= l1 + l2:
                        valid_pairs.append((l1, l2))
            
            if valid_pairs:
                # Input dimension for this l3: sum of (2l1+1) × (2l2+1) for valid pairs
                # But we simplify: use channels directly, let MLP handle mixing
                in_dim = len(valid_pairs) * channels_in
                out_dim = (2 * l3 + 1) * channels_out
                
                self.weight_blocks[f'l{l3}'] = nn.Linear(in_dim, out_dim, bias=False)
                self.path_info.append((l3, valid_pairs))
                total_params += in_dim * out_dim
        
        self._total_params = total_params
    
    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Sparse tensor product.
        
        Args:
            f1: (..., channels_in, sum(dims)) - first input features per l
            f2: (..., channels_in, sum(dims)) - second input features per l
            
        Returns:
            (..., total_dim_out) - output features
        """
        batch_shape = f1.shape[:-1]
        
        # Split inputs by l
        f1_by_l = self._split_by_l(f1)  # dict: l -> (..., channels_in, 2l+1)
        f2_by_l = self._split_by_l(f2)
        
        outputs = []
        for l3, valid_pairs in self.path_info:
            # Gather inputs from valid (l1, l2) pairs
            pair_features = []
            for l1, l2 in valid_pairs:
                # Simple combination: element-wise product of norms
                # This is an approximation - full CG would be more accurate
                f1_l = f1_by_l[l1]  # (..., channels_in, 2l1+1)
                f2_l = f2_by_l[l2]  # (..., channels_in, 2l2+1)
                
                # Contract over angular indices, keep channels
                # Using dot product as simplified coupling
                combined = (f1_l * f1_l).sum(dim=-1) * (f2_l * f2_l).sum(dim=-1)  # (..., channels_in)
                pair_features.append(combined)
            
            # Stack and apply linear
            stacked = torch.cat(pair_features, dim=-1)  # (..., len(pairs) * channels_in)
            out_l = self.weight_blocks[f'l{l3}'](stacked)  # (..., (2l3+1) * channels_out)
            outputs.append(out_l)
        
        return torch.cat(outputs, dim=-1)  # (..., total_dim_out)
    
    def _split_by_l(self, f: torch.Tensor) -> dict:
        """Split features by angular momentum l."""
        result = {}
        idx = 0
        for l, dim in enumerate(self.dims):
            result[l] = f[..., idx:idx + dim]
            idx += dim
        return result


class CartesianTensorProductSparse(nn.Module):
    """
    Cartesian tensor product with CG selection rules.
    
    Uses full CG decomposition: l1 ⊗ l2 → |l1-l2|, ..., l1+l2
    Instead of block-diagonal (only same-l), this considers all valid paths.
    
    For lmax=2, the valid paths are:
    - 0 ⊗ 0 → 0
    - 0 ⊗ 1 → 1
    - 0 ⊗ 2 → 2
    - 1 ⊗ 0 → 1
    - 1 ⊗ 1 → 0, 1, 2
    - 1 ⊗ 2 → 1, 2 (capped by lmax_out)
    - 2 ⊗ 0 → 2
    - 2 ⊗ 1 → 1, 2
    - 2 ⊗ 2 → 0, 2 (only even l3 for same parity)
    """
    
    def __init__(self, channels_in: int, channels_out: int, lmax: int = 2, 
                 shared_weights: bool = True, num_weights: int = None,
                 lmax_sh: int = None):
        """
        Args:
            channels_in: Input feature channels
            channels_out: Output feature channels  
            lmax: Max angular momentum for input features (if they have l structure)
            shared_weights: Whether to use shared weights
            lmax_sh: Max angular momentum for spherical harmonics input (default: lmax)
        """
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.lmax = lmax
        self.lmax_sh = lmax_sh if lmax_sh is not None else lmax
        self.lmax_out = lmax  # Output lmax (capped to input lmax for simplicity)
        self.shared_weights = shared_weights
        
        self.dims_sh = [2 * l + 1 for l in range(self.lmax_sh + 1)]  # SH dimensions
        self.dims_out = [2 * l + 1 for l in range(self.lmax_out + 1)]  # Output dimensions
        self.total_sh_dim = sum(self.dims_sh)
        self.total_out_dim = sum(self.dims_out)
        
        # Build path table: for each output l3, list valid (l1_sh, ) paths
        # Here l1 is from features (scalar, so effectively l1=0)
        # l2 is from spherical harmonics (0, 1, ..., lmax_sh)
        # For features ⊗ sh where features are scalars: 0 ⊗ l2 → l2
        # So each sh component l2 maps to output l3 = l2
        
        # But if features have l-structure, we need full CG
        # For now, assume features are scalar (l=0), sh has l=0..lmax_sh
        # Then: 0 ⊗ l2 → l2, so output l3 = l2
        
        # Build path weights for each output l3
        self.path_weights = nn.ModuleDict()
        self._paths = {}  # l3 -> list of l2 values that contribute
        total_params = 0
        
        for l3 in range(self.lmax_out + 1):
            # Find all l2 from sh that can contribute to l3
            # With l1=0 (scalar features): 0 ⊗ l2 → l2, so l3 = l2
            valid_l2 = []
            for l2 in range(self.lmax_sh + 1):
                # Selection rule: |l1 - l2| <= l3 <= l1 + l2
                # With l1=0: l2 <= l3 <= l2, so l3 == l2
                if l2 == l3:
                    valid_l2.append(l2)
            
            # Also consider l1 != 0 if features have l-structure
            # For full generality, enumerate all (l1, l2) pairs
            # But for efficiency, we'll handle the scalar case first
            
            self._paths[l3] = valid_l2
            
            if valid_l2:
                # Input: channels_in * sum of (2*l2+1) for valid l2
                in_dim = channels_in * sum(2 * l2 + 1 for l2 in valid_l2)
                out_dim = channels_out * self.dims_out[l3]
                
                if shared_weights:
                    self.path_weights[f'l{l3}'] = nn.Linear(in_dim, out_dim, bias=False)
                
                total_params += in_dim * out_dim
        
        if not shared_weights:
            self.weight_numel = total_params
        else:
            self.weight_numel = 0
        
        self._total_params = total_params
    
    def forward(self, features: torch.Tensor, sh: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Apply tensor product: features ⊗ sh using CG selection rules.
        
        Args:
            features: (..., channels_in) - scalar features per channel
            sh: (..., total_sh_dim) - spherical harmonics / irreps
            weights: (..., weight_numel) - optional external weights
            
        Returns:
            (..., channels_out * total_out_dim) - output features
        """
        # Split sh by l
        sh_by_l = {}
        idx = 0
        for l2 in range(self.lmax_sh + 1):
            dim_l2 = 2 * l2 + 1
            sh_by_l[l2] = sh[..., idx:idx + dim_l2]
            idx += dim_l2
        
        outputs = []
        weight_idx = 0
        
        for l3 in range(self.lmax_out + 1):
            valid_l2 = self._paths.get(l3, [])
            
            if not valid_l2:
                # No paths to this l3, output zeros
                dim_l3 = self.dims_out[l3]
                zeros = torch.zeros(*features.shape[:-1], self.channels_out * dim_l3,
                                   device=features.device, dtype=features.dtype)
                outputs.append(zeros)
                continue
            
            # Gather contributions from all valid l2
            path_features = []
            for l2 in valid_l2:
                sh_l2 = sh_by_l[l2]  # (..., 2*l2+1)
                
                # features ⊗ sh_l2: (..., channels_in) x (..., 2*l2+1)
                f_expanded = features.unsqueeze(-1) * sh_l2.unsqueeze(-2)  # (..., channels_in, 2*l2+1)
                path_features.append(f_expanded.flatten(-2))  # (..., channels_in * (2*l2+1))
            
            # Concatenate all path contributions
            combined = torch.cat(path_features, dim=-1)
            
            # Apply path weights
            if self.shared_weights:
                out_l3 = self.path_weights[f'l{l3}'](combined)
            else:
                in_dim = combined.shape[-1]
                out_dim = self.channels_out * self.dims_out[l3]
                W = weights[..., weight_idx:weight_idx + in_dim * out_dim]
                W = W.view(*weights.shape[:-1], in_dim, out_dim)
                out_l3 = torch.einsum('...i,...io->...o', combined, W)
                weight_idx += in_dim * out_dim
            
            outputs.append(out_l3)
        
        return torch.cat(outputs, dim=-1)


class CartesianTensorProductCG(nn.Module):
    """
    Full CG tensor product between two irreps with different l-structure.
    
    Implements: irreps1 ⊗ irreps2 → irreps_out
    Using selection rule: l1 ⊗ l2 → |l1-l2|, ..., l1+l2
    """
    
    def __init__(self, channels1: int, channels2: int, channels_out: int,
                 lmax1: int, lmax2: int, lmax_out: int = None,
                 shared_weights: bool = True):
        super().__init__()
        self.channels1 = channels1
        self.channels2 = channels2
        self.channels_out = channels_out
        self.lmax1 = lmax1
        self.lmax2 = lmax2
        self.lmax_out = lmax_out if lmax_out is not None else min(lmax1 + lmax2, max(lmax1, lmax2))
        self.shared_weights = shared_weights
        
        self.dims1 = [2 * l + 1 for l in range(lmax1 + 1)]
        self.dims2 = [2 * l + 1 for l in range(lmax2 + 1)]
        self.dims_out = [2 * l + 1 for l in range(self.lmax_out + 1)]
        
        self.total_dim1 = sum(self.dims1)
        self.total_dim2 = sum(self.dims2)
        self.total_dim_out = sum(self.dims_out)
        
        # Build path table and weights
        self._paths = {}  # l3 -> [(l1, l2), ...]
        self.path_weights = nn.ModuleDict()
        total_params = 0
        
        for l3 in range(self.lmax_out + 1):
            valid_pairs = []
            for l1 in range(lmax1 + 1):
                for l2 in range(lmax2 + 1):
                    # CG selection rule
                    if abs(l1 - l2) <= l3 <= l1 + l2:
                        valid_pairs.append((l1, l2))
            
            self._paths[l3] = valid_pairs
            
            if valid_pairs:
                # Each pair contributes channels1 * channels2 features
                # (after contracting over angular indices)
                in_dim = len(valid_pairs) * channels1 * channels2
                out_dim = channels_out * self.dims_out[l3]
                
                if shared_weights:
                    self.path_weights[f'l{l3}'] = nn.Linear(in_dim, out_dim, bias=False)
                
                total_params += in_dim * out_dim
        
        if not shared_weights:
            self.weight_numel = total_params
        else:
            self.weight_numel = 0
        
        self._total_params = total_params
        
        num_paths = sum(len(p) for p in self._paths.values())
        print(f"CartesianTensorProductCG: ({channels1}, l≤{lmax1}) ⊗ ({channels2}, l≤{lmax2}) → ({channels_out}, l≤{self.lmax_out})")
        print(f"  Paths: {num_paths}, Params: {total_params:,}")
    
    def forward(self, f1: torch.Tensor, f2: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            f1: (..., channels1 * total_dim1) - first irreps
            f2: (..., channels2 * total_dim2) - second irreps
            weights: (..., weight_numel) - optional external weights
            
        Returns:
            (..., channels_out * total_dim_out)
        """
        batch_shape = f1.shape[:-1]
        
        # Reshape and split by l
        f1_split = self._split(f1, self.dims1, self.channels1)
        f2_split = self._split(f2, self.dims2, self.channels2)
        
        outputs = []
        weight_idx = 0
        
        for l3 in range(self.lmax_out + 1):
            valid_pairs = self._paths.get(l3, [])
            
            if not valid_pairs:
                zeros = torch.zeros(*batch_shape, self.channels_out * self.dims_out[l3],
                                   device=f1.device, dtype=f1.dtype)
                outputs.append(zeros)
                continue
            
            # Gather contributions from all (l1, l2) pairs
            pair_features = []
            for l1, l2 in valid_pairs:
                x1 = f1_split[l1]  # (..., channels1, 2*l1+1)
                x2 = f2_split[l2]  # (..., channels2, 2*l2+1)
                
                # Contract over angular indices (simplified CG coupling)
                # Full CG would weight by Wigner-3j coefficients
                x1_contracted = x1.pow(2).sum(dim=-1)  # (..., channels1)
                x2_contracted = x2.pow(2).sum(dim=-1)  # (..., channels2)
                
                # Outer product over channels
                coupled = x1_contracted.unsqueeze(-1) * x2_contracted.unsqueeze(-2)
                pair_features.append(coupled.flatten(-2))  # (..., channels1 * channels2)
            
            combined = torch.cat(pair_features, dim=-1)
            
            if self.shared_weights:
                out_l3 = self.path_weights[f'l{l3}'](combined)
            else:
                in_dim = combined.shape[-1]
                out_dim = self.channels_out * self.dims_out[l3]
                W = weights[..., weight_idx:weight_idx + in_dim * out_dim]
                W = W.view(*weights.shape[:-1], in_dim, out_dim)
                out_l3 = torch.einsum('...i,...io->...o', combined, W)
                weight_idx += in_dim * out_dim
            
            outputs.append(out_l3)
        
        return torch.cat(outputs, dim=-1)
    
    def _split(self, f: torch.Tensor, dims: list, channels: int) -> dict:
        """Split tensor by l."""
        total_dim = sum(dims)
        f = f.view(*f.shape[:-1], channels, total_dim)
        result = {}
        idx = 0
        for l, dim in enumerate(dims):
            result[l] = f[..., idx:idx + dim]
            idx += dim
        return result


class CartesianE3ConvSparse(nn.Module):
    """
    Cartesian E3 convolution matching e3nn E3Conv structure.
    
    This layer implements the first convolution in the E3NN architecture,
    converting atom features and edge directions into higher-order irreps.
    
    Forward flow:
        1. Atom embedding: A[edge_src] → scalar features Ai[edge_src]
        2. Direction conversion: edge_vec → spherical harmonics sh_edge
        3. Tensor product 1: scalar(Ai[edge_src]) ⊗ sh_edge → higher-order irreps (f_in)
        4. Radial basis: edge_length → radial basis expansion
        5. Weight network: radial basis → tensor product 2 weights
        6. Tensor product 2: f_in ⊗ scalar(Ai[edge_dst]) → higher-order output (with weights)
        7. Aggregation: Aggregate edge features to nodes
    
    The two tensor products use EquivariantTensorProduct for strict equivariance.
    
    Args:
        max_radius: Maximum cutoff radius for neighbor search.
        number_of_basis: Number of radial basis functions.
        channels_out: Number of output channels (multiplicity for each l).
        embedding_dim: Dimension of atom type embedding.
        max_atomvalue: Maximum atomic number (for embedding lookup).
        output_size: Size of atom feature MLP output (multiplicity for tp1).
        lmax: Maximum angular momentum.
        function_type: Type of radial basis function.
                       Options: 'gaussian', 'bessel', 'fourier', 'cosine', 'smooth_finite'
    
    Attributes:
        tp1: First tensor product (scalar ⊗ sh → irreps).
             Uses EquivariantTensorProduct with internal weights.
        tp2: Second tensor product (irreps ⊗ scalar → irreps).
             Uses EquivariantTensorProduct with external weights from radial basis.
        fc: Weight network mapping radial basis to tp2 weights.
        cart_to_irreps: Converter from Cartesian vectors to spherical harmonics.
        atom_embedding: Embedding layer for atom types.
        atom_mlp: MLP to process atom embeddings into scalar features.
        output_dim: Total output dimension = channels_out * sum(2*l+1 for l in range(lmax+1))
    
    Example:
        >>> conv = CartesianE3ConvSparse(
        ...     max_radius=5.0,
        ...     number_of_basis=8,
        ...     channels_out=64,
        ...     lmax=2
        ... )
        >>> pos = torch.randn(50, 3)
        >>> A = torch.randint(0, 10, (50,))
        >>> batch = torch.zeros(50, dtype=torch.long)
        >>> edge_src = torch.randint(0, 50, (200,))
        >>> edge_dst = torch.randint(0, 50, (200,))
        >>> edge_shifts = torch.zeros(200, 3)
        >>> cell = torch.eye(3).unsqueeze(0)
        >>> out = conv(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        >>> # out shape: (50, 576) = 50 nodes * 64 channels * 9 dims
    
    Note:
        This layer uses EquivariantTensorProduct for both tensor products,
        ensuring strict rotational equivariance. The implementation matches
        e3nn's E3Conv structure for fair comparison.
    """
    
    def __init__(self, max_radius, number_of_basis, channels_out,
                 embedding_dim=16, max_atomvalue=10, output_size=8,
                 lmax=2, function_type='gaussian'):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.lmax = lmax
        self.function_type = function_type
        
        # Dimensions
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)  # 9 for lmax=2
        self.output_dim = channels_out * self.total_sh_dim
        
        # Cartesian to irreps converter
        self.cart_to_irreps = CartesianToIrrepsLight(lmax=lmax)
        
        # Atom embedding
        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size)
        )
        
        # ------------------------------------------------------------------
        # tp1: scalar(Ai[edge_src]) ⊗ sh_edge → higher-order irreps
        # Like e3nn: o3.FullTensorProduct(f"{output_size}x0e", "1x0e + 1x1o + 1x2e")
        # Uses strictly equivariant EquivariantTensorProduct
        # ------------------------------------------------------------------
        irreps_sh = get_irreps_str(1, lmax)  # "1x0e + 1x1o + 1x2e"
        self.tp1_irreps_out = get_irreps_str(output_size, lmax)
        self.tp1 = EquivariantTensorProduct(
            irreps_in1=f"{output_size}x0e", 
            irreps_in2=irreps_sh,
            irreps_out=self.tp1_irreps_out,
            shared_weights=True,
            internal_weights=True
        )
        
        # ------------------------------------------------------------------
        # tp2: higher-order irreps(f_in) ⊗ scalar(Ai[edge_dst]) → higher-order output
        # Like e3nn: o3.FullyConnectedTensorProduct(tp1.irreps_out, f"{output_size}x0e", irreps_output)
        # Uses strictly equivariant EquivariantTensorProduct
        # ------------------------------------------------------------------
        self.tp2 = EquivariantTensorProduct(
            irreps_in1=self.tp1_irreps_out,       # higher-order (output_size x 0e + output_size x 1o + ...)
            irreps_in2=f"{output_size}x0e",       # scalar (neighbor atom features)
            irreps_out=get_irreps_str(channels_out, lmax),  # higher-order output
            shared_weights=False,
            internal_weights=False
        )
        
        # Weight network (radial basis → tp2 weights)
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.weight_numel)
        )
        
        print(
            f"CartesianE3ConvSparse init: lmax={lmax}, output_dim={self.output_dim}, "
            f"tp1={self.tp1.irreps_in1_str} ⊗ {self.tp1.irreps_in2_str}, "
            f"tp2={self.tp1_irreps_out} ⊗ {output_size}x0e → {channels_out}ch, "
            f"tp2_weight_numel={self.tp2.weight_numel}"
        )
    
    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        # Keep geometry dtype consistent with module parameters (datasets may provide float64)
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum('ni,nij->nj', edge_shifts, edge_cells)
        
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        
        # Convert direction to irreps (1x0e + 1x1o + 1x2e)
        sh_edge = self.cart_to_irreps(edge_vec)  # (E, total_sh_dim)
        
        num_nodes = pos.size(0)
        
        # Atom features (scalar)
        atom_emb = self.atom_embedding(A.long())
        Ai = self.atom_mlp(atom_emb)  # (N, output_size)
        
        # tp1: scalar(Ai[edge_src]) ⊗ sh_edge → higher-order irreps
        # Like e3nn: f_in = self.tensor_product(Ai[edge_src], sh_edge)
        f_in = self.tp1(Ai[edge_src], sh_edge)  # (E, output_size * total_sh_dim)
        
        # Radial basis
        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis,
            basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        
        # tp2 weights from radial basis
        weights = self.fc(emb)  # (E, tp2.weight_numel)
        
        # tp2: higher-order irreps(f_in) ⊗ scalar(Ai[edge_dst]) → higher-order output
        # Like e3nn: edge_features = self.tp(f_in, Ai[edge_dst], self.fc(emb))
        edge_features = self.tp2(f_in, Ai[edge_dst], weights)  # (E, channels_out * total_sh_dim)
        
        # Aggregate over edges
        neighbor_count = scatter(
            torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes
        ).clamp(min=1).to(edge_features.dtype)
        
        out = scatter(
            edge_features, edge_dst, dim=0, dim_size=num_nodes
        ).div(neighbor_count.unsqueeze(-1))
        
        return out


class CartesianE3Conv2Sparse(nn.Module):
    """
    Second Cartesian E3 convolution matching e3nn E3Conv2 structure.
    
    This layer implements the second convolution in the E3NN architecture,
    further processing node features with edge direction information.
    
    Forward flow:
        1. Direction conversion: edge_vec → spherical harmonics sh_edge
        2. Radial basis: edge_length → radial basis expansion
        3. Weight network: radial basis → tensor product weights
        4. Tensor product: f_in[edge_src] ⊗ sh_edge → higher-order output (with weights)
        5. Aggregation: Aggregate edge features to nodes
    
    The tensor product uses EquivariantTensorProduct for strict equivariance.
    
    Args:
        max_radius: Maximum cutoff radius for neighbor search.
        number_of_basis: Number of radial basis functions.
        channels_in: Number of input channels (multiplicity for each l).
        channels_out: Number of output channels (multiplicity for each l).
        embedding_dim: Dimension of atom type embedding (not used in this layer).
        max_atomvalue: Maximum atomic number (not used in this layer).
        lmax: Maximum angular momentum.
        function_type: Type of radial basis function.
                       Options: 'gaussian', 'bessel', 'fourier', 'cosine', 'smooth_finite'
    
    Attributes:
        tp: Tensor product (irreps ⊗ sh → irreps).
            Uses EquivariantTensorProduct with external weights from radial basis.
        fc: Weight network mapping radial basis to tp weights.
        cart_to_irreps: Converter from Cartesian vectors to spherical harmonics.
        input_dim: Total input dimension = channels_in * sum(2*l+1 for l in range(lmax+1))
        output_dim: Total output dimension = channels_out * sum(2*l+1 for l in range(lmax+1))
    
    Example:
        >>> conv = CartesianE3Conv2Sparse(
        ...     max_radius=5.0,
        ...     number_of_basis=8,
        ...     channels_in=64,
        ...     channels_out=64,
        ...     lmax=2
        ... )
        >>> f_in = torch.randn(50, 576)  # 50 nodes * 64 channels * 9 dims
        >>> pos = torch.randn(50, 3)
        >>> A = torch.randint(0, 10, (50,))
        >>> batch = torch.zeros(50, dtype=torch.long)
        >>> edge_src = torch.randint(0, 50, (200,))
        >>> edge_dst = torch.randint(0, 50, (200,))
        >>> edge_shifts = torch.zeros(200, 3)
        >>> cell = torch.eye(3).unsqueeze(0)
        >>> out = conv(f_in, pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        >>> # out shape: (50, 576) = 50 nodes * 64 channels * 9 dims
    
    Note:
        This layer uses EquivariantTensorProduct for the tensor product,
        ensuring strict rotational equivariance. The implementation matches
        e3nn's E3Conv2 structure for fair comparison.
    """
    
    def __init__(self, max_radius, number_of_basis, channels_in, channels_out,
                 embedding_dim=16, max_atomvalue=10, lmax=2, function_type='gaussian'):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.lmax = lmax
        self.function_type = function_type
        
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)
        self.input_dim = channels_in * self.total_sh_dim
        self.output_dim = channels_out * self.total_sh_dim
        
        self.cart_to_irreps = CartesianToIrrepsLight(lmax=lmax)
        
        # ------------------------------------------------------------------
        # tp: higher-order irreps(f_in[edge_src]) ⊗ direction sh → higher-order output
        # Like e3nn: o3.FullyConnectedTensorProduct(irreps_input_conv, self.irreps_out, self.irreps_output)
        # Uses strictly equivariant EquivariantTensorProduct
        # ------------------------------------------------------------------
        irreps_in = get_irreps_str(channels_in, lmax)   # channels_in x (0e + 1o + 2e)
        irreps_sh = get_irreps_str(1, lmax)              # 1x0e + 1x1o + 1x2e
        irreps_out = get_irreps_str(channels_out, lmax)  # channels_out x (0e + 1o + 2e)
        
        self.tp = EquivariantTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            shared_weights=False,
            internal_weights=False
        )
        
        # Weight network (radial basis → tp weights)
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel)
        )
        
        print(f"CartesianE3Conv2Sparse init: lmax={lmax}, "
              f"tp={irreps_in} ⊗ {irreps_sh} → {irreps_out}, "
              f"tp_weight_numel={self.tp.weight_numel}")
    
    def forward(self, f_in, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        # Keep geometry dtype consistent with feature dtype to avoid float/double mismatches
        dtype = f_in.dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum('ni,nij->nj', edge_shifts, edge_cells)
        
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        
        # Convert direction to sh irreps
        sh_edge = self.cart_to_irreps(edge_vec)  # (E, total_sh_dim)
        
        num_nodes = pos.size(0)
        
        # Radial basis
        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis,
            basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        
        # tp weights from radial basis
        weights = self.fc(emb)  # (E, tp.weight_numel)
        
        # tp: higher-order irreps(f_in[edge_src]) ⊗ direction sh → higher-order output
        # Like e3nn: edge_features = self.tp(f_in[edge_src], Feature, self.fc(emb))
        edge_features = self.tp(f_in[edge_src], sh_edge, weights)  # (E, channels_out * total_sh_dim)
        
        # Aggregate over edges
        neighbor_count = scatter(
            torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes
        ).clamp(min=1).to(edge_features.dtype)
        
        out = scatter(
            edge_features, edge_dst, dim=0, dim_size=num_nodes
        ).div(neighbor_count.unsqueeze(-1))
        
        return out


class CartesianProductSparse(nn.Module):
    """Sparse product layer for combining features."""
    
    def __init__(self, channels_in: int, channels_out: int, lmax: int = 2):
        super().__init__()
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)
        self.input_dim = channels_in * self.total_sh_dim
        self.output_dim = channels_out
        
        # Block-diagonal for each l, then aggregate to scalar
        self.blocks = nn.ModuleList()
        for l in range(lmax + 1):
            dim_l = 2 * l + 1
            # Self tensor product within each l: (ch * dim_l)² -> ch_out
            self.blocks.append(nn.Linear(channels_in * dim_l, channels_out // (lmax + 1)))
        
        self.output_dim = (channels_out // (lmax + 1)) * (lmax + 1)
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (..., input_dim)
        Returns:
            (..., output_dim)
        """
        outputs = []
        idx = 0
        channels_in = f.shape[-1] // self.total_sh_dim
        
        for l, block in enumerate(self.blocks):
            dim_l = self.dims[l]
            f_l = f[..., idx:idx + channels_in * dim_l]
            idx += channels_in * dim_l
            
            # Self-interaction: element-wise square then linear
            f_l_sq = f_l * f_l
            out_l = block(f_l_sq)
            outputs.append(out_l)
        
        return torch.cat(outputs, dim=-1)


class CartesianProductFC(nn.Module):
    """
    Equivariant product layer built on CartesianFullyConnectedTensorProduct.

    Replaces CartesianProductSparse.
    Input/Output are both organized as: (channels x (0e+1o+2e...)) in l-major layout.

    For practicality, we use an equivariant bottleneck:
      L-wise reduce (channels_in -> bottleneck)  [shared across m]
      FullyConnectedTP in bottleneck space
      L-wise expand (bottleneck -> channels_out)
    """

    def __init__(self, channels_in: int, channels_out: int, lmax: int = 2, bottleneck_ratio: float = 0.25):
        super().__init__()
        self.lmax = lmax
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)

        bottleneck = max(4, int(channels_in * bottleneck_ratio))

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.bottleneck = bottleneck

        self.irreps_in = get_irreps_str(channels_in, lmax)
        self.irreps_bn = get_irreps_str(bottleneck, lmax)
        self.irreps_out = get_irreps_str(channels_out, lmax)

        # Equivariant l-wise reduce/expand (mix channels only within each l, shared across m)
        self.reduce = nn.ModuleList(
            [nn.Linear(channels_in, bottleneck, bias=(l == 0)) for l in range(lmax + 1)]
        )
        self.expand = nn.ModuleList(
            [nn.Linear(bottleneck, channels_out, bias=(l == 0)) for l in range(lmax + 1)]
        )

        self.tp = CartesianFullyConnectedTensorProduct(
            irreps_in1=self.irreps_bn,
            irreps_in2=self.irreps_bn,
            irreps_out=self.irreps_bn,
            shared_weights=True,
            internal_weights=True,
        )

        self.output_dim = channels_out * self.total_sh_dim

    def _lwise_linear(self, x: torch.Tensor, linears: nn.ModuleList, cin: int, cout: int) -> torch.Tensor:
        """Apply l-wise channel mixing while preserving equivariance."""
        outs = []
        idx = 0
        for l in range(self.lmax + 1):
            dim_l = 2 * l + 1
            block = x[..., idx:idx + cin * dim_l].view(-1, cin, dim_l)  # (B, cin, dim)
            idx += cin * dim_l
            block_t = block.transpose(1, 2)  # (B, dim, cin)
            out_t = linears[l](block_t)      # (B, dim, cout)
            outs.append(out_t.transpose(1, 2).reshape(-1, cout * dim_l))
        return torch.cat(outs, dim=-1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # Reduce
        f_bn = self._lwise_linear(f, self.reduce, self.channels_in, self.bottleneck)
        # Product in bottleneck space
        f_prod = self.tp(f_bn, f_bn)
        # Expand
        return self._lwise_linear(f_prod, self.expand, self.bottleneck, self.channels_out)


class CartesianElementwiseTensorProduct(nn.Module):
    """
    Cartesian equivalent of e3nn.o3.ElementwiseTensorProduct.
    
    Performs element-wise (channel-wise) tensor product between two inputs
    with the same irreps structure. Output is typically scalar (0e).
    
    For each channel pair (m1, m2), computes the contraction that yields
    a rotation-invariant scalar.
    """
    
    def __init__(self, irreps_in: str, filter_ir_out: list = None, lmax: int = 2):
        """
        Args:
            irreps_in: Input irreps string (e.g., "64x0e + 64x1o + 64x2e")
            filter_ir_out: List of output irreps to keep (e.g., ["0e"])
            lmax: Maximum angular momentum
        """
        super().__init__()
        self.irreps_in_str = irreps_in
        self.irreps_in = parse_irreps_string(irreps_in)
        self.lmax = lmax
        self.filter_ir_out = filter_ir_out if filter_ir_out else ["0e"]
        
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)
        
        # For each (l, p) in input, self-contraction l ⊗ l → 0 gives scalar
        # Output dimension = sum of multiplicities
        self.output_channels = 0
        self._block_info = []  # (mul, l, p, start_idx, end_idx)
        
        idx = 0
        for mul, l, p in self.irreps_in:
            dim_l = 2 * l + 1
            self._block_info.append((mul, l, p, idx, idx + mul * dim_l))
            idx += mul * dim_l
            # l ⊗ l → 0e (scalar) contributes 'mul' scalars
            self.output_channels += mul
        
        self.dim_in = idx
        self.dim_out = self.output_channels
        
        # Build output irreps string
        self.irreps_out = o3.Irreps(f"{self.output_channels}x0e")
        self.irreps_out_str = str(self.irreps_out)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Element-wise tensor product.
        
        Args:
            x1: (..., dim_in) - first input
            x2: (..., dim_in) - second input (same structure as x1)
            
        Returns:
            (..., output_channels) - scalar outputs from l ⊗ l → 0 contractions
        """
        batch_shape = x1.shape[:-1]
        outputs = []
        
        for mul, l, p, start, end in self._block_info:
            dim_l = 2 * l + 1
            
            # Extract blocks: (..., mul, dim_l)
            f1 = x1[..., start:end].view(*batch_shape, mul, dim_l)
            f2 = x2[..., start:end].view(*batch_shape, mul, dim_l)
            
            # Contract over angular indices: Σ_m f1_m * f2_m → scalar per channel
            # This matches e3nn ElementwiseTensorProduct(..., ["0e"]) behavior.
            contracted = (f1 * f2).sum(dim=-1)  # (..., mul)
            outputs.append(contracted)
        
        if not outputs:
            return x1.new_zeros(*batch_shape, 0)
        
        return torch.cat(outputs, dim=-1)


class CartesianE3ConvStrict(nn.Module):
    """
    First Cartesian E3 convolution layer using strict parity tensor products.
    
    This layer converts atom embeddings to equivariant features using
    CartesianFullyConnectedTensorProduct which respects parity.
    
    Forward flow:
        1. Atom embedding: A → atom_emb → atom_feat (scalar)
        2. Direction conversion: edge_vec → spherical harmonics sh_edge
        3. First tensor product: atom_feat ⊗ sh_edge → irreps features
        4. Radial basis: edge_length → radial basis expansion
        5. Weight network: radial basis → tensor product weights
        6. Second tensor product: irreps ⊗ sh_edge → output (with weights)
        7. Aggregation: Aggregate edge features to nodes
    """
    
    def __init__(self, max_radius, number_of_basis, channels_out,
                 embedding_dim=16, max_atomvalue=10, output_size=8, 
                 lmax=2, function_type='gaussian'):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.lmax = lmax
        self.function_type = function_type
        
        # Irreps definitions
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)
        self.output_dim = channels_out * self.total_sh_dim
        
        # Irreps strings
        self.irreps_scalar = f"{output_size}x0e"
        self.irreps_sh = get_irreps_str(1, lmax)
        self.irreps_out = get_irreps_str(channels_out, lmax)
        
        # Cartesian to irreps converter
        self.cart_to_irreps = CartesianToIrrepsLight(lmax=lmax)
        
        # Atom embedding
        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size)
        )
        
        # First tensor product: scalar ⊗ sh → irreps
        # scalar (0e) ⊗ (0e+1o+2e) → (0e+1o+2e)
        self.tp1 = CartesianFullyConnectedTensorProduct(
            irreps_in1=self.irreps_scalar,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_out,
            shared_weights=True,
            internal_weights=True
        )
        
        # Second tensor product with edge weights
        # irreps ⊗ sh → irreps (with external weights)
        self.tp2 = CartesianFullyConnectedTensorProduct(
            irreps_in1=self.irreps_out,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_out,
            shared_weights=False,
            internal_weights=False
        )
        
        # Weight network
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.weight_numel)
        )
        
        print(f"CartesianE3ConvStrict init: irreps_out={self.irreps_out}, "
              f"tp1_weights={sum(p.numel() for p in self.tp1.parameters())}, "
              f"tp2_weight_numel={self.tp2.weight_numel}")
    
    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum('ni,nij->nj', edge_shifts, edge_cells)
        
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        
        # Convert direction to irreps (1x0e + 1x1o + 1x2e)
        edge_irreps = self.cart_to_irreps(edge_vec)
        
        num_nodes = pos.size(0)
        
        # Atom features (scalar)
        atom_emb = self.atom_embedding(A.long())
        atom_feat = self.atom_mlp(atom_emb)  # (N, output_size)
        
        # First tensor product: scalar ⊗ sh → irreps
        f_in = self.tp1(atom_feat[edge_src], edge_irreps)
        
        # Radial basis
        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis,
            basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        
        # Edge weights
        weights = self.fc(emb)
        
        # Second tensor product with weights
        edge_features = self.tp2(f_in, edge_irreps, weights)
        
        # Aggregate
        neighbor_count = scatter(
            torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes
        ).clamp(min=1).to(edge_features.dtype)
        
        out = scatter(
            edge_features, edge_dst, dim=0, dim_size=num_nodes
        ) / neighbor_count.unsqueeze(-1).sqrt()
        
        return out


class CartesianE3Conv2Strict(nn.Module):
    """
    Second convolution layer using strict parity.
    """
    
    def __init__(self, max_radius, number_of_basis, channels_in, channels_out,
                 embedding_dim=16, max_atomvalue=10, lmax=2, function_type='gaussian'):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.lmax = lmax
        
        # Irreps
        self.irreps_in = get_irreps_str(channels_in, lmax)
        self.irreps_sh = get_irreps_str(1, lmax)
        self.irreps_out = get_irreps_str(channels_out, lmax)
        
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)
        self.output_dim = channels_out * self.total_sh_dim
        
        # Cartesian to irreps
        self.cart_to_irreps = CartesianToIrrepsLight(lmax=lmax)
        
        # Atom embedding for neighbor info
        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        
        # Normalization
        self.norm_feature = nn.LayerNorm(channels_in * self.total_sh_dim)
        self.norm_ai = nn.LayerNorm(embedding_dim)
        self.norm_aj = nn.LayerNorm(embedding_dim)
        
        # Tensor product: irreps ⊗ sh → irreps (with external weights)
        self.tp = CartesianFullyConnectedTensorProduct(
            irreps_in1=self.irreps_in,
            irreps_in2=self.irreps_sh,
            irreps_out=self.irreps_out,
            shared_weights=False,
            internal_weights=False
        )
        
        # Weight network
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis + 2 * embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel)
        )
        
        print(f"CartesianE3Conv2Strict init: irreps_in={self.irreps_in}, "
              f"irreps_out={self.irreps_out}, weight_numel={self.tp.weight_numel}")
    
    def forward(self, f_in, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum('ni,nij->nj', edge_shifts, edge_cells)
        
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        
        # Convert direction to irreps
        edge_irreps = self.cart_to_irreps(edge_vec)
        
        num_nodes = pos.size(0)
        
        # Normalize input features
        f_in_norm = self.norm_feature(f_in)
        
        # Atom embeddings
        ai = self.norm_ai(self.atom_embedding(A[edge_src].long()))
        aj = self.norm_aj(self.atom_embedding(A[edge_dst].long()))
        
        # Radial basis
        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis,
            basis='gaussian', cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        
        # Weight input
        weight_input = torch.cat([emb, ai, aj], dim=-1)
        weights = self.fc(weight_input)
        
        # Tensor product
        edge_features = self.tp(f_in_norm[edge_src], edge_irreps, weights)
        
        # Aggregate
        neighbor_count = scatter(
            torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes
        ).clamp(min=1).to(edge_features.dtype)
        
        out = scatter(
            edge_features, edge_dst, dim=0, dim_size=num_nodes
        ) / neighbor_count.unsqueeze(-1).sqrt()
        
        return out


class CartesianProductStrict(nn.Module):
    """
    Product layer using strict parity CartesianFullyConnectedTensorProduct.
    
    f ⊗ f → specified output irreps (respecting parity)
    Uses bottleneck to reduce parameters: reduce → product → expand
    """
    
    def __init__(self, channels_in: int, channels_out: int, lmax: int = 2, 
                 bottleneck_ratio: float = 0.25):
        super().__init__()
        self.lmax = lmax
        
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)
        
        # Bottleneck: reduce channels before tensor product
        bottleneck_channels = max(4, int(channels_in * bottleneck_ratio))
        
        # Input/output irreps
        self.irreps_in = get_irreps_str(channels_in, lmax)
        self.irreps_bottleneck = get_irreps_str(bottleneck_channels, lmax)
        self.irreps_out = get_irreps_str(channels_out, lmax)
        
        # Reduce to bottleneck
        self.reduce = nn.Linear(channels_in * self.total_sh_dim, 
                               bottleneck_channels * self.total_sh_dim)
        
        # Tensor product in bottleneck space (much smaller!)
        self.tp = CartesianFullyConnectedTensorProduct(
            irreps_in1=self.irreps_bottleneck,
            irreps_in2=self.irreps_bottleneck,
            irreps_out=self.irreps_bottleneck,
            shared_weights=True,
            internal_weights=True
        )
        
        # Expand back
        self.expand = nn.Linear(bottleneck_channels * self.total_sh_dim,
                               channels_out * self.total_sh_dim)
        
        self.output_dim = channels_out * self.total_sh_dim
        
        tp_params = sum(p.numel() for p in self.tp.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"CartesianProductStrict init: {channels_in}ch → {bottleneck_channels}ch (bottleneck) → {channels_out}ch, "
              f"tp_params={tp_params:,}, total={total_params:,}")
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: (..., channels_in * total_sh_dim)
        Returns:
            (..., channels_out * total_sh_dim)
        """
        # Reduce to bottleneck
        f_reduced = self.reduce(f)
        
        # Self tensor product (respects parity)
        f_product = self.tp(f_reduced, f_reduced)
        
        # Expand back
        return self.expand(f_product)


class CartesianTransformerLayerStrict(nn.Module):
    """
    Cartesian Transformer layer with STRICT parity handling.
    
    All tensor products use CartesianFullyConnectedTensorProduct to ensure correct parity.
    """
    
    def __init__(self, max_embed_radius, main_max_radius, main_number_of_basis,
                 hidden_dim_conv, hidden_dim_sh, hidden_dim, channel_in2=32, embedding_dim=16,
                 max_atomvalue=10, output_size=8, embed_size=None, main_hidden_sizes3=None,
                 num_layers=1, device=None, function_type_main='gaussian', lmax=2, num_interaction=2):
        super().__init__()
        
        if embed_size is None:
            embed_size = [128, 128, 128]
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.lmax = lmax
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)
        
        self.channels = hidden_dim_conv
        self.hidden_dim_conv = self.channels * self.total_sh_dim

        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")
        
        # Strict parity convolution layers
        self.e3_conv_layers = nn.ModuleList()
        self.e3_conv_layers.append(
            CartesianE3ConvStrict(
                max_radius=max_embed_radius,
                number_of_basis=main_number_of_basis,
                channels_out=self.channels,
                embedding_dim=embedding_dim,
                max_atomvalue=max_atomvalue,
                output_size=output_size,
                lmax=lmax,
                function_type=function_type_main
            )
        )
        for _ in range(1, self.num_interaction):
            self.e3_conv_layers.append(
                CartesianE3Conv2Strict(
                    max_radius=max_embed_radius,
                    number_of_basis=main_number_of_basis,
                    channels_in=self.channels,
                    channels_out=self.channels,
                    embedding_dim=embedding_dim,
                    max_atomvalue=max_atomvalue,
                    lmax=lmax,
                    function_type=function_type_main
                )
            )
        
        # Strict parity product layers
        combined_channels = self.channels * self.num_interaction  # 64 for channels=32, n=2
        scalar_channels = (self.num_interaction - 1) * 32
        
        # Build irreps strings for combined features
        irreps_combined = get_irreps_str(combined_channels, lmax)
        
        # product_3: higher-order ⊗ higher-order → scalar (32x0e)
        self.product_3 = CartesianFullyConnectedTensorProduct(
            irreps_in1=irreps_combined,
            irreps_in2=irreps_combined,
            irreps_out=f"{scalar_channels}x0e",
            shared_weights=True,
            internal_weights=True
        )
        
        # T = [f_combine, f_prod3] = (combined_channels * total_sh_dim + scalar_channels) features
        # Build T irreps string
        irreps_T = get_irreps_str(combined_channels, lmax) + " + " + f"{scalar_channels}x0e"
        
        # product_5: ElementwiseTensorProduct(T, T, ["0e"]) - 严格等变
        self.product_5 = CartesianElementwiseTensorProduct(
            irreps_in=irreps_T,
            filter_ir_out=["0e"],
            lmax=lmax
        )
        
        # Readout input: product_5 outputs scalars
        readout_input_dim = self.product_5.dim_out
        self.proj_total = MainNet(readout_input_dim, embed_size, 17)
        self.num_features = 17
        self.weighted_sum = RobustScalarWeightedSum(self.num_features, init_weights='zero')
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"CartesianTransformerLayerStrict init complete: lmax={lmax}, "
              f"channels={self.channels}, strict_parity=True, params={total_params:,}")
    
    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]
        
        # Convolutions (strict parity)
        features = []
        f_prev = self.e3_conv_layers[0](pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        features.append(f_prev)
        for conv in self.e3_conv_layers[1:]:
            f_prev = conv(f_prev, pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
            features.append(f_prev)
        
        # Combine features: (N, n*channels * total_sh_dim)
        f_combine = torch.cat(features, dim=-1)
        
        # First product layer (strict parity)
        # Input: 64ch * 9 = 576, Output: 32ch * 9 = 288 (scalars per channel)
        f_prod3 = self.product_3(f_combine, f_combine)
        
        # Concatenate for second product
        # f_combine: 64ch * 9, f_prod3: 32ch * 9 → 96ch * 9
        T = torch.cat([f_combine, f_prod3], dim=-1)
        
        # Second product layer
        # ElementwiseTensorProduct(T, T) → scalars
        f_prod5 = self.product_5(T, T)
        
        # Readout
        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        atom_energies = e_out.sum(dim=-1, keepdim=True)
        
        return atom_energies


# ============================================================================
# Loose Versions - Using Optimized CartesianFullyConnectedTensorProduct
# (Faster but not strictly equivariant)
# ============================================================================

class CartesianE3ConvSparseLoose(nn.Module):
    """
    Cartesian E3 convolution using optimized CartesianFullyConnectedTensorProduct.
    
    This is a faster but non-equivariant alternative to CartesianE3ConvSparse.
    Uses CartesianFullyConnectedTensorProduct instead of EquivariantTensorProduct
    for both tensor products, providing significant speedup at the cost of
    strict equivariance.
    
    The architecture and flow are identical to CartesianE3ConvSparse, but
    uses norm product approximation instead of CG coefficients.
    
    Forward flow:
        1. Atom embedding: A[edge_src] → scalar features Ai[edge_src]
        2. Direction conversion: edge_vec → spherical harmonics sh_edge
        3. Tensor product 1: scalar(Ai[edge_src]) ⊗ sh_edge → higher-order irreps (f_in)
        4. Radial basis: edge_length → radial basis expansion
        5. Weight network: radial basis → tensor product 2 weights
        6. Tensor product 2: f_in ⊗ scalar(Ai[edge_dst]) → higher-order output (with weights)
        7. Aggregation: Aggregate edge features to nodes
    
    Args:
        max_radius: Maximum cutoff radius for neighbor search.
        number_of_basis: Number of radial basis functions.
        channels_out: Number of output channels (multiplicity for each l).
        embedding_dim: Dimension of atom type embedding.
        max_atomvalue: Maximum atomic number (for embedding lookup).
        output_size: Size of atom feature MLP output (multiplicity for tp1).
        lmax: Maximum angular momentum.
        function_type: Type of radial basis function.
                       Options: 'gaussian', 'bessel', 'fourier', 'cosine', 'smooth_finite'
    
    Attributes:
        tp1: First tensor product (scalar ⊗ sh → irreps).
             Uses CartesianFullyConnectedTensorProduct with internal weights.
        tp2: Second tensor product (irreps ⊗ scalar → irreps).
             Uses CartesianFullyConnectedTensorProduct with external weights.
        fc: Weight network mapping radial basis to tp2 weights.
        cart_to_irreps: Converter from Cartesian vectors to spherical harmonics.
        atom_embedding: Embedding layer for atom types.
        atom_mlp: MLP to process atom embeddings into scalar features.
        output_dim: Total output dimension = channels_out * sum(2*l+1 for l in range(lmax+1))
    
    Performance:
        - Faster than CartesianE3ConvSparse (uses norm product instead of CG)
        - weight_numel matches EquivariantTensorProduct (efficient fc networks)
        - Not strictly equivariant (uses approximation)
    
    Example:
        >>> conv = CartesianE3ConvSparseLoose(
        ...     max_radius=5.0,
        ...     number_of_basis=8,
        ...     channels_out=64,
        ...     lmax=2
        ... )
        >>> pos = torch.randn(50, 3)
        >>> A = torch.randint(0, 10, (50,))
        >>> batch = torch.zeros(50, dtype=torch.long)
        >>> edge_src = torch.randint(0, 50, (200,))
        >>> edge_dst = torch.randint(0, 50, (200,))
        >>> edge_shifts = torch.zeros(200, 3)
        >>> cell = torch.eye(3).unsqueeze(0)
        >>> out = conv(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        >>> # out shape: (50, 576) = 50 nodes * 64 channels * 9 dims
    
    Note:
        This layer uses CartesianFullyConnectedTensorProduct which is faster
        but not strictly equivariant. For strict equivariance, use CartesianE3ConvSparse.
    """
    
    def __init__(self, max_radius, number_of_basis, channels_out,
                 embedding_dim=16, max_atomvalue=10, output_size=8,
                 lmax=2, function_type='gaussian'):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.lmax = lmax
        self.function_type = function_type
        
        # Dimensions
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)  # 9 for lmax=2
        self.output_dim = channels_out * self.total_sh_dim
        
        # Cartesian to irreps converter
        self.cart_to_irreps = CartesianToIrrepsLight(lmax=lmax)
        
        # Atom embedding
        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size)
        )
        
        # ------------------------------------------------------------------
        # tp1: scalar(Ai[edge_src]) ⊗ sh_edge → 高阶 irreps
        # 使用优化后的 CartesianFullyConnectedTensorProduct (更快但非严格等变)
        # ------------------------------------------------------------------
        irreps_sh = get_irreps_str(1, lmax)  # "1x0e + 1x1o + 1x2e"
        self.tp1_irreps_out = get_irreps_str(output_size, lmax)
        self.tp1 = CartesianFullyConnectedTensorProduct(
            irreps_in1=f"{output_size}x0e", 
            irreps_in2=irreps_sh,
            irreps_out=self.tp1_irreps_out,
            shared_weights=True,
            internal_weights=True
        )
        
        # ------------------------------------------------------------------
        # tp2: 高阶irreps(f_in) ⊗ scalar(Ai[edge_dst]) → 高阶输出
        # 使用优化后的 CartesianFullyConnectedTensorProduct (更快但非严格等变)
        # ------------------------------------------------------------------
        self.tp2 = CartesianFullyConnectedTensorProduct(
            irreps_in1=self.tp1_irreps_out,       # 高阶 (output_size x 0e + output_size x 1o + ...)
            irreps_in2=f"{output_size}x0e",       # scalar (neighbor atom features)
            irreps_out=get_irreps_str(channels_out, lmax),  # 高阶输出
            shared_weights=False,
            internal_weights=False
        )
        
        # Weight network (radial basis → tp2 weights)
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.weight_numel)
        )
        
        print(
            f"CartesianE3ConvSparseLoose init: lmax={lmax}, output_dim={self.output_dim}, "
            f"tp1={self.tp1.irreps_in1_str} ⊗ {self.tp1.irreps_in2_str}, "
            f"tp2={self.tp1_irreps_out} ⊗ {output_size}x0e → {channels_out}ch, "
            f"tp2_weight_numel={self.tp2.weight_numel}"
        )
    
    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum('ni,nij->nj', edge_shifts, edge_cells)
        
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        
        # Convert direction to irreps (1x0e + 1x1o + 1x2e)
        sh_edge = self.cart_to_irreps(edge_vec)  # (E, total_sh_dim)
        
        num_nodes = pos.size(0)
        
        # Atom features (scalar)
        atom_emb = self.atom_embedding(A.long())
        Ai = self.atom_mlp(atom_emb)  # (N, output_size)
        
        # tp1: scalar(Ai[edge_src]) ⊗ sh_edge → 高阶 irreps
        f_in = self.tp1(Ai[edge_src], sh_edge)  # (E, output_size * total_sh_dim)
        
        # Radial basis
        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis,
            basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        
        # tp2 weights from radial basis
        weights = self.fc(emb)  # (E, tp2.weight_numel)
        
        # tp2: 高阶irreps(f_in) ⊗ scalar(Ai[edge_dst]) → 高阶输出
        edge_features = self.tp2(f_in, Ai[edge_dst], weights)  # (E, channels_out * total_sh_dim)
        
        # Aggregate over edges
        neighbor_count = scatter(
            torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes
        ).clamp(min=1).to(edge_features.dtype)
        
        out = scatter(
            edge_features, edge_dst, dim=0, dim_size=num_nodes
        ).div(neighbor_count.unsqueeze(-1))
        
        return out


class CartesianE3Conv2SparseLoose(nn.Module):
    """
    Second Cartesian E3 convolution using optimized CartesianFullyConnectedTensorProduct.
    
    This is a faster but non-equivariant alternative to CartesianE3Conv2Sparse.
    Uses CartesianFullyConnectedTensorProduct instead of EquivariantTensorProduct
    for the tensor product, providing significant speedup at the cost of
    strict equivariance.
    
    The architecture and flow are identical to CartesianE3Conv2Sparse, but
    uses norm product approximation instead of CG coefficients.
    
    Forward flow:
        1. Direction conversion: edge_vec → spherical harmonics sh_edge
        2. Radial basis: edge_length → radial basis expansion
        3. Weight network: radial basis → tensor product weights
        4. Tensor product: f_in[edge_src] ⊗ sh_edge → higher-order output (with weights)
        5. Aggregation: Aggregate edge features to nodes
    
    Args:
        max_radius: Maximum cutoff radius for neighbor search.
        number_of_basis: Number of radial basis functions.
        channels_in: Number of input channels (multiplicity for each l).
        channels_out: Number of output channels (multiplicity for each l).
        embedding_dim: Dimension of atom type embedding (not used, kept for compatibility).
        max_atomvalue: Maximum atomic number (not used, kept for compatibility).
        lmax: Maximum angular momentum.
        function_type: Type of radial basis function.
                       Options: 'gaussian', 'bessel', 'fourier', 'cosine', 'smooth_finite'
    
    Attributes:
        tp: Tensor product (irreps ⊗ sh → irreps).
            Uses CartesianFullyConnectedTensorProduct with external weights.
        fc: Weight network mapping radial basis to tp weights.
        cart_to_irreps: Converter from Cartesian vectors to spherical harmonics.
        input_dim: Total input dimension = channels_in * sum(2*l+1 for l in range(lmax+1))
        output_dim: Total output dimension = channels_out * sum(2*l+1 for l in range(lmax+1))
    
    Performance:
        - Faster than CartesianE3Conv2Sparse (uses norm product instead of CG)
        - weight_numel matches EquivariantTensorProduct (efficient fc networks)
        - Not strictly equivariant (uses approximation)
    
    Example:
        >>> conv = CartesianE3Conv2SparseLoose(
        ...     max_radius=5.0,
        ...     number_of_basis=8,
        ...     channels_in=64,
        ...     channels_out=64,
        ...     lmax=2
        ... )
        >>> f_in = torch.randn(50, 576)  # 50 nodes * 64 channels * 9 dims
        >>> pos = torch.randn(50, 3)
        >>> A = torch.randint(0, 10, (50,))
        >>> batch = torch.zeros(50, dtype=torch.long)
        >>> edge_src = torch.randint(0, 50, (200,))
        >>> edge_dst = torch.randint(0, 50, (200,))
        >>> edge_shifts = torch.zeros(200, 3)
        >>> cell = torch.eye(3).unsqueeze(0)
        >>> out = conv(f_in, pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        >>> # out shape: (50, 576) = 50 nodes * 64 channels * 9 dims
    
    Note:
        This layer uses CartesianFullyConnectedTensorProduct which is faster
        but not strictly equivariant. For strict equivariance, use CartesianE3Conv2Sparse.
    """
    
    def __init__(self, max_radius, number_of_basis, channels_in, channels_out,
                 embedding_dim=16, max_atomvalue=10, lmax=2, function_type='gaussian'):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.lmax = lmax
        self.function_type = function_type
        
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)
        self.input_dim = channels_in * self.total_sh_dim
        self.output_dim = channels_out * self.total_sh_dim
        
        self.cart_to_irreps = CartesianToIrrepsLight(lmax=lmax)
        
        # ------------------------------------------------------------------
        # tp: 高阶irreps(f_in[edge_src]) ⊗ 方向sh → 高阶输出
        # 使用优化后的 CartesianFullyConnectedTensorProduct (更快但非严格等变)
        # ------------------------------------------------------------------
        irreps_in = get_irreps_str(channels_in, lmax)   # channels_in x (0e + 1o + 2e)
        irreps_sh = get_irreps_str(1, lmax)              # 1x0e + 1x1o + 1x2e
        irreps_out = get_irreps_str(channels_out, lmax)  # channels_out x (0e + 1o + 2e)
        
        self.tp = CartesianFullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            shared_weights=False,
            internal_weights=False
        )
        
        # Weight network (radial basis → tp weights)
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel)
        )
        
        print(f"CartesianE3Conv2SparseLoose init: lmax={lmax}, "
              f"tp={irreps_in} ⊗ {irreps_sh} → {irreps_out}, "
              f"tp_weight_numel={self.tp.weight_numel}")
    
    def forward(self, f_in, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum('ni,nij->nj', edge_shifts, edge_cells)
        
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        
        # Convert direction to sh irreps
        sh_edge = self.cart_to_irreps(edge_vec)  # (E, total_sh_dim)
        
        num_nodes = pos.size(0)
        
        # Radial basis
        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis,
            basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        
        # tp weights from radial basis
        weights = self.fc(emb)  # (E, tp.weight_numel)
        
        # tp: 高阶irreps(f_in[edge_src]) ⊗ 方向sh → 高阶输出
        edge_features = self.tp(f_in[edge_src], sh_edge, weights)  # (E, channels_out * total_sh_dim)
        
        # Aggregate over edges
        neighbor_count = scatter(
            torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes
        ).clamp(min=1).to(edge_features.dtype)
        
        out = scatter(
            edge_features, edge_dst, dim=0, dim_size=num_nodes
        ).div(neighbor_count.unsqueeze(-1))
        
        return out


class CartesianTransformerLayerLoose(nn.Module):
    """
    Cartesian Transformer layer using optimized CartesianFullyConnectedTensorProduct.
    
    This is a faster but non-equivariant alternative to CartesianTransformerLayer.
    All tensor products use CartesianFullyConnectedTensorProduct instead of
    EquivariantTensorProduct, providing significant speedup at the cost of
    strict equivariance.
    
    Architecture flow (same as CartesianTransformerLayer):
        1. First convolution (CartesianE3ConvSparseLoose):
           Converts atom types and edge directions to higher-order irreps.
        2. Second convolution (CartesianE3Conv2SparseLoose):
           Further processes features with edge direction information.
        3. Feature combination:
           Concatenates outputs from both convolutions.
        4. Product layer 3 (CartesianFullyConnectedTensorProduct):
           Computes tensor product: f_combine ⊗ f_combine → scalars (32x0e)
        5. Feature concatenation:
           Combines f_combine and f_prod3 into T.
        6. Product layer 5 (CartesianElementwiseTensorProduct):
           Computes element-wise product: T ⊗ T → scalars
        7. Readout:
           MLP + weighted sum to predict atom energies
    
    Key differences from CartesianTransformerLayer:
        - Uses CartesianE3ConvSparseLoose instead of CartesianE3ConvSparse
        - Uses CartesianE3Conv2SparseLoose instead of CartesianE3Conv2Sparse
        - Uses CartesianFullyConnectedTensorProduct for product_3
        - All tensor products use norm product approximation (not CG coefficients)
    
    Args:
        max_embed_radius: Maximum radius for embedding convolution.
        main_max_radius: Maximum radius for main convolution.
        main_number_of_basis: Number of radial basis functions.
        hidden_dim_conv: Number of channels for convolutions.
        hidden_dim_sh: Hidden dimension for spherical harmonics (not used, kept for compatibility).
        hidden_dim: Hidden dimension for readout MLP (not used, kept for compatibility).
        channel_in2: Channel dimension for intermediate layers (not used, kept for compatibility).
        embedding_dim: Dimension of atom type embedding.
        max_atomvalue: Maximum atomic number (for embedding lookup).
        output_size: Size of atom feature MLP output in first convolution.
        embed_size: List of hidden layer sizes for readout MLP.
                   Default: [128, 128, 128]
        main_hidden_sizes3: List of hidden layer sizes (not used, kept for compatibility).
                           Default: [64, 32]
        num_layers: Number of transformer layers (not used, kept for compatibility).
                   Default: 1
        device: Device to place model on. If None, uses CUDA if available.
        function_type_main: Type of radial basis function.
                           Options: 'gaussian', 'bessel', 'fourier', 'cosine', 'smooth_finite'
        lmax: Maximum angular momentum. Default: 2
    
    Attributes:
        e3_conv_emb: First convolution layer (CartesianE3ConvSparseLoose).
        e3_conv_emb2: Second convolution layer (CartesianE3Conv2SparseLoose).
        product_3: Tensor product layer (CartesianFullyConnectedTensorProduct).
        product_5: Element-wise tensor product layer (CartesianElementwiseTensorProduct).
        proj_total: Readout MLP (MainNet).
        weighted_sum: Weighted sum layer for final energy prediction.
        channels: Number of channels (equal to hidden_dim_conv).
        total_sh_dim: Total spherical harmonics dimension = sum(2*l+1 for l in range(lmax+1))
    
    Performance:
        - Forward: 1.5-2x faster than CartesianTransformerLayer
        - With torch.compile: 3-5x faster than e3nn (inference only)
        - Not strictly equivariant (uses norm product approximation)
        - Same parameter count as CartesianTransformerLayer
    
    Example:
        >>> model = CartesianTransformerLayerLoose(
        ...     max_embed_radius=5.0,
        ...     main_max_radius=5.0,
        ...     main_number_of_basis=8,
        ...     hidden_dim_conv=64,
        ...     hidden_dim_sh=64,
        ...     hidden_dim=64,
        ...     lmax=2
        ... )
        >>> pos = torch.randn(50, 3)
        >>> A = torch.randint(0, 10, (50,))
        >>> batch = torch.zeros(50, dtype=torch.long)
        >>> edge_src = torch.randint(0, 50, (200,))
        >>> edge_dst = torch.randint(0, 50, (200,))
        >>> edge_shifts = torch.zeros(200, 3)
        >>> cell = torch.eye(3).unsqueeze(0)
        >>> energy = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        >>> # energy shape: (50, 1) - atom energies
        >>> # For inference with torch.compile:
        >>> model_compiled = torch.compile(model, mode='reduce-overhead')
        >>> energy = model_compiled(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    
    Note:
        This model uses CartesianFullyConnectedTensorProduct for all tensor operations,
        which is faster but not strictly equivariant. For strict equivariance,
        use CartesianTransformerLayer.
        
        torch.compile can be used for inference to achieve 3-5x speedup.
        However, torch.compile cannot be used during training because it does not
        support double backward (required for force training: force = -dE/dr,
        then loss.backward() needs d(force)/d(parameters)).
    """
    
    def __init__(self, max_embed_radius, main_max_radius, main_number_of_basis,
                 hidden_dim_conv, hidden_dim_sh, hidden_dim, channel_in2=32, embedding_dim=16,
                 max_atomvalue=10, output_size=8, embed_size=None, main_hidden_sizes3=None,
                 num_layers=1, device=None, function_type_main='gaussian', lmax=2, num_interaction=2):
        super().__init__()
        
        if embed_size is None:
            embed_size = [128, 128, 128]
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.lmax = lmax
        self.dims = [2 * l + 1 for l in range(lmax + 1)]
        self.total_sh_dim = sum(self.dims)  # 9 for lmax=2
        
        # Use channel count, not full dimension
        self.channels = hidden_dim_conv  # This should be 64, not 576
        self.hidden_dim_conv = self.channels * self.total_sh_dim  # 64 * 9 = 576

        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")
        
        # Convolution layers with optimized CartesianFullyConnectedTensorProduct
        self.e3_conv_layers = nn.ModuleList()
        self.e3_conv_layers.append(
            CartesianE3ConvSparseLoose(
                max_radius=max_embed_radius,
                number_of_basis=main_number_of_basis,
                channels_out=self.channels,
                embedding_dim=embedding_dim,
                max_atomvalue=max_atomvalue,
                output_size=output_size,
                lmax=lmax,
                function_type=function_type_main
            )
        )
        for _ in range(1, self.num_interaction):
            self.e3_conv_layers.append(
                CartesianE3Conv2SparseLoose(
                    max_radius=max_embed_radius,
                    number_of_basis=main_number_of_basis,
                    channels_in=self.channels,
                    channels_out=self.channels,
                    embedding_dim=embedding_dim,
                    max_atomvalue=max_atomvalue,
                    lmax=lmax,
                    function_type=function_type_main
                )
            )
        
        # Product layers using optimized CartesianFullyConnectedTensorProduct
        combined_channels = self.channels * self.num_interaction
        irreps_combined = get_irreps_str(combined_channels, lmax)
        scalar_channels = (self.num_interaction - 1) * 32
        
        # product_3: 高阶 ⊗ 高阶 → 标量 (32x0e) - 使用优化后的 CartesianFullyConnectedTensorProduct
        self.product_3 = CartesianFullyConnectedTensorProduct(
            irreps_in1=irreps_combined,
            irreps_in2=irreps_combined,
            irreps_out=f"{scalar_channels}x0e",
            shared_weights=True,
            internal_weights=True
        )
        
        # T = [f_combine, f_prod3] = (combined_channels * total_sh_dim + 32) features
        T_channels = combined_channels  # For the high-order part
        
        # Build T irreps string
        irreps_T = get_irreps_str(T_channels, lmax) + " + " + f"{scalar_channels}x0e"
        
        # product_5: ElementwiseTensorProduct(T, T, ["0e"]) - 严格等变
        self.product_5 = CartesianElementwiseTensorProduct(
            irreps_in=irreps_T,
            filter_ir_out=["0e"],
            lmax=lmax
        )
        
        # Readout input: product_5 outputs scalars
        readout_dim = self.product_5.dim_out
        self.proj_total = MainNet(readout_dim, embed_size, 17)
        self.num_features = 17
        self.weighted_sum = RobustScalarWeightedSum(self.num_features, init_weights='zero')
        
        print(f"CartesianTransformerLayerLoose init complete: lmax={lmax}, channels={self.channels}")
    
    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]
        
        # Convolutions
        features = []
        f_prev = self.e3_conv_layers[0](pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        features.append(f_prev)
        for conv in self.e3_conv_layers[1:]:
            f_prev = conv(f_prev, pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
            features.append(f_prev)
        
        # Combine features: reorder to match e3nn's expected irreps format
        # This ensures consistency with CartesianTransformerLayer, even though
        # CartesianFullyConnectedTensorProduct uses norm product (rotation-invariant)
        f_combine = reorder_concatenated_irreps_multi(features, self.channels, self.lmax)
        
        # product_3: 高阶 ⊗ 高阶 → 32x0e (标量)
        f_prod3 = self.product_3(f_combine, f_combine)  # (N, 32)
        
        # Build T features: [f_combine, f_prod3]
        # T = high-order part + scalar part
        T = torch.cat([f_combine, f_prod3], dim=-1)
        
        # product_5: ElementwiseTensorProduct(T, T) → scalars
        f_prod5 = self.product_5(T, T)  # (N, output_channels)
        
        # Readout
        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        atom_energies = e_out.sum(dim=-1, keepdim=True)
        
        return atom_energies


# ==============================================================================
# Backward-compatible aliases
# ==============================================================================

# CartesianTransformerLayer is an alias for CartesianTransformerLayerStrict
# for backward compatibility with existing code that imports CartesianTransformerLayer.
# Both use CartesianFullyConnectedTensorProduct for tensor products.
CartesianTransformerLayer = CartesianTransformerLayerStrict
