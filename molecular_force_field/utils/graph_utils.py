"""Graph utility functions for molecular modeling."""

import torch
from torch import Tensor
try:
    from torch_cluster import radius_graph, radius
except Exception:
    # Fallback if torch_cluster is not available
    radius_graph = None
    radius = None
from molecular_force_field.utils.scatter import scatter


def optimized_sorted_radius_graph(
    pos: Tensor,
    radius: float,
    max_num_neighbors: int,
    loop: bool = False
) -> Tensor:
    """
    Optimized sorted radius graph with neighbor limit.
    
    Args:
        pos: Node positions tensor
        radius: Cutoff radius
        max_num_neighbors: Maximum number of neighbors per node
        loop: Whether to include self-loops
        
    Returns:
        Edge index tensor with sorted neighbors
    """
    # Step 1: Generate candidate edges using radius_graph (avoid O(N²))
    edge_index = radius_graph(pos, r=radius, loop=loop)
    
    # Step 2: Calculate distance for each edge
    edge_src, edge_dst = edge_index
    edge_vec = pos[edge_dst] - pos[edge_src]
    distances = edge_vec.norm(dim=1)
    
    # Step 3: Sort and truncate by target node
    # Generate unique target node indices and grouping pointers
    sorted_dst, sorted_indices = torch.sort(edge_dst)
    unique_dst, counts = torch.unique_consecutive(sorted_dst, return_counts=True)
    ptr = torch.cat([torch.zeros(1, device=pos.device, dtype=torch.long), counts.cumsum(0)])
    
    keep_mask = torch.zeros_like(edge_dst, dtype=torch.bool)
    for i in range(len(unique_dst)):
        dst = unique_dst[i]
        start = ptr[i]
        end = ptr[i + 1]
        # Sort edges for current target node by distance
        local_indices = sorted_indices[start:end]
        local_distances = distances[local_indices]
        # Get top max neighbors by distance
        _, sorted_local_indices = torch.sort(local_distances)
        selected = sorted_local_indices[:max_num_neighbors]
        keep_mask[local_indices[selected]] = True
    
    return edge_index[:, keep_mask]


def S_map(rij, r_cs: float, r_c: float, device=None, eps=1e-10):
    """
    Smooth cutoff function for atomic interactions.
    
    Args:
        rij: Atomic distance tensor (N_edges,)
        r_cs: Inner cutoff radius (<r_c)
        r_c: Outer cutoff radius
        device: Device to place tensors on
        eps: Numerical stability coefficient to prevent division by zero
        
    Returns:
        Weight tensor with smooth cutoff
    """
    if device is None:
        device = rij.device
    
    # Ensure rc > rcs + eps to avoid numerical issues
    r_cs_tensor = torch.tensor(r_cs, device=device, dtype=torch.float64)
    r_c_tensor = torch.tensor(r_c, device=device, dtype=torch.float64)
    delta_r = r_c_tensor - r_cs_tensor + eps  # Denominator as safety margin
    
    # Calculate u value, denominator may contain very small values, need stable handling
    u = (rij - r_cs_tensor) / (delta_r)
    
    # Ternary condition masks
    cond_1 = rij < r_cs_tensor                     # Condition 1: Inner hard cutoff region
    cond_2 = (rij >= r_cs_tensor) & (rij < r_c_tensor)  # Smooth transition region
    cond_3 = rij >= r_c_tensor                        # Outer complete cutoff region
    
    # Weight calculation for each region
    w = torch.zeros_like(rij)
    # Condition 1 handling: 1 / (|r_ij| + eps)
    w = torch.where(cond_1, 1.0 / (rij), w)
    # Condition 2 handling: Polynomial interpolation part
    cubic_coeff = u ** 3 * (-6 * u**2 + 15 * u - 10) + 1
    safe_denominator = rij + eps  # Prevent division by zero
    w = torch.where(cond_2, cubic_coeff / safe_denominator, w)
    # Condition 3 already zero in w initialization, no additional handling needed
    # Final result is independent of distance direction, ensures non-negativity
    return w.abs()


def get_edge_pairs(edge_src, edge_dst, num_nodes):
    """
    Get edge pairs for three-body interactions.
    
    Args:
        edge_src: Source node indices
        edge_dst: Destination node indices
        num_nodes: Total number of nodes
        
    Returns:
        Tuple of (pair_i, pair_j, centers) edge indices
    """
    device = edge_src.device
    E = edge_src.numel()
    if E == 0:
        return (torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long))

    # Step 1: Force sort (ensure local index calculation is correct)
    sort_idx = torch.argsort(edge_dst)
    edge_src = edge_src[sort_idx]
    edge_dst = edge_dst[sort_idx]

    # Step 2: Calculate local indices (Local ID)
    # Calculate degree for each atom
    deg = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes, reduce='sum')
    max_deg = int(deg.max().item())
    
    if max_deg < 2:
        return (torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long))

    # Calculate sequence number of each edge within its atom (0, 1, 2...deg-1)
    edge_offsets = torch.zeros(num_nodes, device=device, dtype=torch.long)
    edge_offsets[1:] = deg.cumsum(0)[:-1]
    local_idx = torch.arange(E, device=device) - edge_offsets[edge_dst]

    # Step 3: Build "node-neighbor" matrix
    # Matrix size: [total atoms, max neighbors]
    # Matrix stores: index of "edge" in current (sorted) edge_src at this position
    neighbor_matrix = torch.full((num_nodes, max_deg), -1, device=device, dtype=torch.long)
    
    # Core: Fill edge indices into matrix
    neighbor_matrix[edge_dst, local_idx] = torch.arange(E, device=device)

    # Step 4: Pairwise combination (Vectorized Combination)
    pair_i, pair_j, centers = [], [], []
    
    for i in range(max_deg):
        for j in range(i + 1, max_deg):
            col_i = neighbor_matrix[:, i]
            col_j = neighbor_matrix[:, j]
            
            # Only when node has both i-th and j-th neighbors, mask is True
            mask = (col_i != -1) & (col_j != -1)
            
            if mask.any():
                pair_i.append(col_i[mask])
                pair_j.append(col_j[mask])
                # Record center atom IDs for these pairs
                centers.append(torch.where(mask)[0])

    if not pair_i:
        return (torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long))

    # Return results
    return torch.cat(pair_i), torch.cat(pair_j), torch.cat(centers)


def _normalize_pbc_flags(pbc, *, device: torch.device) -> torch.Tensor:
    if pbc is None:
        return torch.ones(3, device=device, dtype=torch.bool)
    if isinstance(pbc, torch.Tensor):
        pbc_tensor = pbc.to(device=device, dtype=torch.bool).view(-1)
    else:
        pbc_tensor = torch.as_tensor(pbc, device=device, dtype=torch.bool).view(-1)
    if pbc_tensor.numel() == 1:
        pbc_tensor = pbc_tensor.expand(3)
    if pbc_tensor.numel() != 3:
        raise ValueError(f"pbc must have length 3, got shape {tuple(pbc_tensor.shape)}")
    return pbc_tensor


def radius_graph_pbc_gpu(pos, r, cell, max_num_neighbors=100, pbc=None):
    """
    Calculate PBC-aware neighbor list on GPU, replacing ASE neighbor_list.
    
    Args:
        pos: Atomic positions tensor
        r: Cutoff radius
        cell: Unit cell tensor [1, 3, 3] or [3, 3]
        max_num_neighbors: Maximum number of neighbors
        pbc: Periodicity flags per axis. ``None`` keeps legacy 3D periodic behavior.
        
    Returns:
        Tuple of (edge_src, edge_dst, edge_shifts)
    """
    if radius is None:
        raise ImportError("torch_cluster is required for radius_graph_pbc_gpu. Install it with: pip install torch-cluster")
    
    device = pos.device
    
    # 1. Generate mirror offsets only along periodic axes.
    pbc_flags = _normalize_pbc_flags(pbc, device=device)
    axis_ranges = []
    for axis in range(3):
        if bool(pbc_flags[axis].item()):
            axis_ranges.append(torch.tensor([-1, 0, 1], device=device, dtype=torch.long))
        else:
            axis_ranges.append(torch.tensor([0], device=device, dtype=torch.long))
    offsets_idx = torch.cartesian_prod(*axis_ranges)
    
    all_src, all_dst, all_shifts = [], [], []

    # Normalize cell shape: allow [1,3,3] or [3,3]
    cell_mat = cell.squeeze(0) if cell.dim() == 3 else cell
    # Convention used across this repo (and ASE): cell is a 3x3 matrix where each ROW
    # is a lattice vector in Cartesian coordinates: [a; b; c].
    # A PBC integer shift s = (sx, sy, sz) corresponds to the translation:
    #   shift_vec = sx * a + sy * b + sz * c  ==  s @ cell_mat

    # 2. Loop through mirrors (very fast on GPU)
    for s_idx in offsets_idx:
        # Calculate physical displacement: shift = s_idx @ cell
        # s_idx: [3] -> shift: [3]
        # IMPORTANT: for triclinic cells this must be s @ cell (row-vector @ matrix),
        # matching the model's usage: einsum('ni,nij->nj', edge_shifts, edge_cells).
        shift = torch.einsum('i,ij->j', s_idx.to(pos.dtype), cell_mat)
        pos_shifted = pos + shift
        
        # GPU radius search
        edge_index = radius(pos_shifted, pos, r, max_num_neighbors=max_num_neighbors)
        src, dst = edge_index[0], edge_index[1]
        
        # Exclude self-loops (0 offset and src==dst)
        if s_idx.abs().sum() == 0:
            mask = src != dst
            src, dst = src[mask], dst[mask]
        
        if src.numel() > 0:
            all_src.append(src)
            all_dst.append(dst)
            all_shifts.append(s_idx.expand(len(src), -1))

    if not all_src:
        return (torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, 3, device=device, dtype=torch.float64))

    return torch.cat(all_src), torch.cat(all_dst), torch.cat(all_shifts).to(torch.float64)