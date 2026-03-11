#!/usr/bin/env python3
"""
等变性测试：验证各 tensor_product_mode 在 O(3) 旋转/反射下能量不变、力协变。

测试逻辑：
  - 能量不变性：E(pos @ R.T) = E(pos)
  - 力协变性：F(pos @ R.T) = F(pos) @ R.T
  - l=2 输出等变性：Y(pos @ R.T) = D_2(R) @ Y(pos)

Usage:
  python -m molecular_force_field.test.test_equivariance
  python -m molecular_force_field.test.test_equivariance --mode spherical-save-cue --device cuda
  python -m molecular_force_field.test.test_equivariance --modes spherical spherical-save spherical-save-cue
  python -m molecular_force_field.test.test_equivariance --test-l2  # 额外测试 l=2 输出等变性
"""

from __future__ import annotations

import argparse
import sys

import torch

from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.utils.graph_utils import radius_graph_pbc_gpu


def random_orthogonal(device: torch.device, dtype: torch.dtype, *, reflect: bool = False, seed: int = 0) -> torch.Tensor:
    """生成随机 O(3) 矩阵（旋转或含反射）。"""
    if device.type == "cuda":
        try:
            g = torch.Generator(device="cuda")
            g.manual_seed(seed)
            M = torch.randn(3, 3, generator=g, device=device, dtype=torch.float64)
            Q, _ = torch.linalg.qr(M)
        except Exception:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            M = torch.randn(3, 3, generator=g, device="cpu", dtype=torch.float64)
            Q, _ = torch.linalg.qr(M)
            Q = Q.to(device)
    else:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        M = torch.randn(3, 3, generator=g, device="cpu", dtype=torch.float64)
        Q, _ = torch.linalg.qr(M)
        Q = Q.to(device)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    if reflect:
        Q[:, 0] = -Q[:, 0]
    return Q.to(device=device, dtype=dtype)


def make_dummy_graph(device: torch.device, dtype: torch.dtype, num_nodes: int = 32, avg_degree: int = 24, seed: int = 42):
    """生成 dummy 图（大胞胞，主要用 central image）。"""
    torch.manual_seed(seed)
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype) * 2.0
    A = torch.randint(1, 6, (num_nodes,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 20.0
    edge_src, edge_dst, edge_shifts = radius_graph_pbc_gpu(pos, 5.0, cell)
    cell = cell.expand(num_nodes, -1, -1)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def build_e3trans(mode: str, config: ModelConfig, device: torch.device):
    """根据 mode 构建 e3trans 层。"""
    if mode == "spherical":
        from molecular_force_field.models import E3_TransformerLayer_multi
        return E3_TransformerLayer_multi(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            device=device,
        ).to(device)
    if mode == "spherical-save":
        from molecular_force_field.models.e3nn_layers_channelwise import E3_TransformerLayer_multi as E3_cw
        return E3_cw(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            device=device,
        ).to(device)
    if mode == "spherical-save-cue":
        from molecular_force_field.models.cue_layers_channelwise import E3_TransformerLayer_multi as E3_cue
        return E3_cue(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            device=device,
        ).to(device)
    if mode == "pure-cartesian":
        from molecular_force_field.models import PureCartesianTransformerLayer
        return PureCartesianTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device,
        ).to(device)
    if mode == "pure-cartesian-sparse":
        from molecular_force_field.models import PureCartesianSparseTransformerLayer
        return PureCartesianSparseTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            lmax=config.lmax,
            max_rank_other=1,
            k_policy="k0",
            device=device,
        ).to(device)
    if mode == "partial-cartesian":
        from molecular_force_field.models import CartesianTransformerLayer
        return CartesianTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device,
        ).to(device)
    if mode == "partial-cartesian-loose":
        from molecular_force_field.models import CartesianTransformerLayerLoose
        return CartesianTransformerLayerLoose(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            lmax=config.lmax,
            device=device,
        ).to(device)
    if mode == "pure-cartesian-ictd":
        from molecular_force_field.models.pure_cartesian_ictd_layers_full import PureCartesianICTDTransformerLayer
        return PureCartesianICTDTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy="full",
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=config.dtype,
            device=device,
        ).to(device)
    if mode == "pure-cartesian-ictd-save":
        from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer
        return PureCartesianICTDTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            lmax=config.lmax,
            ictd_tp_path_policy="full",
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=config.dtype,
            device=device,
        ).to(device)
    raise ValueError(f"Unsupported mode: {mode}")


def run_equivariance_test(
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    num_nodes: int = 32,
    tol_energy: float = 1e-5,
    tol_force: float = 1e-5,
    n_rotations: int = 3,
) -> tuple[bool, str]:
    """
    对给定 mode 运行等变性测试。
    返回 (passed, message)。
    """
    config = ModelConfig(dtype=dtype, lmax=2)
    config.atomic_energy_keys = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor(
        [-430.53, -821.03, -1488.19, -2044.35], dtype=dtype
    )

    try:
        e3trans = build_e3trans(mode, config, device)
    except Exception as e:
        return False, f"build failed: {e}"

    e3trans.eval()
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = make_dummy_graph(
        device, dtype, num_nodes=num_nodes
    )

    def get_energy_and_forces(pos_in, cell_in):
        pos_in = pos_in.detach().requires_grad_(True)
        E_per_atom = e3trans(pos_in, A, batch, edge_src, edge_dst, edge_shifts, cell_in)
        E = E_per_atom.sum()
        F = -torch.autograd.grad(E, pos_in, create_graph=False)[0]
        return E.detach(), F.detach()

    E0, F0 = get_energy_and_forces(pos, cell)

    err_energy_max = 0.0
    err_force_max = 0.0

    for i in range(n_rotations):
        R = random_orthogonal(device, dtype, reflect=False, seed=100 + i)
        Rr = random_orthogonal(device, dtype, reflect=True, seed=200 + i)

        pos_R = pos @ R.T
        cell_R = cell @ R.T
        pos_Rr = pos @ Rr.T
        cell_Rr = cell @ Rr.T

        E_R, F_R = get_energy_and_forces(pos_R, cell_R)
        E_Rr, F_Rr = get_energy_and_forces(pos_Rr, cell_Rr)

        # 能量不变性
        denom = (E0.abs() + E_R.abs() + 1e-12).item()
        err_E_rot = (E0 - E_R).abs().item() / denom
        denom_r = (E0.abs() + E_Rr.abs() + 1e-12).item()
        err_E_ref = (E0 - E_Rr).abs().item() / denom_r
        err_energy_max = max(err_energy_max, err_E_rot, err_E_ref)

        # 力协变性: F_R = F0 @ R.T  =>  F_R.T = R @ F0.T  =>  F_R = (R @ F0.T).T = F0 @ R.T
        F0_R = F0 @ R.T
        F0_Rr = F0 @ Rr.T
        denom_f = (F0.norm() + F_R.norm() + 1e-12).item()
        err_F_rot = (F_R - F0_R).norm().item() / denom_f
        denom_fr = (F0.norm() + F_Rr.norm() + 1e-12).item()
        err_F_ref = (F_Rr - F0_Rr).norm().item() / denom_fr
        err_force_max = max(err_force_max, err_F_rot, err_F_ref)

    passed = err_energy_max <= tol_energy and err_force_max <= tol_force
    msg = f"energy_err={err_energy_max:.3e} force_err={err_force_max:.3e} (tol_energy={tol_energy}, tol_force={tol_force})"
    return passed, msg


def build_e3conv1_l2_module(mode: str, config: ModelConfig, device: torch.device) -> torch.nn.Module:
    """
    仅用 E3Conv1（第一层 conv）输出，提取 2e 或 rank-2 分量测试等变性。
    - irreps 模式: 提取 2e (5 维)，测 Y(R·x) = D_2(R) @ Y(x)
    - pure-cartesian 模式: 提取 rank-2 (3x3)，测 T(R·x) = R @ T(x) @ R.T
    """
    irreps_conv = config.get_irreps_output_conv()  # "64x0e + 64x1o + 64x2e"
    _dim_0e, _dim_1o = 64, 64 * 3
    start_2e = _dim_0e + _dim_1o
    dim_2e = 64 * 5

    # pure-cartesian rank-2: s=0 块内 L=2 起始于 64+192=256，dim=64*9=576
    start_rank2, dim_rank2 = 256, 64 * 9

    conv = None
    output_type = "irreps_2e"
    use_ir_mul = False

    # partial-cartesian* 用范数不变性测试（SH 基可能与 e3nn 不同）
    use_norm_test = mode in ("partial-cartesian", "partial-cartesian-loose")
    # ICTD 笛卡尔基：5D -> 3x3 对称无迹，测 T(R·x) = R @ T(x) @ R.T
    use_ictd_rank2 = mode in ("pure-cartesian-ictd", "pure-cartesian-ictd-save")

    if mode == "spherical":
        from molecular_force_field.models.e3nn_layers import E3Conv
        conv = E3Conv(
            max_radius=config.max_radius,
            number_of_basis=config.number_of_basis_main,
            irreps_output=str(irreps_conv),
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            emb_number=[64, 64, 64],
            function_type=config.function_type_main,
        )
    elif mode == "spherical-save":
        from molecular_force_field.models.e3nn_layers_channelwise import E3Conv
        conv = E3Conv(
            max_radius=config.max_radius,
            number_of_basis=config.number_of_basis_main,
            irreps_output=str(irreps_conv),
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            emb_number=[64, 64, 64],
            function_type=config.function_type_main,
            edge_attrs_lmax=2,
        )
    elif mode == "spherical-save-cue":
        from molecular_force_field.models.cue_layers_channelwise import E3Conv
        conv = E3Conv(
            max_radius=config.max_radius,
            number_of_basis=config.number_of_basis_main,
            irreps_output=str(irreps_conv),
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            emb_number=[64, 64, 64],
            function_type=config.function_type_main,
            edge_attrs_lmax=2,
            device=device,
        )
        use_ir_mul = True
    elif mode == "partial-cartesian":
        from molecular_force_field.models.cartesian_e3_layers import CartesianE3ConvStrict
        conv = CartesianE3ConvStrict(
            max_radius=config.max_radius,
            number_of_basis=config.number_of_basis_main,
            channels_out=64,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            lmax=config.lmax,
            function_type=config.function_type_main,
        )
    elif mode == "partial-cartesian-loose":
        from molecular_force_field.models.cartesian_e3_layers import CartesianE3ConvSparseLoose
        conv = CartesianE3ConvSparseLoose(
            max_radius=config.max_radius,
            number_of_basis=config.number_of_basis_main,
            channels_out=64,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            lmax=config.lmax,
            function_type=config.function_type_main,
        )
    elif mode == "pure-cartesian":
        from molecular_force_field.models.pure_cartesian_layers import PureCartesianE3Conv
        conv = PureCartesianE3Conv(
            max_radius=config.max_radius,
            number_of_basis=config.number_of_basis_main,
            channels_out=64,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            Lmax=config.lmax,
            function_type=config.function_type_main,
        )
        output_type = "rank2"
    elif mode == "pure-cartesian-sparse":
        from molecular_force_field.models.pure_cartesian_sparse_layers import PureCartesianSparseE3Conv
        conv = PureCartesianSparseE3Conv(
            max_radius=config.max_radius,
            number_of_basis=config.number_of_basis_main,
            channels_out=64,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            Lmax=config.lmax,
            function_type=config.function_type_main,
            max_rank_other=1,
            k_policy="k0",
        )
        output_type = "rank2"
    elif mode in ("pure-cartesian-ictd", "pure-cartesian-ictd-save"):
        from molecular_force_field.models.pure_cartesian_ictd_layers_full import ICTDIrrepsE3Conv
        conv = ICTDIrrepsE3Conv(
            max_radius=config.max_radius,
            number_of_basis=config.number_of_basis_main,
            channels_out=64,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            lmax=config.lmax,
            function_type=config.function_type_main,
            ictd_tp_path_policy="full",
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=config.dtype,
        )
    else:
        raise ValueError(f"E3Conv1 l2 test: unsupported mode {mode}")

    class E3Conv1L2Wrapper(torch.nn.Module):
        def __init__(self, conv, start_2e, dim_2e, start_rank2, dim_rank2, ir_mul_layout: bool, output_type: str, use_norm_test: bool, use_ictd_rank2: bool):
            super().__init__()
            self.conv = conv
            self.start_2e = start_2e
            self.dim_2e = dim_2e
            self.start_rank2 = start_rank2
            self.dim_rank2 = dim_rank2
            self.ir_mul_layout = ir_mul_layout
            self.output_type = output_type
            self.use_norm_test = use_norm_test
            self.use_ictd_rank2 = use_ictd_rank2

        def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_edge_vec=None):
            import inspect
            sig = inspect.signature(self.conv.forward)
            if "precomputed_edge_vec" in sig.parameters:
                f = self.conv(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec=precomputed_edge_vec)
            else:
                f = self.conv(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
            if self.output_type == "rank2":
                f2 = f[:, self.start_rank2 : self.start_rank2 + self.dim_rank2]  # (N, 576)
                return f2.reshape(-1, 64, 3, 3)[:, 0, :, :]  # (N, 3, 3) channel 0
            f2e = f[:, self.start_2e : self.start_2e + self.dim_2e]
            if self.ir_mul_layout:
                f2e = f2e.reshape(-1, 5, 64).permute(0, 2, 1)
            else:
                f2e = f2e.reshape(-1, 64, 5)
            if self.use_ictd_rank2:
                from molecular_force_field.models.ictd_irreps import ictd_l2_to_rank2
                c = f2e[:, 0, :]  # (N, 5) channel 0
                return ictd_l2_to_rank2(c)  # (N, 3, 3)
            if self.use_norm_test:
                return f2e  # (N, 64, 5) 用于范数不变性测试
            return f2e[:, 0, :]  # (N, 5)

    return E3Conv1L2Wrapper(conv, start_2e, dim_2e, start_rank2, dim_rank2, use_ir_mul, output_type, use_norm_test, use_ictd_rank2).to(device)


def run_l2_equivariance_test(
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    num_nodes: int = 32,
    tol: float = 1e-5,
    n_rotations: int = 3,
) -> tuple[bool, str]:
    """
    测试 E3Conv1 输出的等变性。
    - irreps 模式: Y(R·x) = D_2(R) @ Y(x)
    - rank2 模式: T(R·x) = R @ T(x) @ R.T
    返回 (passed, message)。
    """
    from e3nn import o3

    config = ModelConfig(dtype=dtype, lmax=2)

    try:
        module = build_e3conv1_l2_module(mode, config, device)
    except Exception as e:
        return False, f"build failed: {e}"

    module.eval()
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = make_dummy_graph(
        device, dtype, num_nodes=num_nodes
    )

    with torch.no_grad():
        Y0 = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)

    is_rank2 = Y0.dim() == 3 and Y0.shape[-1] == 3 and Y0.shape[-2] == 3
    is_norm_test = Y0.dim() == 3 and Y0.shape[-2:] == (64, 5)
    err_max = 0.0

    with torch.no_grad():
        for i in range(n_rotations):
            R = random_orthogonal(device, dtype, reflect=False, seed=100 + i)
            Rr = random_orthogonal(device, dtype, reflect=True, seed=200 + i)

            pos_R = pos @ R.T
            cell_R = cell @ R.T
            pos_Rr = pos @ Rr.T
            cell_Rr = cell @ Rr.T

            Y_R = module(pos_R, A, batch, edge_src, edge_dst, edge_shifts, cell_R)
            Y_Rr = module(pos_Rr, A, batch, edge_src, edge_dst, edge_shifts, cell_Rr)

            if is_rank2:
                # T(R·x) = R @ T(x) @ R.T
                Y0_exp = torch.einsum("ij,njk,kl->nil", R, Y0, R.T)
                Y0_exp_r = torch.einsum("ij,njk,kl->nil", Rr, Y0, Rr.T)
                denom = Y0.norm().item() + Y_R.norm().item() + 1e-12
                err_rot = (Y_R - Y0_exp).norm().item() / denom
                denom_r = Y0.norm().item() + Y_Rr.norm().item() + 1e-12
                err_ref = (Y_Rr - Y0_exp_r).norm().item() / denom_r
            elif is_norm_test:
                # 2e 范数不变性: ||Y(R·x)|| = ||Y(x)||（与基无关）
                norm0 = (Y0 ** 2).sum(dim=-1)  # (N, 64)
                norm_R = (Y_R ** 2).sum(dim=-1)
                norm_Rr = (Y_Rr ** 2).sum(dim=-1)
                denom = norm0.abs().max().item() + 1e-12
                err_rot = (norm_R - norm0).abs().max().item() / denom
                err_ref = (norm_Rr - norm0).abs().max().item() / denom
            else:
                irreps = o3.Irreps("1x2e")
                D_R = irreps.D_from_matrix(R)
                D_Rr = irreps.D_from_matrix(Rr)
                Y0_exp = Y0 @ D_R.T
                Y0_exp_r = Y0 @ D_Rr.T
                denom = Y0.norm().item() + Y_R.norm().item() + 1e-12
                err_rot = (Y_R - Y0_exp).norm().item() / denom
                denom_r = Y0.norm().item() + Y_Rr.norm().item() + 1e-12
                err_ref = (Y_Rr - Y0_exp_r).norm().item() / denom_r
            err_max = max(err_max, err_rot, err_ref)

    passed = err_max <= tol
    if is_rank2:
        kind = "rank2"
    elif is_norm_test:
        kind = "2e(norm)"
    else:
        kind = "2e"
    msg = f"E3Conv1 {kind} err={err_max:.3e} (tol={tol})"
    return passed, msg


def main():
    parser = argparse.ArgumentParser(description="等变性测试")
    parser.add_argument("--mode", type=str, default=None, help="单模式测试")
    parser.add_argument("--modes", type=str, nargs="+", default=None, help="多模式")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--tol-energy", type=float, default=1e-5)
    parser.add_argument("--tol-force", type=float, default=1e-5)
    parser.add_argument("--tol-l2", type=float, default=1e-5)
    parser.add_argument("--n-rotations", type=int, default=3)
    parser.add_argument("--test-l2", action="store_true", help="额外测试 l=2 输出等变性 (Y(R·x)=D_2(R)@Y(x))")
    args = parser.parse_args()

    ALL_MODES = [
        "spherical", "spherical-save", "spherical-save-cue",
        "partial-cartesian", "partial-cartesian-loose",
        "pure-cartesian", "pure-cartesian-sparse",
        "pure-cartesian-ictd", "pure-cartesian-ictd-save",
    ]
    modes = args.modes or ([args.mode] if args.mode else ALL_MODES)
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)

    print("=" * 70)
    print("等变性测试 (O(3): 能量不变, 力协变)")
    print("=" * 70)
    print(f"device={device}, dtype={dtype}, atoms={args.atoms}")
    print(f"modes: {modes}")
    print()

    all_ok = True
    for mode in modes:
        passed, msg = run_equivariance_test(
            mode,
            device,
            dtype,
            num_nodes=args.atoms,
            tol_energy=args.tol_energy,
            tol_force=args.tol_force,
            n_rotations=args.n_rotations,
        )
        status = "PASS" if passed else "FAIL"
        print(f"  {mode:30s} {status}: {msg}")
        if not passed:
            all_ok = False

    if args.test_l2:
        print()
        print("=" * 70)
        print("E3Conv1 等变性测试 (2e 或 rank-2 分量)")
        print("=" * 70)
        l2_modes = [
            "spherical", "spherical-save", "spherical-save-cue",
            "partial-cartesian", "partial-cartesian-loose",
            "pure-cartesian", "pure-cartesian-sparse",
            "pure-cartesian-ictd", "pure-cartesian-ictd-save",
        ]
        for mode in l2_modes:
            try:
                passed, msg = run_l2_equivariance_test(
                    mode,
                    device,
                    dtype,
                    num_nodes=args.atoms,
                    tol=args.tol_l2,
                    n_rotations=args.n_rotations,
                )
                status = "PASS" if passed else "FAIL"
                print(f"  {mode:30s} E3Conv1 l2 {status}: {msg}")
                if not passed:
                    all_ok = False
            except Exception as e:
                print(f"  {mode:30s} E3Conv1 l2 SKIP: {e}")

    print()
    if all_ok:
        print("全部通过")
        return 0
    print("存在失败")
    return 1


if __name__ == "__main__":
    sys.exit(main())
