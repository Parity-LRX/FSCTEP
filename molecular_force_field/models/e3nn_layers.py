"""E3NN-based neural network layers for molecular modeling."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn import nn as e3nn_nn
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import S2Activation
from e3nn.nn import Gate
from molecular_force_field.utils.scatter import scatter

from molecular_force_field.models.mlp import MainNet2, MainNet, RobustScalarWeightedSum

class E3Conv(nn.Module):
    """E3NN-based convolutional layer for molecular graph neural networks."""
    
    def __init__(self, max_radius, number_of_basis, irreps_output,
                 embedding_dim=16, max_atomvalue=10, output_size=8, feature_size=8,
                 embed_size_2=16, main_hidden_sizes3=None, embed_size=None,
                 emb_number=None, function_type='gaussian', atomic_energy_keys=None):
        super(E3Conv, self).__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        self.function_type = function_type
        
        # Default values
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if embed_size is None:
            embed_size = [128, 128, 128]
        if emb_number is None:
            emb_number = [64, 64, 64]
        if atomic_energy_keys is None:
            atomic_energy_keys = torch.tensor([1, 6, 7, 8])
        
        # Define atom embedding layer
        # Use current default dtype
        default_dtype = torch.get_default_dtype()
        self.atom_embedding = nn.Embedding(
            num_embeddings=max_atomvalue,
            embedding_dim=embedding_dim,
            dtype=default_dtype
        )
        self.fitnet1 = MainNet2(input_size=embedding_dim, hidden_sizes=main_hidden_sizes3, output_size=output_size)
        
        # Normalization layers
        self.norm_ai = nn.LayerNorm(output_size)
        self.norm_aj = nn.LayerNorm(output_size)
        self.norm_feature = nn.LayerNorm(64)
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * output_size + 1, embed_size_2),
            nn.LayerNorm(embed_size_2),
            nn.SiLU(),
            nn.Linear(embed_size_2, 2 * len(atomic_energy_keys) + 1)
        )

        self.tensor_product = o3.FullTensorProduct(
            irreps_in1=f"{output_size}x0e",
            irreps_in2="1x0e + 1x1o + 1x2e",
        )
        self.irreps_out = self.tensor_product.irreps_out

        # Initialize TensorProduct and FullyConnectedNet
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_out,
            irreps_in2=f"{output_size}x0e",
            irreps_out=self.irreps_output,
            shared_weights=False
        )
        self.fc = e3nn_nn.FullyConnectedNet(
            [number_of_basis] + emb_number + [self.tp.weight_numel],
            torch.nn.functional.silu
        )
        self.reset_parameters()
        # Convert to default dtype after initialization
        if default_dtype == torch.float64:
            self.double()
        elif default_dtype == torch.float32:
            self.float()
        print("E3Conv initialization complete.")
    
    def reset_parameters(self):
        with torch.no_grad():
            for _, module in self.named_children():
                if isinstance(module, o3.FullyConnectedTensorProduct):
                    self._init_tensor_product(module)
                elif isinstance(module, nn.Linear):
                    self._init_dense(module)
    
    def _init_tensor_product(self, module):
        if module.weight_numel > 0:
            std = 1 / math.sqrt(module.irreps_in1.dim * module.irreps_in2.dim)
            module.weight.data.uniform_(-std, std)
            if module.internal_weights and hasattr(module, 'shift'):
                module.shift.data.zero_()
    
    def _init_dense(self, module):
        nn.init.kaiming_normal_(module.weight, nonlinearity='silu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_edge_vec=None):
        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum('ni,nij->nj', edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)

        num_nodes = pos.size(0)
        atom_embeddings = self.atom_embedding(A.long())
        Ai = self.fitnet1(atom_embeddings)
        
        sh_edge = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization='component'
        )
        f_in = self.tensor_product(Ai[edge_src], sh_edge)
        
        # Calculate spherical harmonics and basis functions
        emb = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis=self.function_type,
            cutoff=True
        ).mul(self.number_of_basis ** 0.5)

        # Use TensorProduct to process neighbor features
        edge_features = self.tp(f_in, Ai[edge_dst], self.fc(emb))
        neighbor_count = scatter(
            torch.ones_like(edge_dst),
            edge_dst,
            dim=0,
            dim_size=num_nodes
        ).clamp(min=1).float()

        # Parallel scatter and normalize
        out = scatter(
            edge_features,
            edge_dst,
            dim=0,
            dim_size=num_nodes
        ).div(neighbor_count.unsqueeze(-1))  # Normalize per node
        
        assert out.shape == (num_nodes, edge_features.shape[1])
        return out


class E3Conv2(nn.Module):
    """Second E3NN-based convolutional layer."""
    
    def __init__(self, max_radius, number_of_basis, irreps_input_conv, irreps_output,
                 embedding_dim=16, max_atomvalue=10, output_size=8, feature_size=8,
                 embed_size_2=16, main_hidden_sizes3=None, embed_size=None,
                 emb_number=None, function_type='gaussian', atomic_energy_keys=None):
        super(E3Conv2, self).__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        self.function_type = function_type
        
        # Default values
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if embed_size is None:
            embed_size = [128, 128, 128]
        if emb_number is None:
            emb_number = [64, 64, 64]
        if atomic_energy_keys is None:
            atomic_energy_keys = torch.tensor([1, 6, 7, 8])
        
        # Initialize TensorProduct and FullyConnectedNet
        self.tensor_product = o3.FullyConnectedTensorProduct(
            irreps_in1="3x0e",
            irreps_in2="1x0e + 1x1o + 1x2e",
            irreps_out=self.irreps_sh,
        )
        self.irreps_out = self.tensor_product.irreps_out
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_input_conv,
            self.irreps_out,
            self.irreps_output,
            shared_weights=False
        )
        self.fc = e3nn_nn.FullyConnectedNet(
            [number_of_basis] + emb_number + [self.tp.weight_numel],
            torch.nn.functional.silu
        )
        
        # Define atom embedding layer
        # Use current default dtype
        default_dtype = torch.get_default_dtype()
        self.atom_embedding = nn.Embedding(
            num_embeddings=max_atomvalue,
            embedding_dim=embedding_dim,
            dtype=default_dtype
        )
        self.norm_feature = e3nn_nn.BatchNorm(irreps=f"{3}x0e")
        
        # Normalization layers
        self.norm_ai = nn.LayerNorm(output_size)
        self.norm_aj = nn.LayerNorm(output_size)
        self.mlp = nn.Sequential(
            nn.Linear(2 * output_size + 1, embed_size_2),
            nn.SiLU(),
            nn.Linear(embed_size_2, 2 * len(atomic_energy_keys) + 1)
        )

        self.reset_parameters()
        # Convert to default dtype after initialization
        if default_dtype == torch.float64:
            self.double()
        elif default_dtype == torch.float32:
            self.float()
        print("E3Conv2 initialization complete.")
    
    def reset_parameters(self):
        with torch.no_grad():
            for _, module in self.named_children():
                if isinstance(module, o3.FullyConnectedTensorProduct):
                    self._init_tensor_product(module)
                elif isinstance(module, nn.Linear):
                    self._init_dense(module)
    
    def _init_tensor_product(self, module):
        if module.weight_numel > 0:
            std = 1 / math.sqrt(module.irreps_in1.dim * module.irreps_in2.dim)
            module.weight.data.uniform_(-std, std)
            if module.internal_weights and hasattr(module, 'shift'):
                module.shift.data.zero_()
    
    def _init_dense(self, module):
        nn.init.kaiming_normal_(module.weight, nonlinearity='silu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def forward(self, f_in, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_edge_vec=None):
        assert not torch.isnan(pos).any(), "Input 'pos' contains NaN values."
        assert not torch.isinf(pos).any(), "Input 'pos' contains Inf values."

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum('ni,nij->nj', edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)

        num_nodes = pos.size(0)
        atom_embeddings = self.atom_embedding(A.long())
        edge_src_emb = atom_embeddings[edge_src]
        
        sh_edge = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization='component'
        )
        Feature = sh_edge

        # Calculate spherical harmonics and basis functions
        emb = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis=self.function_type,
            cutoff=True
        ).mul(self.number_of_basis ** 0.5)

        # Use TensorProduct to process neighbor features
        edge_features = self.tp(f_in[edge_src], Feature, self.fc(emb))

        neighbor_count = scatter(
            torch.ones_like(edge_dst),
            edge_dst,
            dim=0,
            dim_size=num_nodes
        ).clamp(min=1).float()

        # Parallel scatter and normalize
        out = scatter(
            edge_features,
            edge_dst,
            dim=0,
            dim_size=num_nodes
        ).div(neighbor_count.unsqueeze(-1))  # Normalize per node
        
        assert out.shape == (num_nodes, edge_features.shape[1])
        return out


class E3_TransformerLayer_multi(nn.Module):
    """Multi-layer E3NN Transformer for molecular modeling."""
    
    def __init__(self, max_embed_radius, main_max_radius, main_number_of_basis,
                 irreps_input, irreps_query, irreps_key, irreps_value, irreps_output,
                 irreps_sh, hidden_dim_sh, hidden_dim, channel_in2=32, embedding_dim=16,
                 max_atomvalue=10, output_size=8, embed_size=None, main_hidden_sizes3=None,
                 num_layers=1, device=None, function_type_main='gaussian', num_interaction=2):
        super(E3_TransformerLayer_multi, self).__init__()
        
        # Default values
        if embed_size is None:
            embed_size = [128, 128, 128]
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # emb_number should be a list, not an integer
        # Use default [64, 64, 64] if hidden_dim is not a list
        emb_number = [64, 64, 64] if not isinstance(hidden_dim, list) else hidden_dim
        
        # Input irreps (scalar part)
        irreps_scalars = o3.Irreps(f"{32}x0e + {32}x0e")
        # Gate signal irreps
        irreps_gates = o3.Irreps(f"{channel_in2}x0e + {channel_in2}x0e")
        # Gated irreps (higher order part)
        irreps_gated = o3.Irreps(f"{channel_in2}x1o + {channel_in2}x2e")
        
        self.gate_layer = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[torch.tanh, torch.tanh],
            irreps_gates=irreps_gates,
            act_gates=[torch.tanh, torch.tanh],
            irreps_gated=irreps_gated
        )
        self.irreps_output_conv = self.gate_layer.irreps_in
        self.irreps_input = self.gate_layer.irreps_out
        self.irreps_sh = irreps_sh
        self.max_radius = main_max_radius
        self.number_of_basis = main_number_of_basis
        
        
        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")

        # E3 convolution layers (one module per interaction)
        # First layer uses E3Conv; subsequent layers use E3Conv2
        self.e3_conv_layers = nn.ModuleList()
        self.e3_conv_layers.append(
            E3Conv(
                max_radius=max_embed_radius,
                number_of_basis=main_number_of_basis,
                irreps_output=irreps_input,
                embedding_dim=embedding_dim,
                max_atomvalue=max_atomvalue,
                output_size=output_size,
                main_hidden_sizes3=main_hidden_sizes3,
                embed_size=embed_size,
                emb_number=emb_number,
                atomic_energy_keys=torch.tensor([1, 6, 7, 8], device=self.device),
            )
        )
        for _ in range(1, self.num_interaction):
            self.e3_conv_layers.append(
                E3Conv2(
                    max_radius=max_embed_radius,
                    number_of_basis=main_number_of_basis,
                    irreps_input_conv=irreps_input,
                    irreps_output=irreps_input,
                    embedding_dim=embedding_dim,
                    max_atomvalue=max_atomvalue,
                    output_size=output_size,
                    main_hidden_sizes3=main_hidden_sizes3,
                    embed_size=embed_size,
                    emb_number=emb_number,
                    atomic_energy_keys=torch.tensor([1, 6, 7, 8], device=self.device),
                )
            )
        self.f2_proj = o3.Linear(self.irreps_output_conv, self.irreps_output_conv)

        # Product layers
        self.product_1 = o3.FullyConnectedTensorProduct(
            self.irreps_output_conv,
            self.irreps_output_conv,
            "16x0e",
            shared_weights=True,
            internal_weights=True,
            normalization='component'
        )
        
        irreps_input_multi = o3.Irreps(irreps_input) * self.num_interaction
        scalar_channels = (self.num_interaction - 1) * 32
        self.product_3 = o3.FullyConnectedTensorProduct(
            irreps_input_multi,
            irreps_input_multi,
            f"{scalar_channels}x0e",
            shared_weights=True,
            internal_weights=True,
            normalization='component'
        )

        self.product_2 = o3.FullyConnectedTensorProduct(
            self.product_3.irreps_out,
            self.product_3.irreps_out,
            "64x0e",
            shared_weights=True,
            internal_weights=True,
            normalization='component'
        )

        irreps_product_5 = irreps_input_multi + self.product_3.irreps_out
        self.product_5 = o3.ElementwiseTensorProduct(
            irreps_product_5,
            irreps_product_5,
            ["0e"],
            normalization='component'
        )


        self.proj_total = MainNet(self.product_5.irreps_out.dim, embed_size, 17)
        self.weight_mlp = MainNet2(main_number_of_basis * 2, [64, 32, 16], 1)
        self.batch_norm = e3nn_nn.BatchNorm(irreps=self.irreps_input)
        
        # Define atom embedding layer
        # Use current default dtype
        default_dtype = torch.get_default_dtype()
        self.atom_embedding = nn.Embedding(
            num_embeddings=max_atomvalue,
            embedding_dim=embedding_dim,
            dtype=default_dtype
        )
        
        self.linear_layer1 = o3.Linear(self.irreps_output_conv, "1x0e")
        self.linear_layer2 = o3.Linear(self.irreps_output_conv, "1x0e")
        self.linear_layer3 = o3.Linear(self.irreps_input, "1x0e")
        
        # Transformer independent linear layer parameters
        self.linear_layers = nn.ModuleList([
            o3.Linear(self.irreps_input, self.irreps_input)
            for _ in range(num_layers)
        ])
        
        self.linear_layers4 = nn.ModuleList([
            o3.Linear(self.irreps_input, "1x0e")
            for _ in range(num_layers)
        ])

        if isinstance(hidden_dim_sh, int):
            hidden_dim_sh_irreps = f"{hidden_dim_sh}x0e"
        else:
            hidden_dim_sh_irreps = hidden_dim_sh
        self.linear_layer_2 = o3.Linear(self.irreps_input, hidden_dim_sh_irreps)
        self.non_linearity = nn.SiLU()
        self.linear_layer_3 = o3.Linear(hidden_dim_sh_irreps, "1x0e")
        
        self.tp_featrue = o3.FullyConnectedTensorProduct(
            irreps_in1="16x0e",
            irreps_in2="1x0e + 1x1o + 1x2e",
            irreps_out=self.irreps_sh,
            shared_weights=True,
            internal_weights=True,
            normalization="component"
        )

        self.tensor_product = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_input,
            irreps_in2=self.irreps_input,
            irreps_out=self.irreps_input,
            shared_weights=True,
            internal_weights=True,
            normalization="norm"
        )

        self.num_features = 17

        self.readout = o3.TensorSquare(
            irreps_in="6x0e + 3x1o + 2x2e",
            irreps_out=f"{self.num_features}x0e",
        )
        self.weighted_sum = RobustScalarWeightedSum(self.num_features, init_weights='zero')
        
        self.reset_parameters()
        # Convert to default dtype after initialization (default_dtype was defined above)
        if default_dtype == torch.float64:
            self.double()
        elif default_dtype == torch.float32:
            self.float()
        print("E3_TransformerLayer_multi initialization complete.")
    
    def reset_parameters(self):
        with torch.no_grad():
            for _, module in self.named_children():
                if isinstance(module, o3.FullyConnectedTensorProduct):
                    self._init_tensor_product(module)
                elif isinstance(module, nn.Linear):
                    self._init_dense(module)
    
    def _init_tensor_product(self, module):
        if module.weight_numel > 0:
            std = 1 / math.sqrt(module.irreps_in1.dim * module.irreps_in2.dim)
            module.weight.data.uniform_(-std, std)
            if module.internal_weights and hasattr(module, 'shift'):
                module.shift.data.zero_()
    
    def _init_dense(self, module):
        nn.init.kaiming_normal_(module.weight, nonlinearity='silu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_edge_vec=None):
        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec[sort_idx]
        else:
            edge_vec = None

        features = []
        f_prev = self.e3_conv_layers[0](pos, A, batch, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec=edge_vec)
        features.append(f_prev)
        for conv in self.e3_conv_layers[1:]:
            f_prev = conv(f_prev, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec=edge_vec)
            features.append(f_prev)

        if edge_vec is None:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum('ni,nij->nj', edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        
        f_combine = torch.cat(features, dim=-1)
        f_combine_product = self.product_3(f_combine, f_combine)
        
        T = torch.cat(features + [f_combine_product], dim=-1)
        f2_product_5 = self.product_5(T, T)

        product_proj = self.proj_total(f2_product_5)

        e_out = torch.cat([product_proj], dim=-1)
        e_out = (1) * self.weighted_sum(e_out)
        atom_energies = e_out.sum(dim=-1, keepdim=True)

        return atom_energies