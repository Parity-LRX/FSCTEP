"""Configuration management for model hyperparameters."""

from e3nn import o3
from dataclasses import dataclass
from typing import List, Tuple, Union
import torch
import pandas as pd
import os
import logging


@dataclass
class ModelConfig:
    """Model hyperparameters configuration."""
    # Data type configuration
    dtype: Union[torch.dtype, str] = torch.float64  # Default dtype for all tensors
    
    # Embedding network e3 layer parameters
    channel_in: int = 64
    channel_in2: int = 32
    max_atom: int = 5
    max_atomvalue: int = 10  # CLI configurable: maximum atomic number in embedding
    embedding_dim: int = 16  # CLI configurable: atom embedding dimension
    main_hidden_sizes3: List[int] = None  # MLP hidden layers after embedding
    embed_size: List[int] = None  # Embedding MLP sizes
    output_size: int = 8  # Ai and Aj MLP output layer
    feature_size: int = 8
    embed_size_2: int = 16  # O and B MLP hidden layer
    number_of_basis: int = 8  # Number of basis functions in e3nn
    max_radius: float = 5.0
    function_type: str = 'gaussian'  # CLI configurable: basis function type (gaussian, bessel, fourier, etc.)
    emb_number: List[int] = None  # e3MLP hidden layers
    
    # Irreps configuration (CLI configurable)
    irreps_output_conv_channels: int = None  # If set, overrides channel_in for get_irreps_output_conv()
    lmax: int = 2  # Maximum L value for spherical harmonics in irreps (CLI configurable)
    
    # Main network e3 layer parameters
    channel_in3: int = 32  # Q channel number
    channel_in4: int = 32  # K channel number
    channel_in5: int = 32  # V channel number
    num_layers: int = 1  # Transformer layer number
    number_of_basis_main: int = 8
    max_radius_main: float = 5.0
    function_type_main: str = 'gaussian'
    emb_number_main_2: List[int] = None  # Main network e3MLP hidden layers
    
    # Weight network parameters
    main_hidden_sizes4: List[int] = None  # Weight network MLP hidden layers
    input_dim_weight: int = 1  # Should match E3-transformer layer output channels
    
    # Atomic reference energies (keys, values)
    atomic_energy_keys: torch.Tensor = None
    atomic_energy_values: torch.Tensor = None
    
    def __post_init__(self):
        """Set default values for lists if None."""
        # Convert dtype string to torch.dtype if needed
        if isinstance(self.dtype, str):
            if self.dtype == 'float32' or self.dtype == 'float':
                self.dtype = torch.float32
            elif self.dtype == 'float64' or self.dtype == 'double':
                self.dtype = torch.float64
            else:
                raise ValueError(f"Unknown dtype: {self.dtype}")
        
        # Set global default dtype
        torch.set_default_dtype(self.dtype)
        
        if self.main_hidden_sizes3 is None:
            self.main_hidden_sizes3 = [64, 32]
        if self.embed_size is None:
            self.embed_size = [128, 128, 128]
        if self.emb_number is None:
            self.emb_number = [64, 64, 64]
        if self.emb_number_main_2 is None:
            self.emb_number_main_2 = [64, 64, 64]
        if self.main_hidden_sizes4 is None:
            self.main_hidden_sizes4 = [40, 40]
        # Note: atomic_energy_keys and atomic_energy_values are set in load_atomic_energies_from_file()
        # if not provided explicitly, to allow loading from fitted_E0.csv
    
    def load_atomic_energies_from_file(self, filepath='fitted_E0.csv'):
        """
        Load atomic energy keys and values from fitted_E0.csv file.
        If file doesn't exist or loading fails, sets default hardcoded values.
        
        Args:
            filepath: Path to the fitted_E0.csv file
            
        Returns:
            True if file was loaded successfully, False if using default values
        """
        if not os.path.exists(filepath):
            logging.warning(f"fitted_E0.csv not found at {filepath}, using default atomic energy values")
            # Set default hardcoded values if not already set
            if self.atomic_energy_keys is None:
                self.atomic_energy_keys = torch.tensor([1, 6, 7, 8], dtype=torch.long)
            if self.atomic_energy_values is None:
                self.atomic_energy_values = torch.tensor([
                    -430.53299511, -821.03326787, -1488.18856918, -2044.3509823
                ], dtype=self.dtype)
            return False
        
        try:
            df = pd.read_csv(filepath)
            if 'Atom' not in df.columns or 'E0' not in df.columns:
                logging.warning(f"fitted_E0.csv missing required columns (Atom, E0), using default values")
                # Set default hardcoded values if not already set
                if self.atomic_energy_keys is None:
                    self.atomic_energy_keys = torch.tensor([1, 6, 7, 8], dtype=torch.long)
                if self.atomic_energy_values is None:
                    self.atomic_energy_values = torch.tensor([
                        -430.53299511, -821.03326787, -1488.18856918, -2044.3509823
                    ], dtype=self.dtype)
                return False
            
            keys = df['Atom'].values.astype(int)
            values = df['E0'].values.astype(float)
            
            self.atomic_energy_keys = torch.tensor(keys, dtype=torch.long)
            self.atomic_energy_values = torch.tensor(values, dtype=self.dtype)
            
            logging.info(f"Loaded atomic energies from {filepath}:")
            for k, v in zip(keys, values):
                logging.info(f"  Atom {k}: {v:.8f} eV")
            
            return True
        except Exception as e:
            logging.warning(f"Failed to load fitted_E0.csv: {e}, using default atomic energy values")
            # Set default hardcoded values if not already set
            if self.atomic_energy_keys is None:
                self.atomic_energy_keys = torch.tensor([1, 6, 7, 8], dtype=torch.long)
            if self.atomic_energy_values is None:
                self.atomic_energy_values = torch.tensor([
                    -430.53299511, -821.03326787, -1488.18856918, -2044.3509823
                ], dtype=self.dtype)
            return False
    
    def get_irreps_input_conv(self) -> o3.Irreps:
        """Get input convolution irreps."""
        return o3.Irreps("10x0e + 10x1o + 10x2e")
    
    def get_irreps_output_conv(self) -> o3.Irreps:
        """Get output convolution irreps.
        
        Dynamically constructs irreps based on lmax and channels.
        For example:
        - lmax=2, channels=64 → "64x0e + 64x1o + 64x2e"
        - lmax=1, channels=64 → "64x0e + 64x1o"
        - lmax=3, channels=64 → "64x0e + 64x1o + 64x2e + 64x3o"
        """
        # Use irreps_output_conv_channels if set, otherwise use channel_in
        channels = self.irreps_output_conv_channels if self.irreps_output_conv_channels is not None else self.channel_in
        
        # Construct irreps string based on lmax
        # l=0 → 0e (even), l=1 → 1o (odd), l=2 → 2e (even), l=3 → 3o (odd), ...
        irreps_parts = []
        for l in range(self.lmax + 1):
            parity = 'e' if l % 2 == 0 else 'o'
            irreps_parts.append(f"{channels}x{l}{parity}")
        
        return o3.Irreps(" + ".join(irreps_parts))
    
    def get_irreps_output_conv_2(self) -> o3.Irreps:
        """Get output convolution 2 irreps.
        
        Uses channel_in2 and lmax to construct irreps.
        """
        irreps_parts = []
        for l in range(self.lmax + 1):
            parity = 'e' if l % 2 == 0 else 'o'
            irreps_parts.append(f"{self.channel_in2}x{l}{parity}")
        
        return o3.Irreps(" + ".join(irreps_parts))
    
    def get_irreps_input_conv_main(self) -> o3.Irreps:
        """Get main network input convolution irreps."""
        return o3.Irreps(f"{self.channel_in2}x0e + {self.channel_in2}x1o + {self.channel_in2}x2e")
    
    def get_irreps_query_main(self) -> o3.Irreps:
        """Get main network query irreps."""
        return o3.Irreps(f"{self.channel_in3}x0e + {self.channel_in3}x1o + {self.channel_in3}x2e")
    
    def get_irreps_key_main(self) -> o3.Irreps:
        """Get main network key irreps."""
        return o3.Irreps(f"{self.channel_in4}x0e + {self.channel_in4}x1o")
    
    def get_irreps_value_main(self) -> o3.Irreps:
        """Get main network value irreps."""
        return o3.Irreps(f"{self.channel_in5}x0e + {self.channel_in5}x1o + {self.channel_in3}x2e")
    
    def get_irreps_sh_transformer(self) -> o3.Irreps:
        """Get spherical harmonics irreps for transformer.
        
        Uses lmax to construct spherical harmonics irreps dynamically.
        """
        return o3.Irreps.spherical_harmonics(lmax=self.lmax)
    
    def get_hidden_dim_sh(self) -> o3.Irreps:
        """Get hidden dimension spherical harmonics irreps."""
        return o3.Irreps("32x0e")