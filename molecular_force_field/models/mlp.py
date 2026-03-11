"""Multi-layer perceptron (MLP) network definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MainNet(nn.Module):
    """Main MLP network with layer normalization and SiLU activation."""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MainNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.input_norm = nn.LayerNorm(input_size)
        
        # Build hidden layers
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))
        
        # Connections between hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[i + 1]))
        
        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        
        # Initialize weights
        self._initialize_weights()
        
        # Convert to default dtype after initialization
        default_dtype = torch.get_default_dtype()
        if default_dtype == torch.float64:
            self.double()
        elif default_dtype == torch.float32:
            self.float()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Check if current layer is output layer
                if m == self.output:
                    # Output layer: initialize with small values
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                else:
                    # Hidden layers: use Kaiming initialization
                    torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
    
    def forward(self, M):
        x = self.input_norm(M)
        
        # Iterate through all hidden layers
        for layer, ln in zip(self.layers, self.layer_norms):
            x = layer(x)
            x = ln(x)
            x = F.silu(x)  # All hidden layers use activation function
        
        # Final linear output (no activation, no Norm)
        Y = self.output(x)
        return Y


class MainNet2(nn.Module):
    """Alternative MLP network with conservative initialization."""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MainNet2, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.input_norm = nn.LayerNorm(input_size)
        
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))
        
        # Connections between hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[i + 1]))
        
        # Output layer (smaller initialization)
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        self._initialize_weights()
        
        # Convert to default dtype after initialization
        default_dtype = torch.get_default_dtype()
        if default_dtype == torch.float64:
            self.double()
        elif default_dtype == torch.float32:
            self.float()
    
    def _initialize_weights(self):
        # Iterate through all linear layers, uniformly conservative initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
    
    def forward(self, M):
        x = self.input_norm(M)
        
        for i, (layer, ln) in enumerate(zip(self.layers, self.layer_norms)):
            x = layer(x)
            x = ln(x)
            if i < len(self.layers) - 1:
                x = F.silu(x)
        Y = self.output(x)
        return Y

class RobustScalarWeightedSum(nn.Module):
    """Robust scalar weighted sum module."""
    
    def __init__(self, num_features, init_weights=None):
        super().__init__()
        self.num_features = num_features

        if init_weights is None:
            self.weights = nn.Parameter(torch.randn(num_features) * 0.1)
        elif init_weights == 'zero':
            self.weights = nn.Parameter(torch.zeros(num_features))
        else:
            self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float))
        
    def forward(self, x):
        """Weighted sum of features."""
        weighted_features = x * self.weights
        return weighted_features.sum(dim=-1, keepdim=True)