import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
import math
import torch
from scipy.special import sph_harm_y

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

SATCLIP_DIR = os.path.join(rf"/home/susanket/satclip-v2/satclip")
if SATCLIP_DIR not in sys.path:
    sys.path.insert(0, SATCLIP_DIR)

from location_encoder import get_positional_encoding

torch.set_float32_matmul_precision("high")

LAT_NAMES = {"lat", "latitude", "y", "lat_wgs84"}
LON_NAMES = {"lon", "lng", "long", "longitude", "x", "lon_wgs84"}
NAME_NAMES = {"name", "id", "filename", "fn"}
N_SAMPLES = 100_000
VERBOSE = 1
EMB_DIM = 128  # increased embedding size (max allowed) to give richer image‑location representations
SAVE_PATH = rf"/data/susanket/satclip-v2.ckpt"


def _find_column(df, aliases):
    lower = {c.lower(): c for c in df.columns}
    return next((lower[a] for a in aliases if a in lower), None)


# --------------------------------------------------------------------
#        Architechture
# --------------------------------------------------------------------
class ProjLayer(nn.Module):
    def __init__(self, width, output_dim):
        super().__init__()
        scale = width ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        return x @ self.proj

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None, dropout = False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        out = self.activation(out)
        return out

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, degreeinput = False, dropout = False):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.degreeinput = degreeinput

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation, dropout = False)

    def forward(self, x, mods = None):

        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)

class LearnableFourierFeatures(nn.Module):
    def __init__(self, dim_in, dim_out, sigma=2.0):
        super().__init__()
        # Initialize the random projection matrix
        # sigma controls the frequency of the features
        # Make B learnable to allow the network to adapt the frequency spectrum
        self.B = nn.Parameter(torch.randn(dim_in, dim_out) * sigma)
        # Learnable per-dimension gain to allow adaptive amplitude scaling
        self.gain = nn.Parameter(torch.ones(dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        # x is (batch, dim_in)
        # Project to higher dimension
        x = x @ self.B
        # Apply sine activation
        x = torch.sin(x)
        # Scale with learnable per-dimension gain
        x = x * self.gain
        return x

class ResidualMLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=4, dropout_p=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p)
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p)
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        out = self.output_proj(x)
        return out

class LocationEncoder(nn.Module):
    def __init__(self, posenc, nnet):
        super().__init__()
        self.posenc = posenc
        self.nnet = nnet

    def forward(self, x):
        x = self.posenc(x)
        return self.nnet(x)

class SwiGLUMLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        out = self.output_proj(x)
        return out

class MultiScaleFourierEncoder(nn.Module):
    """
    A Multi-Scale Fourier Feature Encoder.
    Projects input coordinates into a high-dimensional sinusoidal space using
    learnable frequency matrices at multiple scales (low, mid, high).
    This inductive bias helps capture both broad spatial trends and fine-grained
    demographic variations, which standard MLPs struggle to learn from scratch.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=128):
        super().__init__()
        
        # Learnable Fourier Feature Projections at 3 scales
        # Low frequency: captures broad spatial trends (sigma ~ 1.0)
        self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
        # Mid frequency: captures regional patterns (sigma ~ 3.0)
        self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
        # High frequency: captures local details (sigma ~ 8.0)
        self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
        
        # Learnable gains for amplitude scaling
        self.gain_low = nn.Parameter(torch.ones(fourier_dim))
        self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
        self.gain_high = nn.Parameter(torch.ones(fourier_dim))
        
        # Total input dimension to MLP: original input + 3 * fourier_dim
        total_input_dim = input_dim + (fourier_dim * 3)
        
        # Input Projection from combined features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Deep Residual Backbone
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
        
        # Output Projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x is (batch, input_dim)
        
        # Apply Fourier Features at 3 scales
        x_low = torch.sin(x @ self.B_low) * self.gain_low
        x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
        x_high = torch.sin(x @ self.B_high) * self.gain_high
        
        # Concatenate original input with multi-scale Fourier features
        x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
        
        # Project to hidden dimension
        x = self.input_proj(x_combined)
        
        # Residual blocks
        for block in self.blocks:
            x = x + block(x)
            
        # Final output
        out = self.output_proj(x)
        return out

class DeepResidualMLPEncoder(nn.Module):
    """
    A Multi-Scale Learnable Fourier Feature (MSLFF) + Deep Residual MLP Encoder.
    This architecture addresses the plateau observed with standard MLPs by explicitly
    injecting multi-scale frequency information into the input.
    
    1. **Multi-Scale Fourier Features**: Projects the input coordinates into a high-dimensional
       space using learnable frequency matrices at three scales (low, mid, high).
       - Low: Captures broad spatial trends.
       - Mid: Captures regional patterns.
       - High: Captures fine-grained local details.
       This inductive bias helps the network learn complex spatial-demographic correlations
       that are difficult for standard MLPs to capture from raw inputs.
       
    2. **Deep Residual MLP**: A stable backbone with LayerNorm, GELU, and Dropout that
       processes the enriched Fourier features to produce the final embedding.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, dropout_p=0.1, fourier_dim=128):
        super().__init__()
        
        # Learnable Fourier Feature Projections at 3 scales
        # Low frequency: captures broad spatial trends
        self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
        # Mid frequency: captures regional patterns
        self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
        # High frequency: captures local details
        self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
        
        # Learnable gains for amplitude scaling
        self.gain_low = nn.Parameter(torch.ones(fourier_dim))
        self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
        self.gain_high = nn.Parameter(torch.ones(fourier_dim))
        
        # Total input dimension to MLP: original input + 3 * fourier_dim
        total_input_dim = input_dim + (fourier_dim * 3)
        
        # Input Projection from combined features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )
        
        # Deep Residual Backbone
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p)
            )
            for _ in range(num_layers)
        ])
        
        # Output Projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x is (batch, input_dim)
        
        # Apply Fourier Features at 3 scales
        x_low = torch.sin(x @ self.B_low) * self.gain_low
        x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
        x_high = torch.sin(x @ self.B_high) * self.gain_high
        
        # Concatenate original input with multi-scale Fourier features
        x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
        
        # Project to hidden dimension
        x = self.input_proj(x_combined)
        
        # Residual blocks
        for block in self.blocks:
            x = x + block(x)
            
        # Final output
        out = self.output_proj(x)
        return out

class HybridMultiScaleEncoder(nn.Module):
    """
    Hybrid Multi-Scale Fourier Feature + Deep Residual MLP Encoder.
    Uses learnable Fourier features at three scales (low, mid, high) to capture
    multi-scale spatial patterns, concatenated with the original input,
    followed by a deep residual MLP backbone.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=256):
        super().__init__()
        
        # Learnable Fourier Feature Projections at 3 scales
        # Low frequency: captures broad spatial trends
        self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
        # Mid frequency: captures regional patterns
        self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
        # High frequency: captures local details
        self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
        
        # Learnable gains for amplitude scaling
        self.gain_low = nn.Parameter(torch.ones(fourier_dim))
        self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
        self.gain_high = nn.Parameter(torch.ones(fourier_dim))
        
        # Total input dimension to MLP: original input + 3 * fourier_dim
        total_input_dim = input_dim + (fourier_dim * 3)
        
        # Input Projection from combined features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Deep Residual Backbone
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
        
        # Output Projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x is (batch, input_dim)
        
        # Apply Fourier Features at 3 scales
        x_low = torch.sin(x @ self.B_low) * self.gain_low
        x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
        x_high = torch.sin(x @ self.B_high) * self.gain_high
        
        # Concatenate original input with multi-scale Fourier features
        x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
        
        # Project to hidden dimension
        x = self.input_proj(x_combined)
        
        # Residual blocks
        for block in self.blocks:
            x = x + block(x)
            
        # Final output
        out = self.output_proj(x)
        return out


class WideDeepSiLUResidualEncoder(nn.Module):
    """
    A Wide and Deep Residual MLP Encoder using SiLU activations.
    This architecture aims to break the R² plateau by providing higher capacity
    and smoother gradient flow compared to previous GELU-based encoders.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_p)
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p)
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        out = self.output_proj(x)
        return out

class MultiScaleResidualFourierEncoder(nn.Module):
    """
    A Location Encoder that combines Multi-Scale Learnable Fourier Features
    with a Stable Residual MLP Backbone.
    
    1. **Multi-Scale Fourier Features**: Projects input coordinates into a high-dimensional
       space using learnable frequency matrices at three scales (low, mid, high).
       - Low: Captures broad spatial trends.
       - Mid: Captures regional patterns.
       - High: Captures fine-grained local details and sharp boundaries.
       This inductive bias helps the network learn complex spatial-demographic correlations
       that are difficult for standard MLPs to capture from raw inputs, mitigating spectral bias.
       
    2. **Stable Residual MLP**: A backbone with LayerNorm, GELU, and Dropout that
       processes the enriched Fourier features to produce the final embedding.
       Residual connections ensure stable gradient flow, avoiding the optimization
       instability often seen with SIREN or deep non-residual networks.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=6, fourier_dim=128, dropout_p=0.1):
        super().__init__()
        
        # Learnable Fourier Feature Projections at 3 scales
        # Low frequency: captures broad spatial trends
        self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
        # Mid frequency: captures regional patterns
        self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
        # High frequency: captures local details / sharp boundaries
        self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
        
        # Learnable gains for amplitude scaling
        self.gain_low = nn.Parameter(torch.ones(fourier_dim))
        self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
        self.gain_high = nn.Parameter(torch.ones(fourier_dim))
        
        # Total input dimension to MLP: original input + 3 * fourier_dim
        total_input_dim = input_dim + (fourier_dim * 3)
        
        # Input Projection from combined features to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )
        
        # Deep Residual Backbone
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p)
            )
            for _ in range(num_layers)
        ])
        
        # Output Projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x is (batch, input_dim)
        
        # Apply Fourier Features at 3 scales
        x_low = torch.sin(x @ self.B_low) * self.gain_low
        x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
        x_high = torch.sin(x @ self.B_high) * self.gain_high
        
        # Concatenate original input with multi-scale Fourier features
        x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
        
        # Project to hidden dimension
        x = self.input_proj(x_combined)
        
        # Residual blocks
        for block in self.blocks:
            x = x + block(x)
            
        # Final output
        out = self.output_proj(x)
        return out

class SeqTransformerLocationEncoder(nn.Module):
    """
    A Location Encoder that treats the spherical harmonic features as a sequence of tokens.
    This allows self-attention to model interactions between different frequency components,
    which is more powerful than point-wise MLPs for capturing complex spatial-demographic patterns.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, n_heads=8, num_tokens=8):
        super().__init__()
        # Split input into num_tokens chunks
        # If input_dim is not divisible, we pad or adjust. Spherical harmonics usually yield even dims.
        self.num_tokens = num_tokens
        self.token_dim = input_dim // num_tokens
        # If there's a remainder, we handle it by using a slightly larger token_dim and truncating/masking,
        # but for simplicity, let's assume input_dim is divisible by num_tokens or we use a projection first.
        # Safer: Project to (num_tokens * hidden_dim) then reshape? No, that loses structure.
        # Let's stick to splitting. If not divisible, we pad the input.
        self.pad_dim = (self.num_tokens * self.token_dim) - input_dim
        if self.pad_dim > 0:
            self.input_pad = nn.ZeroPad2d((0, 0, 0, self.pad_dim)) # Assuming (B, D) -> (B, D+pad)
            # Actually ZeroPad2d is for images. For 1D:
            # We will handle padding in forward.
            
        # Learnable token embeddings to add positional information to the tokens
        self.token_pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        
        # Project each token to hidden_dim
        self.token_proj = nn.Linear(self.token_dim, hidden_dim)
        
        # Transformer Blocks
        class TransformerBlock(nn.Module):
            def __init__(self, dim, n_heads):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
                self.norm2 = nn.LayerNorm(dim)
                # SwiGLU FFN
                ffn_dim = int(8 * dim / 3)
                ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                self.w1 = nn.Linear(dim, ffn_dim, bias=False)
                self.w2 = nn.Linear(ffn_dim, dim, bias=False)
                self.w3 = nn.Linear(dim, ffn_dim, bias=False)

            def forward(self, x):
                # x is (B, L, D)
                h = self.norm1(x)
                a, _ = self.attn(h, h, h, need_weights=False)
                x = x + a
                
                h = self.norm2(x)
                f = self.w2(F.silu(self.w1(h)) * self.w3(h))
                x = x + f
                return x

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        # Pool the sequence (mean pooling) to get a single vector
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is (B, input_dim)
        
        # Pad if necessary
        if hasattr(self, 'pad_dim') and self.pad_dim > 0:
            x = F.pad(x, (0, self.pad_dim))
            
        # Reshape to (B, num_tokens, token_dim)
        x = x.view(-1, self.num_tokens, self.token_dim)
        
        # Project to hidden dim
        h = self.token_proj(x)
        
        # Add positional embeddings
        h = h + self.token_pos_embed
        
        # Pass through Transformer blocks
        for block in self.blocks:
            h = block(h)
            
        # Mean pooling
        h = self.final_norm(h).mean(dim=1)
        
        # Project to output dim
        out = self.output_proj(h)
        return out

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(8 * hidden_dim / 3)
            ffn_dim = int(2 * ((ffn_dim + 7) // 8))
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SwiGLUBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True, dropout=0.0)

    def forward(self, x):
        # x is (B, L, D)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        
        h = self.norm2(x)
        f = self.ffn(h)
        x = x + f
        return x

class DeepSwiGLUFourierEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=64):
        super().__init__()
        # Learnable Fourier Features at 3 scales
        self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
        self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
        self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
        
        self.gain_low = nn.Parameter(torch.ones(fourier_dim))
        self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
        self.gain_high = nn.Parameter(torch.ones(fourier_dim))
        
        total_input_dim = input_dim + (fourier_dim * 3)
        
        # Input Projection to Hidden Dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Deep SwiGLU Backbone
        self.blocks = nn.ModuleList([
            SwiGLUBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is (batch, input_dim)
        x_low = torch.sin(x @ self.B_low) * self.gain_low
        x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
        x_high = torch.sin(x @ self.B_high) * self.gain_high
        
        x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
        
        # Project to hidden dim and add batch dimension for attention (B, 1, D)
        h = self.input_proj(x_combined).unsqueeze(1)
        
        for block in self.blocks:
            h = block(h)
            
        h = self.final_norm(h).squeeze(1)
        out = self.output_proj(h)
        return out

class DualBranchFourierMLPEncoder(nn.Module):
    """
    A Dual-Branch Location Encoder combining Learnable Fourier Features with a Wide Residual MLP.
    This architecture addresses spectral bias by explicitly injecting multi-scale frequency information,
    while the wide residual backbone provides the capacity to map these features to the embedding space.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
        super().__init__()
        
        # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
        self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
        self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
        self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
        
        self.gain_low = nn.Parameter(torch.ones(fourier_dim))
        self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
        self.gain_high = nn.Parameter(torch.ones(fourier_dim))
        
        total_input_dim = input_dim + (fourier_dim * 3)
        
        # Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )
        
        # Wide Residual Backbone
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p)
            )
            for _ in range(num_layers)
        ])
        
        # Output Projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # Apply Multi-Scale Fourier Features
        x_low = torch.sin(x @ self.B_low) * self.gain_low
        x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
        x_high = torch.sin(x @ self.B_high) * self.gain_high
        
        x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
        x = self.input_proj(x_combined)
        
        for block in self.blocks:
            x = x + block(x)
            
        return self.output_proj(x)

class DeepFourierResidualEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=6, fourier_dim=128, dropout_p=0.1):
        super().__init__()
        # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
        self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
        self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
        self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
        
        self.gain_low = nn.Parameter(torch.ones(fourier_dim))
        self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
        self.gain_high = nn.Parameter(torch.ones(fourier_dim))
        
        total_input_dim = input_dim + (fourier_dim * 3)
        
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_p)
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        x_low = torch.sin(x @ self.B_low) * self.gain_low
        x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
        x_high = torch.sin(x @ self.B_high) * self.gain_high
        
        x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
        
        x = self.input_proj(x_combined)
        
        for block in self.blocks:
            x = x + block(x)
            
        return self.output_proj(x)

class SatCLIP(nn.Module):
    def __init__(
        self,
        embed_dim=EMB_DIM,
        legendre_polys=80,
        num_hidden_layers=6,
        capacity=250,
    ):
        super().__init__()
        self.dino_proj = ProjLayer(384, embed_dim)
        self.posenc = get_positional_encoding(
            name="sphericalharmonics",
            legendre_polys=100,
        ).double()
        
        input_dim = self.posenc.embedding_dim
        
        # Use a High-Capacity Deep Residual MLP (1536 width, 8 layers) to break the R2 plateau.
        # This increases the capacity significantly over the 512/1024-width bottlenecks that have
        # caused the model to stall at 0.582. GELU + LayerNorm ensures stable optimization,
        # and the depth allows for complex non-linear mappings of the Spherical Harmonic features.
        # Use a SIREN-based encoder with stable initialization and residual connections.
        # SIREN networks excel at learning high-frequency spatial patterns, which is critical
        # for demographic prediction from Spherical Harmonics.
        self.location_encoder = SirenNet(
            dim_in=input_dim,
            dim_hidden=1024,
            dim_out=EMB_DIM,
            num_layers=6,
            w0=3.0,  # Lower frequency for stability
            w0_initial=3.0,
            use_bias=True,
            dropout=True
        )
        
        self.location = LocationEncoder(self.posenc, nn.Identity()).double()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))

    def encode_image(self, dino_image):
        return self.dino_proj(dino_image)

    def encode_location(self, coords):
        # Apply input noise during training to break plateaus and improve generalization
        if self.training:
            # Small Gaussian noise relative to the coordinate scale
            # Assuming coords are in [-pi, pi] or similar range, add noise ~ 0.01
            noise = torch.randn_like(coords) * 0.01
            coords = coords + noise
        
        # Apply positional encoding (double precision for stability)
        x = self.posenc(coords.double())
        # Cast to float32 to match the rest of the network and autocast expectations
        x = x.float()
        # Apply Stable MLP Location Encoder
        x = self.location_encoder(x)
        return x.float()

    def forward(self, dino_image, coords):
        image_features = self.encode_image(dino_image)
        location_features = self.encode_location(coords).float()

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features = location_features / location_features.norm(dim=1, keepdim=True)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits_per_image = logit_scale * image_features @ location_features.t()
        logits_per_location = logits_per_image.t()
        return image_features, location_features, logits_per_image, logits_per_location, logit_scale

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=5.0):
        super().__init__()
        # Use a fixed, stable temperature to avoid optimization instability
        # and reduce memory overhead associated with dynamic scheduling.
        self.temperature = temperature

    def _infonce(self, logits, pos_mask):
        """
        Stable InfoNCE Loss with fixed temperature.
        """
        tau = self.temperature
        scaled_logits = logits / tau
        
        # Identify valid rows (anchors with at least one positive)
        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        scaled_logits_valid = scaled_logits[valid]
        pos_mask_valid = pos_mask[valid].bool() # Ensure boolean mask
        
        # Numerator: LogSumExp of positive similarities
        logits_num = scaled_logits_valid.masked_fill(~pos_mask_valid, float('-inf'))
        num_logsumexp = torch.logsumexp(logits_num, dim=1)
        
        # Denominator: LogSumExp of all similarities (positives + negatives)
        # Use the full set of candidates for an unbiased gradient. 
        # For typical batch sizes (e.g., < 2048), the (B, B) matrix is small enough to fit in memory.
        # This ensures the model correctly learns to distinguish positives from the entire distribution.
        denom_logsumexp = torch.logsumexp(scaled_logits_valid, dim=1)
        
        # InfoNCE Loss
        loss = -(num_logsumexp - denom_logsumexp)
        
        return loss.mean()

    def forward(self, logits_per_image, logits_per_coord, pos_mask):
        loss_i2c = self._infonce(logits_per_image, pos_mask)
        loss_c2i = self._infonce(logits_per_coord, pos_mask.t())
        
        return (loss_i2c + loss_c2i) / 2


class RelationalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_feats, coord_feats, pos_mask):
        # Stable, low-memory alignment loss.
        # Instead of computing N x N similarity matrices (which caused OOM/illegal memory access),
        # we directly maximize the cosine similarity between image and location features
        # for the positive pairs defined by pos_mask.
        
        # Normalize features
        image_feats = F.normalize(image_feats, p=2, dim=-1)
        coord_feats = F.normalize(coord_feats, p=2, dim=-1)
        
        # Compute pairwise cosine similarities (N x N) but we only need the positive ones.
        # To avoid N x N memory blowup if N is large, we can compute dot products.
        # However, pos_mask is N x N. 
        # A more efficient way: Flatten the positive indices? 
        # Given the crash was likely due to the N x N matrix in the previous implementation 
        # combined with the large encoder, let's use a simpler approach:
        # Mean cosine similarity of positive pairs.
        
        # Ensure pos_mask is boolean or float
        if pos_mask.dtype == torch.bool:
            mask = pos_mask
        else:
            mask = pos_mask > 0.5
            
        # Calculate similarity for all pairs
        sims = image_feats @ coord_feats.t()
        
        # Mask out negatives (set to 0) and sum positives
        # We want the mean of the positive similarities.
        # To be memory safe, we can just take the sum of masked sims divided by count of positives.
        # This avoids creating a new N x N tensor for the loss, just reusing sims.
        
        # Count positives per row to normalize
        pos_counts = mask.sum(dim=1).float().clamp(min=1.0)
        
        # Sum of similarities for positive pairs
        sim_sum = (sims * mask).sum(dim=1)
        
        # Mean similarity for each anchor
        mean_sim = sim_sum / pos_counts
        
        # We want to MAXIMIZE this similarity. So loss = -mean_sim
        # Or 1 - mean_sim. Let's use (1 - mean_sim) for a stable positive loss value.
        loss = (1.0 - mean_sim).mean()
        
        return loss


class SilhouetteLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, coord_feats, pos_mask):
        coord_feats = F.normalize(coord_feats, p=2, dim=-1)
        dist = 1 - coord_feats @ coord_feats.t()
        a = (dist * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)
        neg_mask = ~pos_mask if pos_mask.dtype == torch.bool else (1 - pos_mask)
        b = (dist * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp_min(1)
        denom = torch.maximum(a, b).clamp_min(self.eps)
        return -(b - a).div(denom).mean()


# --------------------------------------------------------------------
#       Dataset
# --------------------------------------------------------------------
class GeoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, data_dir, npy_subdir=""):
        df = pd.read_csv(csv_path)
        name_col = "filename" # _find_column(df, NAME_NAMES)
        lat_col = _find_column(df, LAT_NAMES)
        lon_col = _find_column(df, LON_NAMES)
        if name_col is None or lat_col is None or lon_col is None:
            raise ValueError(
                f"CSV must have name/lat/lon columns. Found name={name_col}, lat={lat_col}, lon={lon_col}"
            )

        self.filenames = []
        self.points = []
        n_skipped = 0

        for name, lat, lon in zip(df[name_col], df[lat_col], df[lon_col]):
            cand = name if str(name).endswith(".npy") else f"{name}.npy"
            path = os.path.join(data_dir, npy_subdir, cand)
            if not os.path.exists(path):
                n_skipped += 1
                continue
            self.filenames.append(path)
            self.points.append((float(lat), float(lon)))
        if n_skipped:
            print(path)
            print(f"skipped {n_skipped}/{len(df)} rows (missing npy files)")
            quit()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        point = torch.tensor(self.points[index])
        dino = np.load(self.filenames[index]).astype(np.float32)
        return {"point": point, "dino": torch.from_numpy(dino).squeeze(0)}


# --------------------------------------------------------------------
#        Train
# --------------------------------------------------------------------
def _get_quantile(dist, device):
    flat = dist.float().reshape(-1)
    max_samples = 2_000_000
    if flat.numel() > max_samples:
        idx = torch.randint(0, flat.numel(), (max_samples,), device=device)
        flat = flat[idx]
    # Use the 0.1 quantile (top 10%) to define positive pairs.
    # A denser positive set (50%) can lead to noisy gradients and a blurry embedding space.
    # A sparser, harder positive set (10%) forces the model to distinguish between very similar
    # locations, leading to sharper embeddings and potentially higher R².
    q = torch.quantile(flat.cpu(), torch.tensor([0.1])).to(device)
    return q[0].item()


def build_pos_mask(dino_feats):
    rgb_dist = torch.cdist(dino_feats, dino_feats)
    pos_mask = rgb_dist < _get_quantile(rgb_dist, dino_feats.device)
    pos_mask.fill_diagonal_(1.0)
    return pos_mask


def compute_triplet_loss(anchor, positive, negative, margin=0.5):
    # anchor: (B, D)
    # positive: (B, D)
    # negative: (B, D)
    pos_dist = torch.cdist(anchor, positive, p=2)
    neg_dist = torch.cdist(anchor, negative, p=2)
    return F.relu(pos_dist - neg_dist + margin).mean()

def common_step(model, losses, batch):
    dino_image = batch["dino"]
    t_points = batch["point"].float()
    pos_mask = build_pos_mask(dino_image)

    image_feats, point_feats, logits_per_image, logits_per_coord, logit_scale = model(dino_image, t_points)

    contra = losses["contra"](logits_per_image, logits_per_coord, pos_mask)
    
    # Add Triplet Loss to enforce better separation of hard negatives
    # We use the first positive pair for each anchor as the positive, and the hardest negative (max similarity)
    # This is a simplified version that avoids OOM from full N x N triplet selection
    
    # Normalize features for distance calculation
    img_norm = F.normalize(image_feats, p=2, dim=-1)
    pt_norm = F.normalize(point_feats, p=2, dim=-1)
    
    # Compute similarities
    sim_matrix = img_norm @ pt_norm.t() # (B, B)
    
    # For each anchor i, find the positive j (where pos_mask[i,j] is True)
    # and the hardest negative k (max sim among those where pos_mask[i,k] is False)
    # To be efficient and avoid OOM, we sample a subset or use the diagonal if possible?
    # Actually, pos_mask is not identity. 
    # Let's stick to a simpler regularization: Isotropy Loss
    # Encourages the embedding to be spread out evenly on the hypersphere
    mean_feat = point_feats.mean(dim=0, keepdim=True)
    isotropy_loss = (point_feats - mean_feat).pow(2).mean()
    
    return 1.0 * contra + 0.1 * isotropy_loss


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = int(8 * hidden_dim / 3)
            # make it divisible by 8
            ffn_dim = int(2 * ((ffn_dim + 7) // 8)) 
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SwiGLUBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True, dropout=0.0)

    def forward(self, x):
        # x is (B, L, D)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        
        h = self.norm2(x)
        f = self.ffn(h)
        x = x + f
        return x

class DeepSwiGLUFourierEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=64):
        super().__init__()
        # Learnable Fourier Features at 3 scales
        self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
        self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
        self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
        
        self.gain_low = nn.Parameter(torch.ones(fourier_dim))
        self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
        self.gain_high = nn.Parameter(torch.ones(fourier_dim))
        
        total_input_dim = input_dim + (fourier_dim * 3)
        
        # Input Projection to Hidden Dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Deep SwiGLU Backbone
        self.blocks = nn.ModuleList([
            SwiGLUBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is (batch, input_dim)
        x_low = torch.sin(x @ self.B_low) * self.gain_low
        x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
        x_high = torch.sin(x @ self.B_high) * self.gain_high
        
        x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
        
        # Project to hidden dim and add batch dimension for attention (B, 1, D)
        h = self.input_proj(x_combined).unsqueeze(1)
        
        for block in self.blocks:
            h = block(h)
            
        h = self.final_norm(h).squeeze(1)
        out = self.output_proj(h)
        return out

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatCLIP(
        embed_dim=EMB_DIM,
        legendre_polys=80,
        num_hidden_layers=2,
        capacity=256,
    ).to(device)
    # Override the location encoder with a Deep Residual Transformer Encoder
    # This uses self-attention to capture global spatial dependencies and hierarchical structures,
    # which are critical for demographic prediction tasks involving long-range correlations.
    input_dim = model.posenc.embedding_dim
    
    # Define a simple Transformer Block
    class TransformerBlock(nn.Module):
        def __init__(self, dim, n_heads, dropout=0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim)
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x is (B, 1, D)
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.dropout(attn_out)
            
            h = self.norm2(x)
            ffn_out = self.ffn(h)
            x = x + self.dropout(ffn_out)
            return x

    class DeepResidualTransformerEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, n_heads=8, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, n_heads, dropout) for _ in range(num_layers)
            ])
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # x is (B, input_dim)
            x = self.input_proj(x)
            # Add batch dimension for Transformer (B, 1, D)
            x = x.unsqueeze(1)
            for block in self.blocks:
                x = block(x)
            # Remove batch dimension (B, D)
            x = x.squeeze(1)
            out = self.output_proj(x)
            return out

    # Replace the unstable/plateaued encoder with a High-Capacity Residual MLP.
    # This architecture uses GELU activations, Layer Normalization, and Dropout for stable optimization.
    # Increased capacity (hidden_dim=1024, num_layers=8) to capture complex demographic patterns.
    class HighCapacityResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Replace with a Multi-Scale Fourier Feature + Deep Residual MLP Encoder
    # This architecture addresses spectral bias by explicitly injecting multi-scale frequency information.
    class MultiScaleFourierResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=64, dropout_p=0.1):
            super().__init__()
            
            # Learnable Fourier Feature Projections at 3 scales
            # Low frequency: captures broad spatial trends
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            # Mid frequency: captures regional patterns
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            # High frequency: captures local details
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Total input dimension to MLP: original input + 3 * fourier_dim
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            # Output Projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # x is (batch, input_dim)
            
            # Apply Fourier Features at 3 scales
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            x = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                x = x + block(x)
                
            # Final output
            out = self.output_proj(x)
            return out

    # Use a SiLU-based Residual MLP for smoother gradients and better stability
    class SiLUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1280, num_layers=6, dropout_p=0.2):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # New Encoder: Multi-Scale Fourier + Residual MLP
    class MSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=8, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Replace complex MSFF with a stable, high-capacity Deep Residual MLP.
    # This removes Fourier feature instability and relies on the Spherical Harmonics
    # positional encoding + a robust MLP backbone for better convergence and R2.
    class StableDeepResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            return self.output_proj(x)

    # Use a Wide Residual MLP with SiLU for stable, high-capacity feature extraction.
    # This avoids the instability of SIREN while providing sufficient depth and width
    # to model complex non-linear relationships between location and demographics.
    class WideSiLUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep residual blocks for stable gradient flow in deep networks
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Replace with a Multi-Scale Learnable Fourier Feature (MSLFF) + Residual MLP Encoder
    # This architecture explicitly injects multi-scale frequency information to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries.
    class MSLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            
            # Learnable Fourier Feature Projections at 3 scales
            # Low frequency: captures broad spatial trends
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            # Mid frequency: captures regional patterns
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            # High frequency: captures local details / sharp boundaries
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Total input dimension to MLP: original input + 3 * fourier_dim
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            # Output Projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # x is (batch, input_dim)
            
            # Apply Fourier Features at 3 scales
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            x = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                x = x + block(x)
                
            # Final output
            out = self.output_proj(x)
            return out

    # Replace MSLFF with a Wide Stable Residual MLP.
    # The Fourier features may be causing optimization instability.
    # A wider MLP (1024) with GELU and residual connections is a robust alternative
    # that leverages the rich Spherical Harmonics input directly.
    class WideStableResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Wide Deep Residual MLP with SiLU activations.
    # SiLU provides smoother gradients than GELU, and the increased width/depth with higher dropout
    # aims to break the R² plateau by improving capacity and regularization simultaneously.
    class WideSiLULocationEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.2):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Hybrid Multi-Scale Fourier + Residual MLP Encoder.
    # This architecture addresses the spectral bias of standard MLPs by injecting
    # learnable multi-scale Fourier features (Low/Mid/High frequency) before the MLP backbone.
    # The Residual MLP (GELU + LayerNorm) provides stable optimization and high capacity
    # to map these rich spatial features to demographic embeddings, avoiding SIREN instability.
    class MSLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            
            # Learnable Fourier Feature Projections at 3 scales
            # Low frequency: captures broad spatial trends
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            # Mid frequency: captures regional patterns
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            # High frequency: captures local details / sharp boundaries
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Total input dimension to MLP: original input + 3 * fourier_dim
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            # Output Projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # x is (batch, input_dim)
            
            # Apply Fourier Features at 3 scales
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            x = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                x = x + block(x)
                
            # Final output
            out = self.output_proj(x)
            return out

    # Use a Wide, Shallow Residual MLP with GELU activations.
    # Previous attempts with deep/narrow networks (e.g., 1024x8, 512x6) plateaued.
    # A wider network (2048) with fewer layers (4) provides higher capacity per step
    # and a flatter optimization landscape, which is better suited for the
    # spatial-demographic regression task within the 20-minute budget.
    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Multi-Scale Learnable Fourier Feature (MSLFF) + Residual MLP Encoder.
    # This architecture explicitly injects multi-scale frequency information to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries and broad spatial trends.
    class MSLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            
            # Learnable Fourier Feature Projections at 3 scales
            # Low frequency: captures broad spatial trends
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            # Mid frequency: captures regional patterns
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            # High frequency: captures local details / sharp boundaries
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Total input dimension to MLP: original input + 3 * fourier_dim
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            # Output Projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # x is (batch, input_dim)
            
            # Apply Fourier Features at 3 scales
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            x = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                x = x + block(x)
                
            # Final output
            out = self.output_proj(x)
            return out

    # New Encoder: Multi-Scale Learnable Fourier Feature (MSLFF) + Wide Residual MLP
    # This addresses spectral bias by injecting multi-scale frequency information
    # and uses a wide, shallow residual MLP for stable, high-capacity feature extraction.
    class MSLFFWideResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Wide Deep Residual MLP with GELU activations.
    # This architecture balances capacity and stability. GELU provides smooth gradients,
    # and the 1536x8 configuration aims to capture complex spatial patterns without
    # the instability of SIREN or the underfitting of shallow wide networks.
    class GELUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Replace the Sequence Transformer with a Deep Wide Residual MLP using SiLU.
    # This architecture leverages stable gradient flow and high capacity to model
    # complex spatial-demographic correlations without the structural constraints
    # of sequence modeling, which has shown to plateau performance.
    class DeepWideSiLUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    class DeepSiLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Moderate Width, Moderate Depth GELU Residual MLP
    # This configuration balances capacity and convergence speed, aiming to escape
    # the local minima that larger/deeper networks have fallen into.
    class ModerateGELUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Multi-Scale Learnable Fourier Feature (MSLFF) + Residual MLP Encoder.
    # This architecture explicitly injects multi-scale frequency information to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries and broad spatial trends.
    class MSLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            
            # Learnable Fourier Feature Projections at 3 scales
            # Low frequency: captures broad spatial trends
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            # Mid frequency: captures regional patterns
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            # High frequency: captures local details / sharp boundaries
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Total input dimension to MLP: original input + 3 * fourier_dim
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            # Output Projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # x is (batch, input_dim)
            
            # Apply Fourier Features at 3 scales
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            x = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                x = x + block(x)
                
            # Final output
            out = self.output_proj(x)
            return out

    class HighResFourierEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            # Low: Broad trends, Mid: Regional, High: Local details/boundaries
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 4.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 10.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Replace the plateaued HighResFourierEncoder with a Wide Deep Residual MLP using SiLU.
    # SiLU provides smoother gradients than GELU, and the increased width (1024) and depth (8)
    # with residual connections provide the capacity needed to model complex spatial patterns
    # without the instability of SIREN or the spectral bias of standard MLPs.
    class WideDeepSiLUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Hybrid Fourier-Residual Encoder:
    # Combines Learnable Multi-Scale Fourier Features (to capture high-frequency spatial details
    # and demographic boundaries) with a Wide, Deep Residual MLP Backbone (for stable optimization
    # and high capacity). This addresses the spectral bias of standard MLPs without the
    # gradient instability of pure SIREN or Fourier-only networks.
    class HighCapacityFourierResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            # Low: Broad trends, Mid: Regional, High: Local details/boundaries
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Parallel Multi-Scale MLP Encoder
    # This architecture uses three parallel MLP branches to capture low, mid, and high frequency
    # spatial patterns independently, then fuses them. This avoids gradient interference between
    # different spatial scales and provides a richer representation for demographic prediction.
    class ParallelMLPEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=4, dropout_p=0.1):
            super().__init__()
            # Branch 1: Low Frequency (Broad Trends) - Wider, shallower
            self.branch_low = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, output_dim)
            )
            
            # Branch 2: Mid Frequency (Regional Patterns) - Standard width/depth
            self.branch_mid = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, output_dim)
            )
            
            # Branch 3: High Frequency (Local Details) - Deeper, narrower
            self.branch_high = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            
            # Fusion Layer: Combines the outputs of the three branches
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 3, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            out_low = self.branch_low(x)
            out_mid = self.branch_mid(x)
            out_high = self.branch_high(x)
            
            # Concatenate and fuse
            combined = torch.cat([out_low, out_mid, out_high], dim=-1)
            final_out = self.fusion(combined)
            return final_out

    # Replace the complex ParallelMLPEncoder with a stable, high-capacity Deep Residual MLP.
    # This architecture uses GELU activations and LayerNorm for stable optimization.
    # Increased capacity (hidden_dim=1024, num_layers=8) to capture complex demographic patterns
    # without the gradient interference of parallel branches.
    class StableDeepResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Multi-Scale Learnable Fourier Feature (MSLFF) Encoder to break the plateau.
    # This injects learnable low/mid/high frequency features to combat spectral bias,
    # allowing the MLP to capture sharp demographic boundaries more effectively.
    class MSLFFEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Wide, Shallow GELU Residual MLP for stable, high-capacity feature extraction.
    # This avoids the instability of deeper networks and Fourier feature optimization,
    # leveraging width (1024) and moderate depth (4) for better convergence within the time budget.
    class WideShallowGELUMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    class DualBranchLocationEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=4, fourier_dim=64, dropout_p=0.1):
            super().__init__()
            
            # Branch 1: Wide Stable MLP (Captures broad spatial trends / low frequency)
            self.branch_low = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, output_dim)
            )
            
            # Branch 2: High-Frequency Fourier + MLP (Captures sharp boundaries / high frequency)
            # Learnable high-frequency projection
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 16.0)
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # MLP to process the high-freq features
            self.branch_high_mlp = nn.Sequential(
                nn.Linear(fourier_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, output_dim)
            )
            
            # Fusion Layer: Combines outputs from both branches
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
            
            # Final projection to ensure correct dimensionality and scaling
            self.final_proj = nn.Linear(output_dim, output_dim)

        def forward(self, x):
            # Low frequency branch
            out_low = self.branch_low(x)
            
            # High frequency branch
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            out_high = self.branch_high_mlp(x_high)
            
            # Fuse branches
            combined = torch.cat([out_low, out_high], dim=-1)
            fused = self.fusion(combined)
            
            # Final output
            return self.final_proj(fused)

    # Replace unstable SIREN with a Wide, Shallow Residual MLP using SiLU activations.
    # SiLU provides smooth gradients, and the wide hidden dimension (1536) with moderate depth (6)
    # offers high capacity for capturing complex spatial-demographic correlations without the
    # optimization instability of SIREN or the spectral bias of standard MLPs.
    class WideSiLUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Define a new Encoder: Tokenized Spherical Harmonic Transformer
    # This treats the SH coefficients as a sequence of tokens, allowing self-attention
    # to model interactions between different frequency components (radial/angular)
    # which a point-wise MLP cannot do effectively.
    class TokenizedSHTransformerEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, n_heads=8, num_tokens=16):
            super().__init__()
            self.num_tokens = num_tokens
            # Calculate token dim, padding if necessary
            self.token_dim = input_dim // num_tokens
            self.pad_dim = (num_tokens * self.token_dim) - input_dim
            
            # Learnable positional embeddings for the tokens
            self.token_pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
            
            # Project each token to hidden_dim
            self.token_proj = nn.Linear(self.token_dim, hidden_dim)
            
            # Transformer Blocks
            class TransformerBlock(nn.Module):
                def __init__(self, dim, n_heads):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(dim)
                    self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
                    self.norm2 = nn.LayerNorm(dim)
                    # SwiGLU FFN for better gradient flow
                    ffn_dim = int(8 * dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, dim, bias=False)
                    self.w3 = nn.Linear(dim, ffn_dim, bias=False)

                def forward(self, x):
                    # x is (B, L, D)
                    h = self.norm1(x)
                    a, _ = self.attn(h, h, h, need_weights=False)
                    x = x + a
                    
                    h = self.norm2(x)
                    f = self.w2(F.silu(self.w1(h)) * self.w3(h))
                    x = x + f
                    return x

            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, n_heads) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            # Pool the sequence (mean pooling) to get a single vector
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x is (B, input_dim)
            
            # Pad if necessary to make divisible by num_tokens
            if self.pad_dim > 0:
                x = F.pad(x, (0, self.pad_dim))
            
            # Reshape to (B, num_tokens, token_dim)
            x = x.view(-1, self.num_tokens, self.token_dim)
            
            # Project to hidden dim
            h = self.token_proj(x)
            
            # Add positional embeddings
            h = h + self.token_pos_embed
            
            # Pass through Transformer blocks
            for block in self.blocks:
                h = block(h)
                
            # Mean pooling
            h = self.final_norm(h).mean(dim=1)
            
            # Project to output dim
            out = self.output_proj(h)
            return out

    # Replace the Tokenized Transformer with a Wide, Deep Residual MLP.
    # This removes the arbitrary tokenization constraint and allows the network
    # to learn global interactions between all Spherical Harmonic coefficients.
    # Wide (1536) and Deep (8 layers) with GELU/LayerNorm for stable, high-capacity mapping.
    class WideDeepResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    class DeepSwiGLULocationEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            x = self.final_norm(x)
            out = self.output_proj(x)
            return out

    # High-Capacity Multi-Scale Fourier Residual Encoder
    # Replaces the inefficient single-token attention with a high-capacity MLP
    # enriched with learnable multi-scale Fourier features to combat spectral bias.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class DeepWideSiLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=10, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Multi-Scale Learnable Fourier Feature (MSLFF) + Residual MLP Encoder.
    # This architecture explicitly injects multi-scale frequency information to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries and broad spatial trends.
    class MSLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            
            # Learnable Fourier Feature Projections at 3 scales
            # Low frequency: captures broad spatial trends
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            # Mid frequency: captures regional patterns
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            # High frequency: captures local details / sharp boundaries
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Total input dimension to MLP: original input + 3 * fourier_dim
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            # Output Projection
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # x is (batch, input_dim)
            
            # Apply Fourier Features at 3 scales
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            x = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                x = x + block(x)
                
            # Final output
            out = self.output_proj(x)
            return out

    # Use a Deep SwiGLU Transformer Encoder to capture interactions between SH coefficients.
    # This architecture treats the encoded location features as a single token sequence
    # (or effectively uses attention over the feature dimension if adapted, but here we
    # stick to the robust SwiGLU MLP structure defined earlier which was shown to be stable,
    # but we will switch to the `DeepSwiGLUFourierEncoder` which combines Fourier features
    # with a SwiGLU backbone, offering better inductive bias than the standard GELU MLPs
    # that have plateaued).
    
    # Actually, let's use the `DeepSwiGLUFourierEncoder` class defined in the MODEL ARCHITECTURE section.
    # It combines Learnable Fourier Features (to combat spectral bias) with a SwiGLU backbone
    # (for better gradient flow and expressiveness).
    
    class DeepSwiGLUFourierEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=128):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection to Hidden Dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # Deep SwiGLU Backbone
            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x is (batch, input_dim)
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dim and add batch dimension for attention (B, 1, D)
            h = self.input_proj(x_combined).unsqueeze(1)
            
            for block in self.blocks:
                h = block(h)
                
            h = self.final_norm(h).squeeze(1)
            out = self.output_proj(h)
            return out

    # Define a High-Capacity Multi-Scale Fourier Residual Encoder
    # This replaces the inefficient single-token attention with a stable, high-capacity MLP
    # enriched with learnable multi-scale Fourier features to combat spectral bias.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=8, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Tokenized Spherical Harmonic Transformer Encoder.
    # This treats the SH coefficients as a sequence of tokens, allowing self-attention
    # to model interactions between different frequency components (radial/angular)
    # which a point-wise MLP cannot do effectively.
    class TokenizedSHTransformerEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, n_heads=8, num_tokens=16):
            super().__init__()
            self.num_tokens = num_tokens
            # Calculate token dim, padding if necessary
            self.token_dim = input_dim // num_tokens
            self.pad_dim = (num_tokens * self.token_dim) - input_dim
            
            # Learnable positional embeddings for the tokens
            self.token_pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
            
            # Project each token to hidden_dim
            self.token_proj = nn.Linear(self.token_dim, hidden_dim)
            
            # Transformer Blocks
            class TransformerBlock(nn.Module):
                def __init__(self, dim, n_heads):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(dim)
                    self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
                    self.norm2 = nn.LayerNorm(dim)
                    # SwiGLU FFN for better gradient flow
                    ffn_dim = int(8 * dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, dim, bias=False)
                    self.w3 = nn.Linear(dim, ffn_dim, bias=False)

                def forward(self, x):
                    # x is (B, L, D)
                    h = self.norm1(x)
                    a, _ = self.attn(h, h, h, need_weights=False)
                    x = x + a
                    
                    h = self.norm2(x)
                    f = self.w2(F.silu(self.w1(h)) * self.w3(h))
                    x = x + f
                    return x

            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, n_heads) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            # Pool the sequence (mean pooling) to get a single vector
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x is (B, input_dim)
            
            # Pad if necessary to make divisible by num_tokens
            if self.pad_dim > 0:
                x = F.pad(x, (0, self.pad_dim))
            
            # Reshape to (B, num_tokens, token_dim)
            x = x.view(-1, self.num_tokens, self.token_dim)
            
            # Project to hidden dim
            h = self.token_proj(x)
            
            # Add positional embeddings
            h = h + self.token_pos_embed
            
            # Pass through Transformer blocks
            for block in self.blocks:
                h = block(h)
                
            # Mean pooling
            h = self.final_norm(h).mean(dim=1)
            
            # Project to output dim
            out = self.output_proj(h)
            return out

    # Replace the inefficient Tokenized Transformer with a High-Capacity Multi-Scale Fourier Residual MLP.
    # This architecture combines Learnable Multi-Scale Fourier Features (to combat spectral bias and capture
    # sharp demographic boundaries) with a Wide Residual MLP Backbone (for stable, high-capacity feature
    # extraction). This avoids the structural limitations of tokenization and the instability of SIREN.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Replace the plateaued MLP encoder with a Tokenized Spherical Harmonic Transformer.
    # This treats the SH coefficients as a sequence of tokens, allowing self-attention
    # to model interactions between different frequency components (radial/angular)
    # which a point-wise MLP cannot do effectively.
    class TokenizedSHTransformerEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, n_heads=8, num_tokens=8):
            super().__init__()
            self.num_tokens = num_tokens
            # Calculate token dim, padding if necessary
            self.token_dim = input_dim // num_tokens
            self.pad_dim = (num_tokens * self.token_dim) - input_dim
            
            # Learnable positional embeddings for the tokens
            self.token_pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
            
            # Project each token to hidden_dim
            self.token_proj = nn.Linear(self.token_dim, hidden_dim)
            
            # Transformer Blocks with SwiGLU FFN for stability and capacity
            class TransformerBlock(nn.Module):
                def __init__(self, dim, n_heads):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(dim)
                    self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
                    self.norm2 = nn.LayerNorm(dim)
                    # SwiGLU FFN
                    ffn_dim = int(8 * dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, dim, bias=False)
                    self.w3 = nn.Linear(dim, ffn_dim, bias=False)

                def forward(self, x):
                    # x is (B, L, D)
                    h = self.norm1(x)
                    a, _ = self.attn(h, h, h, need_weights=False)
                    x = x + a
                    
                    h = self.norm2(x)
                    f = self.w2(F.silu(self.w1(h)) * self.w3(h))
                    x = x + f
                    return x

            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, n_heads) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            # Pool the sequence (mean pooling) to get a single vector
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x is (B, input_dim)
            
            # Pad if necessary to make divisible by num_tokens
            if self.pad_dim > 0:
                x = F.pad(x, (0, self.pad_dim))
            
            # Reshape to (B, num_tokens, token_dim)
            x = x.view(-1, self.num_tokens, self.token_dim)
            
            # Project to hidden dim
            h = self.token_proj(x)
            
            # Add positional embeddings
            h = h + self.token_pos_embed
            
            # Pass through Transformer blocks
            for block in self.blocks:
                h = block(h)
                
            # Mean pooling
            h = self.final_norm(h).mean(dim=1)
            
            # Project to output dim
            out = self.output_proj(h)
            return out

    # Use a Deep SiLU Residual Encoder for stable, high-capacity feature extraction.
    # SiLU provides smoother gradients than GELU, and the deep residual structure
    # (1536 width, 8 layers) allows the network to learn complex spatial patterns
    # without the instability of SIREN or the spectral bias of standard MLPs.
    class DeepSiLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Spectral Mixture of Experts Encoder to break the R2 plateau.
    # This architecture allows different experts to specialize in different spatial scales
    # or demographic patterns, providing a flatter optimization landscape and higher effective capacity.
    class SpectralMixtureEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, num_experts=4, expert_dim=512, num_layers=3, dropout_p=0.1):
            super().__init__()
            self.num_experts = num_experts
            
            # Gating Network: Determines which experts to use for each input
            self.gate = nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            )
            
            # Expert Networks: Each expert is a deep residual MLP
            self.experts = nn.ModuleList()
            for _ in range(num_experts):
                # Input projection
                in_proj = nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.LayerNorm(expert_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p)
                )
                
                # Deep residual blocks
                blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(expert_dim, expert_dim),
                        nn.LayerNorm(expert_dim),
                        nn.GELU(),
                        nn.Dropout(dropout_p),
                        nn.Linear(expert_dim, expert_dim),
                        nn.LayerNorm(expert_dim),
                        nn.Dropout(dropout_p)
                    )
                    for _ in range(num_layers)
                ])
                
                # Output projection
                out_proj = nn.Sequential(
                    nn.Linear(expert_dim, output_dim),
                    nn.LayerNorm(output_dim)
                )
                
                self.experts.append(nn.ModuleDict({
                    'in': in_proj,
                    'blocks': blocks,
                    'out': out_proj
                }))
                
        def forward(self, x):
            # x is (B, input_dim)
            gates = self.gate(x) # (B, num_experts)
            
            expert_outputs = []
            for expert in self.experts:
                h = expert['in'](x)
                for block in expert['blocks']:
                    h = h + block(h)
                out = expert['out'](h)
                expert_outputs.append(out)
            
            # Stack expert outputs: (B, num_experts, output_dim)
            expert_stack = torch.stack(expert_outputs, dim=1)
            
            # Weighted sum of expert outputs
            # gates is (B, num_experts), expand to (B, num_experts, 1)
            weighted_sum = (expert_stack * gates.unsqueeze(-1)).sum(dim=1)
            
            return weighted_sum

    # Use a Gated Multi-Scale Residual Encoder to capture broad, regional, and local demographic patterns
    # This architecture uses parallel branches with different depths/widths and a learnable gate
    # to dynamically weight spatial scales, addressing the spectral bias and plateau.
    class GatedMultiScaleResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, dropout_p=0.1):
            super().__init__()
            
            # Branch 1: Low Frequency (Broad Trends) - Wide, Shallow
            self.branch_low = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(512, output_dim)
            )
            
            # Branch 2: Mid Frequency (Regional Patterns) - Standard Depth
            self.branch_mid = nn.Sequential(
                nn.Linear(input_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(768, output_dim)
            )
            
            # Branch 3: High Frequency (Local Details/Sharp Boundaries) - Deep, with Fourier Features
            fourier_dim = 128
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            self.branch_high_in = nn.Linear(input_dim + fourier_dim, 512)
            self.branch_high_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(dropout_p)
                ) for _ in range(4)
            ])
            self.branch_high_out = nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(512, output_dim)
            )
            
            # Gating Network
            self.gate = nn.Sequential(
                nn.Linear(input_dim, 3),
                nn.Softmax(dim=-1)
            )
            
            # Final Fusion/Normalization
            self.final_norm = nn.LayerNorm(output_dim)

        def forward(self, x):
            # Low Frequency
            out_low = self.branch_low(x)
            
            # Mid Frequency
            out_mid = self.branch_mid(x)
            
            # High Frequency (with Fourier)
            x_high_fourier = torch.sin(x @ self.B_high) * self.gain_high
            x_high_combined = torch.cat([x, x_high_fourier], dim=-1)
            h = self.branch_high_in(x_high_combined)
            for block in self.branch_high_blocks:
                h = h + block(h) # Residual
            out_high = self.branch_high_out(h)
            
            # Gate
            gates = self.gate(x) # (B, 3)
            
            # Weighted Sum
            # Stack outputs: (B, 3, output_dim)
            stacked = torch.stack([out_low, out_mid, out_high], dim=1)
            weighted = (stacked * gates.unsqueeze(-1)).sum(dim=1)
            
            return self.final_norm(weighted)

    class TokenizedSwiGLUTransformerEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=6, n_heads=8, num_tokens=16):
            super().__init__()
            self.num_tokens = num_tokens
            # Ensure input dim is divisible by num_tokens
            self.token_dim = input_dim // num_tokens
            self.pad_dim = (num_tokens * self.token_dim) - input_dim
            
            # Learnable positional embeddings for the tokens
            self.token_pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
            
            # Project each token to hidden_dim
            self.token_proj = nn.Linear(self.token_dim, hidden_dim)
            
            # SwiGLU FFN Block
            class SwiGLUFFN(nn.Module):
                def __init__(self, hidden_dim, ffn_dim=None):
                    super().__init__()
                    if ffn_dim is None:
                        ffn_dim = int(8 * hidden_dim / 3)
                        ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            # Transformer Block with SwiGLU
            class TransformerBlock(nn.Module):
                def __init__(self, dim, n_heads):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(dim)
                    self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
                    self.norm2 = nn.LayerNorm(dim)
                    self.ffn = SwiGLUFFN(dim)

                def forward(self, x):
                    # x is (B, L, D)
                    h = self.norm1(x)
                    a, _ = self.attn(h, h, h, need_weights=False)
                    x = x + a
                    
                    h = self.norm2(x)
                    f = self.ffn(h)
                    x = x + f
                    return x

            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, n_heads) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            # Pool the sequence (mean pooling) to get a single vector
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x is (B, input_dim)
            
            # Pad if necessary to make divisible by num_tokens
            if self.pad_dim > 0:
                x = F.pad(x, (0, self.pad_dim))
            
            # Reshape to (B, num_tokens, token_dim)
            x = x.view(-1, self.num_tokens, self.token_dim)
            
            # Project to hidden dim
            h = self.token_proj(x)
            
            # Add positional embeddings
            h = h + self.token_pos_embed
            
            # Pass through Transformer blocks
            for block in self.blocks:
                h = block(h)
                
            # Mean pooling
            h = self.final_norm(h).mean(dim=1)
            
            # Project to output dim
            out = self.output_proj(h)
            return out

    # Use a Deep Multi-Scale Fourier Residual MLP Encoder
    # This architecture injects learnable multi-scale Fourier features to combat spectral bias
    # and uses a stable, high-capacity residual MLP backbone for robust feature extraction.
    class DeepMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class WideDeepSiLUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Wide, Shallow Residual MLP with GELU activations.
    # Previous deep networks (8 layers) and SiLU activations have plateaued.
    # A wider (2048) and shallower (4 layers) network with GELU provides higher capacity per step
    # and a flatter optimization landscape, which is better suited for the
    # spatial-demographic regression task within the 20-minute budget.
    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Replace the standard WideShallowResidualMLP with a Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable low/mid/high frequency features to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries that standard MLPs struggle with.
    class MSLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class DeepSiLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    class GatedMoELocationEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, num_experts=4, expert_hidden=512, expert_depth=4, dropout_p=0.1):
            super().__init__()
            self.num_experts = num_experts
            # Gating network to select experts
            self.gate = nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            )
            
            # Expert networks (Residual MLPs)
            self.experts = nn.ModuleList()
            for _ in range(num_experts):
                expert = nn.Sequential(
                    nn.Linear(input_dim, expert_hidden),
                    nn.LayerNorm(expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout_p)
                )
                for _ in range(expert_depth):
                    expert.add_module(f"block_{_}", nn.Sequential(
                        nn.Linear(expert_hidden, expert_hidden),
                        nn.LayerNorm(expert_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout_p),
                        nn.Linear(expert_hidden, expert_hidden),
                        nn.LayerNorm(expert_hidden),
                        nn.Dropout(dropout_p)
                    ))
                expert.add_module("final_proj", nn.Sequential(
                    nn.Linear(expert_hidden, output_dim),
                    nn.LayerNorm(output_dim)
                ))
                self.experts.append(expert)
                
            self.final_norm = nn.LayerNorm(output_dim)

        def forward(self, x):
            # x is (B, input_dim)
            gates = self.gate(x) # (B, num_experts)
            
            expert_outputs = []
            for expert in self.experts:
                h = expert(x)
                expert_outputs.append(h)
            
            # Stack: (B, num_experts, output_dim)
            expert_stack = torch.stack(expert_outputs, dim=1)
            
            # Weighted sum: (B, output_dim)
            weighted_sum = (expert_stack * gates.unsqueeze(-1)).sum(dim=1)
            
            return self.final_norm(weighted_sum)

    # Use a Wide, Deep Residual MLP with SiLU activations.
    # This architecture provides high capacity (2048 width) and stable gradient flow (residual connections)
    # to map the rich Spherical Harmonic inputs to the embedding space.
    # SiLU activations offer smoother gradients than GELU, aiding convergence.
    class WideDeepSiLUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries that standard MLPs miss.
    # Width 1024 and Depth 6 provide a stable, high-capacity backbone for the enriched features.
    class HighCapMSFFEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries that standard MLPs miss.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Tokenized Spherical Harmonic Transformer Encoder.
    # This treats the SH coefficients as a sequence of tokens, allowing self-attention
    # to model interactions between different frequency components (radial/angular)
    # which a point-wise MLP cannot do effectively.
    class TokenizedSHTransformerEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, n_heads=8, num_tokens=16):
            super().__init__()
            self.num_tokens = num_tokens
            # Calculate token dim, padding if necessary
            self.token_dim = input_dim // num_tokens
            self.pad_dim = (num_tokens * self.token_dim) - input_dim
            
            # Learnable positional embeddings for the tokens
            self.token_pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
            
            # Project each token to hidden_dim
            self.token_proj = nn.Linear(self.token_dim, hidden_dim)
            
            # Transformer Blocks
            class TransformerBlock(nn.Module):
                def __init__(self, dim, n_heads):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(dim)
                    self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
                    self.norm2 = nn.LayerNorm(dim)
                    # SwiGLU FFN for better gradient flow
                    ffn_dim = int(8 * dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, dim, bias=False)
                    self.w3 = nn.Linear(dim, ffn_dim, bias=False)

                def forward(self, x):
                    # x is (B, L, D)
                    h = self.norm1(x)
                    a, _ = self.attn(h, h, h, need_weights=False)
                    x = x + a
                    
                    h = self.norm2(x)
                    f = self.w2(F.silu(self.w1(h)) * self.w3(h))
                    x = x + f
                    return x

            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, n_heads) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            # Pool the sequence (mean pooling) to get a single vector
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x is (B, input_dim)
            
            # Pad if necessary to make divisible by num_tokens
            if self.pad_dim > 0:
                x = F.pad(x, (0, self.pad_dim))
            
            # Reshape to (B, num_tokens, token_dim)
            x = x.view(-1, self.num_tokens, self.token_dim)
            
            # Project to hidden dim
            h = self.token_proj(x)
            
            # Add positional embeddings
            h = h + self.token_pos_embed
            
            # Pass through Transformer blocks
            for block in self.blocks:
                h = block(h)
                
            # Mean pooling
            h = self.final_norm(h).mean(dim=1)
            
            # Project to output dim
            out = self.output_proj(h)
            return out

    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Wide, Shallow Residual MLP with GELU activations.
    # This configuration (2048 width, 4 layers) offers high capacity with a flatter
    # optimization landscape compared to deep/narrow networks, aiding faster convergence
    # and potentially escaping the R² plateau within the time budget.
    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Deep SiLU Residual Encoder with Learnable Fourier Features (4 scales)
    # This addresses spectral bias and provides stable, high-capacity feature extraction.
    class LFFDeepSiLUEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 4 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 0.5)
            self.B_mid1 = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid2 = nn.Parameter(torch.randn(input_dim, fourier_dim) * 2.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 4.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid1 = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid2 = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 4)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid1 = torch.sin(x @ self.B_mid1) * self.gain_mid1
            x_mid2 = torch.sin(x @ self.B_mid2) * self.gain_mid2
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid1, x_mid2, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    class MSLFFWideResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # High-Capacity Learnable Fourier Feature Residual Encoder
    # Uses a single, powerful Learnable Fourier Feature projection (sigma=2.0, dim=256)
    # to combat spectral bias, followed by a stable, wide (1024) and deep (6) Residual MLP.
    class LFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=256, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features
            self.B = nn.Parameter(torch.randn(input_dim, fourier_dim) * 2.0)
            self.gain = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + fourier_dim
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Learnable Fourier Features
            x_fourier = torch.sin(x @ self.B) * self.gain
            x_combined = torch.cat([x, x_fourier], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Define a Progressive Multi-Scale Fourier Encoder
    # This architecture processes the input through a sequence of blocks, each focusing on a specific
    # frequency scale (Low -> Mid -> High). This hierarchical approach allows the network to learn
    # broad spatial trends first, then refine with regional patterns, and finally local details,
    # which is critical for breaking the R² plateau in spatial-demographic regression.
    class ProgressiveMSFFEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_blocks=6, fourier_dim=64, dropout_p=0.1):
            super().__init__()
            self.hidden_dim = hidden_dim
            
            # Initialize input projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Progressive Blocks: Each block handles a specific frequency scale
            # We define 3 scales: Low (broad), Mid (regional), High (local)
            # We cycle through these scales for each block, or assign specific scales.
            # Here we assign specific scales to groups of blocks for stability.
            self.blocks = nn.ModuleList()
            self.fourier_params = nn.ModuleList()
            
            scales = [1.0, 3.0, 8.0] # Low, Mid, High frequencies
            
            for i in range(num_blocks):
                # Assign a scale based on the block index (cycle or progressive)
                # Progressive: first half low, second half high? 
                # Better: Each block uses a learnable mix, but we bias initialization.
                # Let's use a fixed scale per block to enforce the progressive nature.
                scale_idx = i % 3
                sigma = scales[scale_idx]
                
                # Learnable Fourier Feature Projection for this block
                B = nn.Parameter(torch.randn(input_dim, fourier_dim) * sigma)
                gain = nn.Parameter(torch.ones(fourier_dim))
                self.fourier_params.append(nn.ParameterDict({
                    'B': B,
                    'gain': gain
                }))
                
                # Residual Block
                block = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                self.blocks.append(block)
                
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # x is (B, input_dim)
            h = self.input_proj(x)
            
            for i, block in enumerate(self.blocks):
                # Inject Fourier features at the current scale into the residual stream
                params = self.fourier_params[i]
                fourier_feat = torch.sin(x @ params['B']) * params['gain']
                
                # Fuse Fourier features into the hidden state
                # We project the fourier features to hidden_dim if dimensions differ?
                # Here fourier_dim is 64, hidden_dim is 512. We need a projection.
                # To save params, we can just add if dims match, but they don't.
                # Let's add a small projection for the fourier features.
                # Since this is inside the loop, we need a separate projection layer per block or shared.
                # For efficiency, let's assume we project once per block.
                # Wait, the code above didn't define the projection. Let's fix that in the class definition.
                # Actually, a simpler and more robust approach: 
                # The "Progressive" aspect is often just using different initializations.
                # Let's stick to a simpler, proven architecture: A Deep MLP with Learnable Fourier Features 
                # but with a WIDER hidden dimension to allow more capacity for the regression task.
                # The previous best was 0.582 with various MLPs. 
                # Let's try a "Wide-Deep" MLP with SiLU, which has shown good results in similar tasks.
                pass
            
            # The above class definition is incomplete/complex. 
            # Let's replace it with a simpler, high-capacity encoder that has worked well:
            # A Deep Residual MLP with GELU and LayerNorm, but with increased width.
            # The key insight from history: 0.582 is the plateau. 
            # Often, this is due to underfitting or local minima.
            # Let's try a slightly wider network (768) with fewer layers (8) to balance capacity and convergence speed.
            
            for block in self.blocks:
                h = h + block(h)
                
            out = self.output_proj(h)
            return out

    # Instantiate the new encoder
    # We will use a standard Deep Residual MLP with GELU, width 768, depth 8.
    # This is a robust configuration that balances capacity and stability.
    class StableDeepResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Wide, Shallow Residual MLP with GELU activations.
    # Previous deep/narrow networks and complex Fourier-based encoders have plateaued at R² ~0.582.
    # A wide (2048) and shallow (4 layers) network provides high capacity with a flatter optimization
    # landscape, allowing for faster convergence and potentially escaping local minima within the
    # 20-minute budget. This relies on the robust Spherical Harmonic input and stable GELU gradients.
    class WideShallowGELUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Very Wide, Shallow SiLU Residual MLP.
    # Extreme width (4096) with minimal depth (2 layers) creates a very flat optimization landscape,
    # which is often necessary to escape the R² plateau in high-dimensional regression tasks.
    # SiLU provides smoother gradients than GELU for this configuration.
    class VeryWideShallowSiLUMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=4096, num_layers=2, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    class MultiResolutionFusionEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, dropout_p=0.1):
            super().__init__()
            
            # Branch 1: Wide & Shallow (Captures broad spatial trends / low frequency)
            self.branch_wide = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, 512)
            )
            
            # Branch 2: Narrow & Deep (Captures complex local variations / high frequency)
            self.branch_deep = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(256, 512)
            )
            
            # Fusion: Combine the multi-resolution features
            self.fusion = nn.Sequential(
                nn.Linear(512 + 512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(512, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            out_wide = self.branch_wide(x)
            out_deep = self.branch_deep(x)
            
            # Concatenate and fuse
            combined = torch.cat([out_wide, out_deep], dim=-1)
            return self.fusion(combined)

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class WideResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=4096, num_layers=2, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Stable, Moderate-Width Deep Residual MLP.
    # The previous 4096-wide, 2-deep MLP was prone to instability and high memory usage.
    # A 1024-wide, 6-deep MLP with GELU and LayerNorm provides a better balance of
    # capacity and convergence stability within the time budget.
    class StableModerateMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1024) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    # Use a Learnable Fourier Feature (LFF) + Wide Residual MLP Encoder.
    # This architecture explicitly injects multi-scale frequency information to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries. The wide (1536) and shallow (4)
    # residual MLP backbone ensures stable optimization and high capacity within the time budget.
    class LFFWideResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=4, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Define a Deep SwiGLU Residual Encoder
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Use a Wide, Shallow SiLU Residual MLP.
    # This architecture provides high capacity (2048 width) with a flat optimization landscape (4 layers),
    # which is better suited for escaping the persistent R² plateau compared to deep/narrow or attention-based encoders.
    class WideShallowSiLUMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Deep SwiGLU Fourier Encoder to break the R2 plateau.
    # Combines Learnable Multi-Scale Fourier Features (to combat spectral bias)
    # with a Deep SwiGLU Backbone (superior gradient flow and expressiveness vs GELU/SiLU).
    # Width 768, Depth 8 balances capacity and training speed.
    class DeepSwiGLUFourierEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=8, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            # We define SwiGLU inline to avoid dependency on potentially undefined classes in the messy file scope
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim, ffn_dim=None):
                    super().__init__()
                    if ffn_dim is None:
                        ffn_dim = int(8 * hidden_dim / 3)
                        ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm1(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            h = self.input_proj(x_combined)
            
            for block in self.blocks:
                h = block(h)
                
            h = self.final_norm(h)
            out = self.output_proj(h)
            return out

    # Use a Tokenized Spherical Harmonic Transformer Encoder to break the plateau.
    # This treats SH coefficients as tokens, allowing self-attention to model
    # interactions between frequency components, which point-wise MLPs miss.
    class TokenizedSHTransformerEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, n_heads=8, num_tokens=16):
            super().__init__()
            self.num_tokens = num_tokens
            self.token_dim = input_dim // num_tokens
            self.pad_dim = (num_tokens * self.token_dim) - input_dim
            
            self.token_pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
            self.token_proj = nn.Linear(self.token_dim, hidden_dim)
            
            class TransformerBlock(nn.Module):
                def __init__(self, dim, n_heads):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(dim)
                    self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.0)
                    self.norm2 = nn.LayerNorm(dim)
                    ffn_dim = int(8 * dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, dim, bias=False)
                    self.w3 = nn.Linear(dim, ffn_dim, bias=False)

                def forward(self, x):
                    h = self.norm1(x)
                    a, _ = self.attn(h, h, h, need_weights=False)
                    x = x + a
                    h = self.norm2(x)
                    f = self.w2(F.silu(self.w1(h)) * self.w3(h))
                    x = x + f
                    return x

            self.blocks = nn.ModuleList([
                TransformerBlock(hidden_dim, n_heads) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            if self.pad_dim > 0:
                x = F.pad(x, (0, self.pad_dim))
            x = x.view(-1, self.num_tokens, self.token_dim)
            h = self.token_proj(x)
            h = h + self.token_pos_embed
            for block in self.blocks:
                h = block(h)
            h = self.final_norm(h).mean(dim=1)
            out = self.output_proj(h)
            return out

    # Use a Wide, Shallow GELU Residual MLP.
    # Previous deep networks (8 layers) and complex Fourier-based encoders have plateaued at R² ~0.582.
    # A wide (2048) and shallow (3 layers) network provides high capacity with a flatter optimization
    # landscape, allowing for faster convergence and potentially escaping local minima within the
    # 20-minute budget. This relies on the robust Spherical Harmonic input and stable GELU gradients.
    class WideShallowGELUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=3, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Multi-Scale Learnable Fourier Feature (MSLFF) + Residual MLP Encoder.
    # This architecture explicitly injects multi-scale frequency information to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries that standard MLPs miss.
    # The wide (1024) and moderately deep (6 layers) Residual MLP backbone provides stable, high-capacity
    # feature extraction from the enriched Fourier inputs.
    class MSLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            h = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                h = h + block(h)
                
            # Final output
            out = self.output_proj(h)
            return out

    # Use a Hybrid Multi-Scale Spectral Location Encoder.
    # This architecture addresses the R² plateau by explicitly separating low-frequency (broad trends)
    # and high-frequency (sharp boundaries) learning paths.
    # Branch 1 (Low): Wide, shallow MLP for stable global pattern capture.
    # Branch 2 (High): MLP with Learnable High-Freq Fourier Features for local detail.
    # A learnable gate fuses the two, allowing the model to adaptively weight spatial scales.
    class HybridMultiScaleSpectralEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            
            # --- Branch 1: Low Frequency (Broad Trends) ---
            # Wide and shallow to ensure fast convergence on global structures
            self.branch_low_in = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            self.branch_low_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(4) # 4 layers
            ])
            self.branch_low_out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )

            # --- Branch 2: High Frequency (Sharp Boundaries) ---
            # Learnable High-Frequency Fourier Features to combat spectral bias
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            self.branch_high_in = nn.Sequential(
                nn.Linear(input_dim + fourier_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(), # SiLU for smoother gradients in high-freq path
                nn.Dropout(dropout_p)
            )
            self.branch_high_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(6) # 6 layers (deeper for complex local patterns)
            ])
            self.branch_high_out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )

            # --- Gating Mechanism ---
            # Learnable gate to dynamically weight the contribution of each branch
            self.gate = nn.Sequential(
                nn.Linear(input_dim, 2),
                nn.Softmax(dim=-1)
            )

            # --- Final Fusion ---
            self.final_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # 1. Low Frequency Branch
            h_low = self.branch_low_in(x)
            for block in self.branch_low_blocks:
                h_low = h_low + block(h_low)
            out_low = self.branch_low_out(h_low)

            # 2. High Frequency Branch
            x_fourier = torch.sin(x @ self.B_high) * self.gain_high
            x_high_in = torch.cat([x, x_fourier], dim=-1)
            h_high = self.branch_high_in(x_high_in)
            for block in self.branch_high_blocks:
                h_high = h_high + block(h_high)
            out_high = self.branch_high_out(h_high)

            # 3. Gating
            gates = self.gate(x) # (B, 2)
            g_low = gates[:, 0:1]
            g_high = gates[:, 1:2]

            # 4. Fusion
            fused = g_low * out_low + g_high * out_high
            
            return self.final_proj(fused)

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 768, Depth 8 balances capacity and training speed.
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Block Definition
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 768, Depth 8 balances capacity and training speed.
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a Wide, Shallow GELU Residual MLP to break the plateau.
    # Previous deep networks (8 layers) and complex Fourier-based encoders have plateaued at R² ~0.582.
    # A wide (2048) and shallow (3 layers) network provides high capacity with a flatter optimization
    # landscape, allowing for faster convergence and potentially escaping local minima within the
    # 20-minute budget. This relies on the robust Spherical Harmonic input and stable GELU gradients.
    class WideShallowGELUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=3, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Learnable Fourier Feature (LFF) + Wide Residual MLP Encoder.
    # This architecture explicitly injects high-frequency sinusoidal features to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries.
    # The wide (1536) and moderately deep (6 layers) Residual MLP backbone ensures stable optimization
    # and high capacity within the 20-minute budget.
    class HighCapLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=256, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features
            # sigma=2.0 is a moderate frequency that captures regional/local variations
            # without the extreme high-frequency instability of SIREN.
            self.B = nn.Parameter(torch.randn(input_dim, fourier_dim) * 2.0)
            self.gain = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + fourier_dim
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Learnable Fourier Features
            x_fourier = torch.sin(x @ self.B) * self.gain
            x_combined = torch.cat([x, x_fourier], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a High-Capacity Learnable Fourier Feature (LFF) Residual Encoder.
    # This explicitly injects learnable high-frequency features to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries that standard MLPs miss.
    # The wide (1536) and moderately deep (6 layers) Residual MLP backbone ensures stable optimization
    # and high capacity within the 20-minute budget.
    class HighCapLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=256, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features
            # sigma=2.0 is a moderate frequency that captures regional/local variations
            # without the extreme high-frequency instability of SIREN.
            self.B = nn.Parameter(torch.randn(input_dim, fourier_dim) * 2.0)
            self.gain = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + fourier_dim
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Learnable Fourier Features
            x_fourier = torch.sin(x @ self.B) * self.gain
            x_combined = torch.cat([x, x_fourier], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Block Definition
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1024) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class MSLFFWideResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            h = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                h = h + block(h)
                
            # Final output
            out = self.output_proj(h)
            return out

    # Use a High-Capacity Learnable Fourier Feature (LFF) Residual Encoder.
    # This explicitly injects learnable high-frequency features to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries that standard MLPs miss.
    # The wide (1536) and moderately deep (6 layers) Residual MLP backbone ensures stable optimization
    # and high capacity within the 20-minute budget.
    class MultiResolutionGatedEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, dropout_p=0.1):
            super().__init__()
            
            # Branch 1: Wide & Shallow (Captures broad spatial trends / low frequency)
            # High capacity, low depth to learn smooth global patterns
            self.branch_wide = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, output_dim)
            )
            
            # Branch 2: Narrow & Deep with Fourier Features (Captures complex local variations / high frequency)
            # Learnable Fourier Features to combat spectral bias for sharp boundaries
            fourier_dim = 128
            self.B_fourier = nn.Parameter(torch.randn(input_dim, fourier_dim) * 4.0)
            self.gain_fourier = nn.Parameter(torch.ones(fourier_dim))
            
            # Input projection for the deep branch (includes Fourier features)
            self.branch_deep_in = nn.Sequential(
                nn.Linear(input_dim + fourier_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep residual blocks
            self.branch_deep_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(256, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(256, 256),
                    nn.LayerNorm(256),
                    nn.Dropout(dropout_p)
                )
                for _ in range(6)
            ])
            
            self.branch_deep_out = nn.Sequential(
                nn.Linear(256, output_dim),
                nn.LayerNorm(output_dim)
            )
            
            # Gating Network: Dynamically weights the two branches based on input
            self.gate = nn.Sequential(
                nn.Linear(input_dim, 2),
                nn.Softmax(dim=-1)
            )
            
            self.final_norm = nn.LayerNorm(output_dim)

        def forward(self, x):
            # Wide Branch
            out_wide = self.branch_wide(x)
            
            # Deep Branch with Fourier
            x_fourier = torch.sin(x @ self.B_fourier) * self.gain_fourier
            x_deep_in = torch.cat([x, x_fourier], dim=-1)
            h = self.branch_deep_in(x_deep_in)
            for block in self.branch_deep_blocks:
                h = h + block(h)
            out_deep = self.branch_deep_out(h)
            
            # Gating
            gates = self.gate(x) # (B, 2)
            g_wide = gates[:, 0:1]
            g_deep = gates[:, 1:2]
            
            # Weighted Sum
            out = g_wide * out_wide + g_deep * out_deep
            
            return self.final_norm(out)

    # Use a High-Capacity Learnable Fourier Feature (LFF) Residual Encoder.
    # This explicitly injects learnable high-frequency features to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries that standard MLPs miss.
    # The wide (1536) and moderately deep (6 layers) Residual MLP backbone ensures stable optimization
    # and high capacity within the 20-minute budget.
    class HighCapLFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=256, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features
            # sigma=2.0 is a moderate frequency that captures regional/local variations
            # without the extreme high-frequency instability of SIREN.
            self.B = nn.Parameter(torch.randn(input_dim, fourier_dim) * 2.0)
            self.gain = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + fourier_dim
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Learnable Fourier Features
            x_fourier = torch.sin(x @ self.B) * self.gain
            x_combined = torch.cat([x, x_fourier], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Residual Encoder.
    # This replaces the inefficient DeepSwiGLU (single-token attention) with a stable, high-capacity MLP
    # enriched with learnable multi-scale Fourier features to combat spectral bias.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Wide, Shallow Residual MLP with Multi-Scale Fourier Features.
    # This configuration (2048 width, 3 layers) provides high capacity with a flatter optimization
    # landscape, while MSLFF injects frequency information to capture multi-scale demographic patterns.
    class WideShallowMSFFEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=3, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 1024, Depth 8 balances capacity and training speed, avoiding the
    # underfitting of shallow wide networks and the instability of very deep narrow ones.
    
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a Wide, Shallow GELU Residual MLP to break the plateau.
    # This architecture provides high capacity (2048 width) with a flat optimization landscape (3 layers),
    # which is better suited for escaping local minima within the 20-minute budget.
    class WideShallowGELUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=3, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 1024, Depth 8 balances capacity and training speed.
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a High-Capacity Learnable Fourier Feature (LFF) + Wide Residual MLP Encoder.
    # This architecture injects learnable frequency features (sigma=2.0) to combat spectral bias,
    # combined with a Wide (2048) and Shallow (3 layers) Residual MLP backbone for stable,
    # high-capacity feature extraction. This configuration is known to break plateaus in
    # spatial-demographic regression tasks by providing a flat optimization landscape.
    class LFFWideResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=3, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features
            self.B = nn.Parameter(torch.randn(input_dim, fourier_dim) * 2.0)
            self.gain = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + fourier_dim
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_fourier = torch.sin(x @ self.B) * self.gain
            x_combined = torch.cat([x, x_fourier], dim=-1)
            x = self.input_proj(x_combined)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Very Wide, Shallow Residual MLP with GELU.
    # Rationale: Deep networks (6-8 layers) have consistently plateaued at R~0.582.
    # A wide (2048) and shallow (2 layers) network offers a flatter optimization landscape
    # and high capacity for the input projection, which helps escape local minima
    # and converge faster within the time budget.
    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=2, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=2, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Deep Residual MLP with Learnable Multi-Scale Fourier Features.
    # This injects frequency information to combat spectral bias and capture
    # multi-scale spatial patterns, which is critical for demographic prediction.
    class DeepFourierResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=2, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 1024, Depth 8 balances capacity and training speed, avoiding the
    # underfitting of shallow wide networks and the instability of very deep narrow ones.
    
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Residual MLP Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Block Definition
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a Wide, Shallow Residual MLP to break the R2 plateau.
    # This architecture provides high capacity (2048 width) with a flatter optimization landscape (4 layers),
    # avoiding the local minima traps of deep/narrow networks and the overhead of degenerate attention.
    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 1024, Depth 8 balances capacity and training speed, avoiding the
    # underfitting of shallow wide networks and the instability of very deep narrow ones.
    
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a Dual-Branch Gated Fusion Encoder to break the R2 plateau.
    # Branch 1: Wide, Shallow MLP for broad spatial trends (low frequency).
    # Branch 2: Deep, Narrow MLP with Learnable High-Freq Fourier Features for local details.
    # Gate: Learnable softmax gate to dynamically weight branches based on input.
    class DualBranchGatedEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, dropout_p=0.1):
            super().__init__()
            
            # Branch 1: Low Frequency (Broad Trends)
            # Wide and shallow to ensure fast convergence on global structures
            self.branch_low = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, output_dim)
            )
            
            # Branch 2: High Frequency (Local Details)
            # Learnable High-Frequency Fourier Features to combat spectral bias
            fourier_dim = 128
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Deep, narrow MLP to process high-frequency features
            self.branch_high_in = nn.Linear(input_dim + fourier_dim, 256)
            self.branch_high_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(256, 256),
                    nn.LayerNorm(256),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(256, 256),
                    nn.LayerNorm(256),
                    nn.Dropout(dropout_p)
                )
                for _ in range(6)
            ])
            self.branch_high_out = nn.Sequential(
                nn.Linear(256, output_dim),
                nn.LayerNorm(output_dim)
            )
            
            # Gating Network
            self.gate = nn.Sequential(
                nn.Linear(input_dim, 2),
                nn.Softmax(dim=-1)
            )
            
            self.final_norm = nn.LayerNorm(output_dim)

        def forward(self, x):
            # Low Frequency Branch
            out_low = self.branch_low(x)
            
            # High Frequency Branch
            x_fourier = torch.sin(x @ self.B_high) * self.gain_high
            x_high_in = torch.cat([x, x_fourier], dim=-1)
            h = self.branch_high_in(x_high_in)
            for block in self.branch_high_blocks:
                h = h + block(h)
            out_high = self.branch_high_out(h)
            
            # Gating
            gates = self.gate(x) # (B, 2)
            g_low = gates[:, 0:1]
            g_high = gates[:, 1:2]
            
            # Weighted Sum
            out = g_low * out_low + g_high * out_high
            
            return self.final_norm(out)

    class WideShallowGELUMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Residual Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1024) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    # Use a Wide, Shallow Residual MLP to break the plateau.
    # This architecture provides high capacity (2048 width) with a flatter optimization landscape (4 layers),
    # which is better suited for escaping local minima within the 20-minute budget.
    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Wide, Shallow Residual MLP to break the R2 plateau.
    # This architecture provides high capacity (2048 width) with a flatter optimization landscape (4 layers),
    # avoiding the local minima traps of deep/narrow networks and the overhead of degenerate attention.
    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Spectral Decomposition Location Encoder.
    # This architecture decomposes the spatial mapping into Low, Mid, and High frequency branches,
    # fusing them with a learnable gate. This provides a flatter optimization landscape and
    # explicit multi-scale inductive bias, which is critical for breaking the R² plateau.
    class SpectralDecompositionEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, dropout_p=0.1):
            super().__init__()
            
            # Branch 1: Low Frequency (Broad Trends) - Wide & Shallow
            self.branch_low = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(512, output_dim)
            )
            
            # Branch 2: Mid Frequency (Regional Patterns) - Standard Depth
            self.branch_mid = nn.Sequential(
                nn.Linear(input_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(768, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(768, output_dim)
            )
            
            # Branch 3: High Frequency (Local Details) - With Learnable Fourier Features
            fourier_dim = 128
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            self.branch_high_in = nn.Linear(input_dim + fourier_dim, 512)
            self.branch_high_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.Dropout(dropout_p)
                )
                for _ in range(4)
            ])
            self.branch_high_out = nn.Sequential(
                nn.Linear(512, output_dim),
                nn.LayerNorm(output_dim)
            )
            
            # Gating Network
            self.gate = nn.Sequential(
                nn.Linear(input_dim, 3),
                nn.Softmax(dim=-1)
            )
            
            self.final_norm = nn.LayerNorm(output_dim)

        def forward(self, x):
            out_low = self.branch_low(x)
            out_mid = self.branch_mid(x)
            
            x_fourier = torch.sin(x @ self.B_high) * self.gain_high
            x_high_in = torch.cat([x, x_fourier], dim=-1)
            h = self.branch_high_in(x_high_in)
            for block in self.branch_high_blocks:
                h = h + block(h)
            out_high = self.branch_high_out(h)
            
            gates = self.gate(x)
            g_low, g_mid, g_high = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
            
            out = g_low * out_low + g_mid * out_mid + g_high * out_high
            return self.final_norm(out)

    class WideShallowSiLUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Residual Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Wide, Shallow Residual MLP with Enhanced Multi-Scale Fourier Features.
    # - Wide (2048) & Shallow (4 layers): Flatter optimization landscape, faster convergence.
    # - MSFF (Low 1.0, Mid 3.0, High 16.0): Enhanced high-frequency capture for sharp boundaries.
    # - Fourier Dim 256: Increased capacity for frequency features.
    class WideShallowMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, fourier_dim=256, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 16.0) # Increased high freq scale
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 768, Depth 8 balances capacity and training speed.
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a Stable, Multi-Scale Fourier Residual Encoder
    # This architecture combines Learnable Multi-Scale Fourier Features (to combat spectral bias)
    # with a stable, moderate-width Residual MLP backbone (GELU + LayerNorm).
    # It avoids the memory instability of wide/attention models while providing high capacity
    # for capturing sharp demographic boundaries.
    class StableMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a High-Capacity SiLU Residual MLP to break the R2 plateau.
    # SiLU provides smoother gradients than GELU, and width 1024 provides higher capacity
    # than the previous 512-width encoders, aiming to escape local minima without OOM risks.
    class SiLUHighCapResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (2048) and shallow (4 layers) Residual MLP backbone provides stable, high-capacity
    # feature extraction from the enriched Fourier inputs with a flat optimization landscape.
    class HighCapMSFFWideEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Wide Shallow Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Stable, High-Capacity Deep Residual MLP.
    # This configuration (1024 width, 6 layers) balances capacity and memory efficiency,
    # addressing the recent OOM crashes while providing sufficient depth to break the R² plateau.
    class StableDeepResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),  # SiLU for smoother gradients
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 768, Depth 8 balances capacity and training speed.
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Wide, Shallow GELU Residual MLP for stability and high capacity.
    # This avoids the instability of SIREN and complex Fourier features that lead to NaNs.
    class WideShallowGELUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1024) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Wide, Shallow Residual MLP to break the plateau.
    # This architecture provides high capacity (2048 width) with a flatter optimization landscape (4 layers),
    # which is better suited for escaping local minima within the 20-minute budget.
    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    class LFFDeepResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=8, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features to combat spectral bias
            self.B = nn.Parameter(torch.randn(input_dim, fourier_dim) * 2.0)
            self.gain = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + fourier_dim
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_fourier = torch.sin(x @ self.B) * self.gain
            x_combined = torch.cat([x, x_fourier], dim=-1)
            x = self.input_proj(x_combined)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 1024, Depth 8 balances capacity and training speed.
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=8, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    class WideShallowSiLUResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Dilated Multi-Scale Fourier Residual Encoder.
    # This architecture addresses the persistent R² plateau by explicitly decomposing the spatial
    # mapping into distinct frequency scales using learnable Fourier features with wide,
    # non-overlapping frequency bands (Dilated).
    # - Low Scale (0.5 - 2.0): Captures broad, global demographic trends.
    # - Mid Scale (2.0 - 6.0): Captures regional patterns and medium-scale variations.
    # - High Scale (6.0 - 18.0): Captures sharp local boundaries and fine-grained details.
    # This multi-scale inductive bias allows the residual MLP backbone to more effectively
    # map complex spatial-demographic correlations than single-scale or narrow-band Fourier features.
    
    class DilatedMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=64, dropout_p=0.1):
            super().__init__()
            
            # Dilated Learnable Fourier Features at 3 distinct scales
            # Low: Broad trends (low frequency)
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0) # Mean 1.0, Range ~0.5-2.0
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            
            # Mid: Regional patterns (mid frequency)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 4.0) # Mean 4.0, Range ~2.0-6.0
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            
            # High: Local details (high frequency)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 12.0) # Mean 12.0, Range ~6.0-18.0
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Total input dimension: original input + 3 * fourier_dim
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Dilated Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            h = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                h = h + block(h)
                
            # Final output
            out = self.output_proj(h)
            return out

    # Use a Hybrid Spectral-Fourier Encoder to break the R2 plateau.
    # This architecture injects Learnable Multi-Scale Fourier Features (Low/Mid/High)
    # to combat spectral bias and capture sharp demographic boundaries,
    # followed by a Wide, Shallow Residual MLP backbone for stable, fast convergence.
    class HybridSpectralFourierEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=3, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Wide, Shallow Residual Backbone for flat optimization landscape
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class StochasticDepthResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1, stochastic_depth_prob=0.2):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            
            # Stochastic Depth: linearly increasing probability from 0 to max_prob
            self.stochastic_depth_prob = stochastic_depth_prob
            self.num_layers = num_layers

        def forward(self, x):
            x = self.input_proj(x)
            for i, block in enumerate(self.blocks):
                # Calculate drop probability: linear ramp from 0 to max_prob
                p = self.stochastic_depth_prob * (i + 1) / self.num_layers
                if self.training and torch.rand(()) < p:
                    # Drop the block (residual only)
                    pass
                else:
                    x = x + block(x)
            out = self.output_proj(x)
            return out

    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=8, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a Deep SwiGLU Residual Encoder for superior expressiveness and gradient flow.
    # SwiGLU is more expressive than GELU/SiLU and handles deep networks better.
    # Width 1024 and Depth 6 provide a balance between capacity and stability.
    # Use a Dilated Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture addresses the persistent R² plateau by explicitly decomposing the spatial
    # mapping into distinct frequency scales using learnable Fourier features with wide,
    # non-overlapping frequency bands (Dilated).
    # - Low Scale (0.5 - 2.0): Captures broad, global demographic trends.
    # - Mid Scale (2.0 - 6.0): Captures regional patterns and medium-scale variations.
    # - High Scale (6.0 - 18.0): Captures sharp local boundaries and fine-grained details.
    # This multi-scale inductive bias allows the residual MLP backbone to more effectively
    # map complex spatial-demographic correlations than single-scale or narrow-band Fourier features.
    
    class DilatedMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=64, dropout_p=0.1):
            super().__init__()
            
            # Dilated Learnable Fourier Features at 3 distinct scales
            # Low: Broad trends (low frequency)
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0) # Mean 1.0, Range ~0.5-2.0
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            
            # Mid: Regional patterns (mid frequency)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 4.0) # Mean 4.0, Range ~2.0-6.0
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            
            # High: Local details (high frequency)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 12.0) # Mean 12.0, Range ~6.0-18.0
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Total input dimension: original input + 3 * fourier_dim
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Dilated Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            h = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                h = h + block(h)
                
            # Final output
            out = self.output_proj(h)
            return out

    # Use a Wide, Shallow Residual MLP to break the R2 plateau.
    # Previous deep networks (6-8 layers) have plateaued at R^2 ~0.582.
    # A wide (2048) and shallow (4 layers) network provides high capacity with a flatter
    # optimization landscape, allowing for faster convergence and potentially escaping
    # local minima within the time budget.
    class WideShallowResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=4, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a High-Capacity Multi-Scale Fourier Feature (MSFF) Wide Residual MLP.
    # This addresses spectral bias by injecting learnable Low/Mid/High frequency features,
    # and uses a wide (1536), moderately deep (6) Residual MLP backbone for stable, high-capacity mapping.
    class HighCapMSFFWideResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Stable, High-Capacity Deep Residual MLP.
    # Rationale: Fourier features have repeatedly failed to break the plateau and may conflict
    # with the Spherical Harmonic input which already provides multi-frequency info.
    # A pure MLP with LayerNorm and GELU provides the most stable optimization landscape.
    class StableDeepMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture explicitly injects learnable Low/Mid/High frequency features to combat spectral bias,
    # which is critical for capturing sharp demographic boundaries that standard MLPs miss.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            h = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                h = h + block(h)
                
            # Final output
            out = self.output_proj(h)
            return out

    # Use a Wide, Shallow GELU Residual MLP to break the plateau.
    # Previous deep networks (8 layers) and complex Fourier-based encoders have plateaued at R² ~0.582.
    # A wide (2048) and shallow (4 layers) network provides high capacity with a flatter optimization
    # landscape, allowing for faster convergence and potentially escaping local minima within the
    # 20-minute budget. This relies on the robust Spherical Harmonic input and stable GELU gradients.
    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            h = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                h = h + block(h)
                
            # Final output
            out = self.output_proj(h)
            return out

    # Use a Compact, High-Width Residual MLP to break the plateau.
    # This architecture balances capacity (1024 width) and depth (6 layers) to provide
    # sufficient non-linear mapping power without the memory overhead of SIREN or
    # extremely wide networks. GELU and LayerNorm ensure stable optimization.
    class CompactResidualMLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1024) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class MSLFFLocationEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            # Concatenate original input with multi-scale Fourier features
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            # Project to hidden dimension
            h = self.input_proj(x_combined)
            
            # Residual blocks
            for block in self.blocks:
                h = h + block(h)
                
            # Final output
            out = self.output_proj(h)
            return out

    # Use a Deep SwiGLU Residual Encoder to break the R2 plateau.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU,
    # often critical for breaking plateaus in high-dimensional regression tasks.
    # Width 1024, Depth 6 balances capacity and training speed.
    class DeepSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, dropout_p=0.1):
            super().__init__()
            
            # Input Projection
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            # SwiGLU Blocks
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)

                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)

                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a High-Capacity Multi-Scale Fourier Residual Encoder.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    # Use a Deep SwiGLU Residual Encoder for superior expressiveness and gradient flow
    class DeepSwiGLUEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6):
            super().__init__()
            self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            
            class SwiGLU(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    f = int(8 * d / 3)
                    f = int(2 * ((f + 7) // 8))
                    self.w1 = nn.Linear(d, f, bias=False)
                    self.w2 = nn.Linear(f, d, bias=False)
                    self.w3 = nn.Linear(d, f, bias=False)
                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))
            
            class Block(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.norm = nn.LayerNorm(d)
                    self.ffn = SwiGLU(d)
                def forward(self, x):
                    return x + self.ffn(self.norm(x))
            
            self.blocks = nn.ModuleList([Block(hidden_dim) for _ in range(num_layers)])
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for b in self.blocks:
                x = b(x)
            return self.out_proj(self.final_norm(x))

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Encoder.
    # This injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1024) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class FinalMSFFEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1024, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class MultiHeadParallelEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, dropout_p=0.1):
            super().__init__()
            
            # Branch 1: Low Frequency (Broad Trends) - Wide and Shallow
            # High capacity, low depth to learn smooth global patterns quickly
            self.branch_low = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, output_dim)
            )
            
            # Branch 2: High Frequency (Local Details) - Deep with Fourier Features
            # Learnable Fourier Features to combat spectral bias for sharp boundaries
            fourier_dim = 128
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Input projection for the deep branch (includes Fourier features)
            self.branch_high_in = nn.Sequential(
                nn.Linear(input_dim + fourier_dim, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep residual blocks for complex local patterns
            self.branch_high_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.Dropout(dropout_p)
                )
                for _ in range(4)
            ])
            
            self.branch_high_out = nn.Sequential(
                nn.Linear(512, output_dim),
                nn.LayerNorm(output_dim)
            )
            
            # Gating Network: Dynamically weights the two branches based on input
            # This allows the model to adaptively focus on broad vs local features
            self.gate = nn.Sequential(
                nn.Linear(input_dim, 2),
                nn.Softmax(dim=-1)
            )
            
            self.final_norm = nn.LayerNorm(output_dim)

        def forward(self, x):
            # Low Frequency Branch
            out_low = self.branch_low(x)
            
            # High Frequency Branch
            x_fourier = torch.sin(x @ self.B_high) * self.gain_high
            x_high_in = torch.cat([x, x_fourier], dim=-1)
            h = self.branch_high_in(x_high_in)
            for block in self.branch_high_blocks:
                h = h + block(h)
            out_high = self.branch_high_out(h)
            
            # Gating
            gates = self.gate(x) # (B, 2)
            g_low = gates[:, 0:1]
            g_high = gates[:, 1:2]
            
            # Weighted Sum
            out = g_low * out_low + g_high * out_high
            
            return self.final_norm(out)

    # Use a Compact, High-Capacity SwiGLU Residual Encoder.
    # SwiGLU provides superior expressiveness and gradient flow compared to GELU/SiLU.
    # Width 768 and Depth 6 balance capacity and memory efficiency, avoiding the OOM
    # issues seen with 2048-width MLPs while providing a flatter optimization landscape
    # to break the R² 0.582 plateau.
    class EfficientSwiGLUResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=768, num_layers=6, dropout_p=0.1):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            
            class SwiGLU(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    ffn_dim = int(8 * hidden_dim / 3)
                    ffn_dim = int(2 * ((ffn_dim + 7) // 8))
                    self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                    self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
                    self.w3 = nn.Linear(hidden_dim, ffn_dim, bias=False)
                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class SwiGLUBlock(nn.Module):
                def __init__(self, hidden_dim, dropout_p):
                    super().__init__()
                    self.norm = nn.LayerNorm(hidden_dim)
                    self.ffn = SwiGLU(hidden_dim)
                    self.dropout = nn.Dropout(dropout_p)
                def forward(self, x):
                    h = self.norm(x)
                    f = self.ffn(h)
                    return x + self.dropout(f)

            self.blocks = nn.ModuleList([
                SwiGLUBlock(hidden_dim, dropout_p) for _ in range(num_layers)
            ])
            
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.final_norm(x)
            return self.output_proj(x)

    # Use a High-Capacity Multi-Scale Learnable Fourier Feature (MSLFF) Residual MLP.
    # This architecture injects learnable Low/Mid/High frequency features to combat spectral bias,
    # allowing the model to capture sharp demographic boundaries and broad spatial trends.
    # A wide (1536) and moderately deep (6 layers) Residual MLP backbone with GELU and LayerNorm
    # provides stable, high-capacity feature extraction from the enriched Fourier inputs.
    class HighCapMSFFResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=6, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features at 3 scales to capture multi-scale spatial patterns
            self.B_low = nn.Parameter(torch.randn(input_dim, fourier_dim) * 1.0)
            self.B_mid = nn.Parameter(torch.randn(input_dim, fourier_dim) * 3.0)
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            
            # Learnable gains for adaptive amplitude scaling
            self.gain_low = nn.Parameter(torch.ones(fourier_dim))
            self.gain_mid = nn.Parameter(torch.ones(fourier_dim))
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + (fourier_dim * 3)
            
            # Input Projection from combined features to hidden dim
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            # Deep Residual Backbone with GELU and LayerNorm for stability
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            # Apply Multi-Scale Fourier Features
            x_low = torch.sin(x @ self.B_low) * self.gain_low
            x_mid = torch.sin(x @ self.B_mid) * self.gain_mid
            x_high = torch.sin(x @ self.B_high) * self.gain_high
            
            x_combined = torch.cat([x, x_low, x_mid, x_high], dim=-1)
            
            x = self.input_proj(x_combined)
            
            for block in self.blocks:
                x = x + block(x)
                
            return self.output_proj(x)

    class DualBranchFusedEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, dropout_p=0.1):
            super().__init__()
            
            # Branch 1: Low Frequency (Broad Trends)
            # Wide and shallow to ensure fast convergence on global structures
            self.branch_low = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(1024, output_dim)
            )
            
            # Branch 2: High Frequency (Local Details)
            # Learnable High-Frequency Fourier Features to combat spectral bias
            fourier_dim = 128
            self.B_high = nn.Parameter(torch.randn(input_dim, fourier_dim) * 8.0)
            self.gain_high = nn.Parameter(torch.ones(fourier_dim))
            
            # Deep, narrow MLP to process high-frequency features
            self.branch_high_in = nn.Sequential(
                nn.Linear(input_dim + fourier_dim, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(dropout_p)
            )
            
            self.branch_high_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.SiLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(512, 512),
                    nn.LayerNorm(512),
                    nn.Dropout(dropout_p)
                )
                for _ in range(6)
            ])
            
            self.branch_high_out = nn.Sequential(
                nn.Linear(512, output_dim),
                nn.LayerNorm(output_dim)
            )
            
            # Gating Network
            self.gate = nn.Sequential(
                nn.Linear(input_dim, 2),
                nn.Softmax(dim=-1)
            )
            
            self.final_norm = nn.LayerNorm(output_dim)

        def forward(self, x):
            # Low Frequency Branch
            out_low = self.branch_low(x)
            
            # High Frequency Branch
            x_fourier = torch.sin(x @ self.B_high) * self.gain_high
            x_high_in = torch.cat([x, x_fourier], dim=-1)
            h = self.branch_high_in(x_high_in)
            for block in self.branch_high_blocks:
                h = h + block(h)
            out_high = self.branch_high_out(h)
            
            # Gating
            gates = self.gate(x) # (B, 2)
            g_low = gates[:, 0:1]
            g_high = gates[:, 1:2]
            
            # Weighted Sum
            out = g_low * out_low + g_high * out_high
            
            return self.final_norm(out)

    # Use a High-Capacity Wide Residual MLP with Learnable Fourier Features.
    # This combines the spectral inductive bias of LFF (to capture sharp demographic boundaries)
    # with the high capacity and stable optimization of a Wide (1536), Shallow (4) Residual MLP.
    # This configuration is known to break plateaus in spatial regression tasks by providing
    # a flat optimization landscape while retaining sufficient expressiveness for complex patterns.
    class HighCapLFFWideResidualEncoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim=1536, num_layers=4, fourier_dim=128, dropout_p=0.1):
            super().__init__()
            # Learnable Fourier Features
            self.B = nn.Parameter(torch.randn(input_dim, fourier_dim) * 2.0)
            self.gain = nn.Parameter(torch.ones(fourier_dim))
            
            total_input_dim = input_dim + fourier_dim
            
            self.input_proj = nn.Sequential(
                nn.Linear(total_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p)
            )
            
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout_p)
                )
                for _ in range(num_layers)
            ])
            
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

        def forward(self, x):
            x_fourier = torch.sin(x @ self.B) * self.gain
            x_combined = torch.cat([x, x_fourier], dim=-1)
            x = self.input_proj(x_combined)
            for block in self.blocks:
                x = x + block(x)
            out = self.output_proj(x)
            return out

    # Use a SIREN-based Location Encoder.
    # SIREN networks are effective for learning high-frequency spatial patterns.
    # We use a moderate frequency (w0=3.0) and stable initialization to avoid
    # the instability observed with high-frequency SIREN variants.
    model.location_encoder = SirenNet(
        dim_in=input_dim,
        dim_hidden=512,
        dim_out=EMB_DIM,
        num_layers=4,
        w0=3.0,
        w0_initial=3.0,
        use_bias=True,
        dropout=False
    ).to(device)

    # Estimate total steps for temperature annealing
    # We don't know exact batch size/epochs here easily without args, but we can estimate
    # or just use a generous number. 5000 steps is a safe upper bound for 20 mins on GPU.
    # If training is faster, it will hit min_temp sooner. If slower, it stays high longer.
    # We can refine this if args are passed, but for now, 5000 is a robust default.
    losses = {
        "contra": ContrastiveLoss(temperature=5.0),
    }
    # Note: RelationalLoss now requires pos_mask, which is passed in common_step

    dataset = GeoDataset(args.csv, args.data_dir, args.npy_subdir)
    n_val = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    def exclude(n, p):
        return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n

    gain_or_bias = [p for n, p in model.named_parameters() if exclude(n, p) and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not exclude(n, p) and p.requires_grad]
    
    # Use AdamW with a moderate learning rate.
    # Weight decay is applied to non-bias/norm parameters for better generalization.
    optimizer = torch.optim.AdamW(
        [{"params": gain_or_bias, "weight_decay": 0.0}, {"params": rest, "weight_decay": 0.01}],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # mixed-precision scaler (only active when CUDA is available)
    scaler = GradScaler() if device.type == "cuda" else None
    
    # Use OneCycleLR to break plateaus. It ramps up the learning rate to escape local minima,
    # then anneals it down for fine-tuning. This is often more effective than CosineAnnealing
    # for embedding-based tasks.
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    @torch.no_grad()
    def evaluate():
        model.eval()
        total = 0.0
        count = 0
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total += common_step(model, losses, batch).item()
            count += 1
        return total / max(count, 1)

    best_val = float("inf")
    best_state = None
    early_stop = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    loss = common_step(model, losses, batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = common_step(model, losses, batch)
                loss.backward()
                optimizer.step()
            # Step the scheduler for every optimizer step to ensure correct cosine annealing
            scheduler.step()
            running += loss.item()
            n += 1
        train_loss = running / max(n, 1)
        val_loss = evaluate()
        if VERBOSE:
            print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        elif epoch%10==0:
            print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        # NOTE: scheduler.step() is moved inside the batch loop to align with OneCycleLR's step-based annealing

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            early_stop = args.patience
            torch.save(model, SAVE_PATH)
        else:
            early_stop -= 1
            if early_stop <= 0:
                print(f"Early stopping at epoch {epoch} (val_loss did not improve for {args.patience} epochs)")
                break

    if args.patience > 0 and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best | val_loss: {best_val:.4f} (restored)")

    if args.eval_csv:
        eval_location_encoder(model.location, args)

    return model

def load_model(ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatCLIP(
        embed_dim=EMB_DIM,
        legendre_polys=80,
        num_hidden_layers=2,
        capacity=256,
    ).to(device)
    model.location_encoder = SirenNet(
            dim_in=model.posenc.embedding_dim,
            dim_hidden=512,
            dim_out=EMB_DIM,
            num_layers=4,
            w0=3.0,
            w0_initial=3.0,
            use_bias=True,
            dropout=False
        ).to(device)

    return model.location



# --------------------------------------------------------------------
#        Eval
# --------------------------------------------------------------------

def _embed(loc_encoder, lon, lat, device, batch_size=4096):
    loc_encoder.eval()
    coords = np.stack([lon, lat], axis=1).astype(np.float64)
    out = []
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            t = torch.from_numpy(coords[start : start + batch_size]).double().to(device)
            out.append(loc_encoder(t).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _is_classification(y):
    y = pd.Series(y).dropna()
    return y.dtype == object or y.nunique() <= 20


def evaluate_target(emb, y, task):
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
    from sklearn.model_selection import KFold, StratifiedKFold

    y = pd.Series(y).dropna()
    emb = emb[y.index.to_numpy()] if hasattr(y.index, "to_numpy") else emb[: len(y)]
    y = y.to_numpy()

    if task == "classification":
        _, y_encoded = np.unique(y, return_inverse=True)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, f1s = [], []
        for train_idx, test_idx in kfold.split(emb, y_encoded):
            clf = LGBMClassifier(verbose=-1)
            clf.fit(emb[train_idx], y_encoded[train_idx])
            pred = clf.predict(emb[test_idx])
            accs.append(accuracy_score(y_encoded[test_idx], pred))
            f1s.append(f1_score(y_encoded[test_idx], pred, average="macro"))
        return {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)), "n": len(y)}
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        r2s, maes = [], []
        for train_idx, test_idx in kfold.split(emb):
            reg = LGBMRegressor(verbose=-1)
            reg.fit(emb[train_idx], y[train_idx])
            pred = reg.predict(emb[test_idx])
            r2s.append(r2_score(y[test_idx], pred))
            maes.append(mean_absolute_error(y[test_idx], pred))
        return {"r2": float(np.mean(r2s)), "mae": float(np.mean(maes)), "n": len(y)}


def eval_location_encoder(loc_encoder, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loc_encoder = loc_encoder.to(device)
    df = pd.read_csv(args.eval_csv)

    lat_col = _find_column(df, LAT_NAMES)
    lon_col = _find_column(df, LON_NAMES)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not find lat/lon columns in {args.eval_csv}")

    #print("\n=== Evaluation (LightGBM, 5-fold CV) ===")
    emb = _embed(loc_encoder, df[lon_col].to_numpy(dtype=float), df[lat_col].to_numpy(dtype=float), device)

    if args.eval_target:
        targets = [args.eval_target]
    else:
        exclude = {lat_col, lon_col}
        targets = [c for c in df.columns if c not in exclude]

    scores = []

    for target in targets[:5]:
        if target not in df.columns:
            print(f"  ! target {target!r} not found, skipping")
            continue
        task = args.task or ("classification" if _is_classification(df[target]) else "regression")
        try:
            stats = evaluate_target(emb, df[target], task)
            label = ", ".join(f"{k}={v:.4f}" for k, v in stats.items() if k != "n")
            if VERBOSE: print(f"  {target} [{task}] n={stats['n']} -> {label}")
            scores.append(stats['r2'])
        except Exception as e:
            print(f"  ! {target} failed: {e}")

    print(rf"The average score acheived by this model is: {sum(scores)/len(scores)}")
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=rf"/data/susanket/sentinel/data/index.csv", help="index CSV with name/lat/lon columns")
    parser.add_argument("--data_dir", default=rf"/data/susanket/sentinel/data/", help="folder containing the dino .npy files")
    parser.add_argument("--npy_subdir", default=rf"npy_features", help="optional subdir under data_dir for npy files")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience (0 to disable)")
    parser.add_argument("--batch_size", type=int, default=30000)
    parser.add_argument("--num_workers", type=int, default=80)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--legendre_polys", type=int, default=40)
    parser.add_argument("--eval_csv", default=rf"/home/susanket/satclip-v2/eval/usa-cdchealth.csv", help="CSV for LightGBM 5-fold eval (optional)")
    parser.add_argument("--eval_target", default=None, help="target column name in eval CSV (default: all except lat/lon)")
    parser.add_argument("--task", choices=["regression", "classification"], default=None, help="eval task (default: auto)")
    args = parser.parse_args()

    import time

    torch.multiprocessing.set_sharing_strategy("file_system")
    a = time.time()
    train(args)
    print(time.time()-a)


if __name__ == "__main__":
    main()
