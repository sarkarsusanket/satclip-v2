import math
import os
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import sph_harm as sph_harm_y
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ==========================================
# 1. SPHERICAL HARMONICS POSITION ENCODING (L=40)
# ==========================================


class SphericalHarmonicsPE(nn.Module):
    """Computes real spherical harmonics up to degree L_max for input (lat, lon) coordinates."""

    def __init__(self, l_max: int = 40):
        super().__init__()
        self.l_max = l_max
        self.out_dim = (l_max + 1) ** 2

    def _latlon_to_spherical(
        self, lat_deg: torch.Tensor, lon_deg: torch.Tensor
    ):
        lat_rad = torch.deg2rad(lat_deg)
        lon_rad = torch.deg2rad(lon_deg)

        theta = torch.pi / 2.0 - lat_rad  # colatitude
        phi = torch.remainder(lon_rad + 2 * torch.pi, 2 * torch.pi)  # azimuth
        return theta, phi

    def forward(self, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        theta, phi = self._latlon_to_spherical(lat, lon)

        theta_np = theta.detach().cpu().numpy()
        phi_np = phi.detach().cpu().numpy()

        sh_components = []
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                y_lm = sph_harm_y(abs(m), l, phi_np, theta_np)

                if m < 0:
                    r_sh = math.sqrt(2) * ((-1) ** m) * y_lm.imag
                elif m == 0:
                    r_sh = y_lm.real
                else:
                    r_sh = math.sqrt(2) * ((-1) ** m) * y_lm.real

                sh_components.append(r_sh)

        sh_numpy = np.column_stack(sh_components).astype(np.float32)
        return torch.from_numpy(sh_numpy).to(lat.device)


# ==========================================
# 2. LOSS FUNCTIONS (MSE + SIGREG)
# ==========================================


class SIGREGLoss(nn.Module):
    """Sigmoid Regularization Loss (SIGREG) to enforce uniform activation spread

    and prevent variance collapse in bottleneck latent representations.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Normalize activations via Sigmoid to [0, 1]
        sig_z = torch.sigmoid(z)

        # Variance across batch dimension
        var = torch.var(sig_z, dim=0) + self.eps

        # Penalize low variance across features (drives features to stay informative)
        std_loss = torch.mean(F.relu(1.0 - torch.sqrt(var)))

        # Sigmoid distribution centering loss (prevents saturation at 0 or 1)
        mean_loss = torch.mean((torch.mean(sig_z, dim=0) - 0.5) ** 2)

        return std_loss + mean_loss


# ==========================================
# 3. ENCODER-DECODER JEPA ARCHITECTURE
# ==========================================


class StrongEncoder(nn.Module):
    """Deep residual MLP encoder compressing high-dim SH PE (1681) to bottleneck (64)."""

    def __init__(
        self, in_dim: int = 1681, hidden_dim: int = 1024, latent_dim: int = 64
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
        )

        self.latent_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.block1(x)  # Residual connection
        x = self.block2(x)
        z = self.latent_proj(x)
        return z


class WeakDecoder(nn.Module):
    """Lightweight single-layer projection from bottleneck (64) to target DINO dim (768)."""

    def __init__(self, latent_dim: int = 64, patch_embed_dim: int = 768):
        super().__init__()
        self.decoder = nn.Linear(latent_dim, patch_embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class LocationConditionedJEPA(nn.Module):

    def __init__(
        self,
        sh_lmax: int = 40,
        latent_dim: int = 64,
        patch_embed_dim: int = 768,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        # Frozen DINOv2 backbone target
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14"
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.sh_encoder = SphericalHarmonicsPE(l_max=sh_lmax)
        sh_dim = self.sh_encoder.out_dim

        # Strong Encoder (1681 -> 64)
        self.encoder = StrongEncoder(
            in_dim=sh_dim, hidden_dim=hidden_dim, latent_dim=latent_dim
        )

        # Weak Decoder (64 -> 768)
        self.decoder = WeakDecoder(
            latent_dim=latent_dim, patch_embed_dim=patch_embed_dim
        )

    def encode_location(
        self, lat: torch.Tensor, lon: torch.Tensor
    ) -> torch.Tensor:
        sh_pe = self.sh_encoder(lat, lon)
        z = self.encoder(sh_pe)
        return z

    def forward(
        self,
        images: torch.Tensor,
        patch_lats: torch.Tensor,
        patch_lons: torch.Tensor,
    ):
        B, N_patches = patch_lats.shape[0], patch_lats.shape[1]

        # Extract frozen DINO patch tokens
        with torch.no_grad():
            features = self.backbone.forward_features(images)
            target_patch_embs = features["x_norm_patchtokens"]

        flat_lats = patch_lats.view(-1)
        flat_lons = patch_lons.view(-1)

        # 1. Compute SH PE -> (B * N, 1681)
        sh_embeds = self.sh_encoder(flat_lats, flat_lons)

        # 2. Strong Encoder -> Bottleneck (B * N, 64)
        z_latent = self.encoder(sh_embeds)

        # 3. Weak Decoder -> Reconstruction (B * N, 768)
        pred_patch_embs = self.decoder(z_latent)
        pred_patch_embs = pred_patch_embs.view(B, N_patches, -1)

        return pred_patch_embs, target_patch_embs, z_latent


# ==========================================
# 4. DATASET & PATCH CENTROID GENERATOR
# ==========================================


class TilePatchDataset(Dataset):

    def __init__(
        self,
        csv_file: str,
        tiles_dir: str,
        patch_size: int = 14,
        img_size: int = 224,
    ):
        self.df = pd.read_csv(csv_file)
        self.tiles_dir = tiles_dir
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_side = img_size // patch_size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.tiles_dir, row["filename"])

        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image)

        tile_lat = row["lat"]
        tile_lon = row["lon"]

        patch_lats, patch_lons = [], []
        half_grid = self.num_patches_per_side / 2.0
        lat_step = 0.001
        lon_step = 0.001 / max(math.cos(math.radians(tile_lat)), 1e-5)

        for i in range(self.num_patches_per_side):
            for j in range(self.num_patches_per_side):
                py = (i + 0.5) - half_grid
                px = (j + 0.5) - half_grid

                p_lat = tile_lat - (py * lat_step)
                p_lon = tile_lon + (px * lon_step)

                patch_lats.append(p_lat)
                patch_lons.append(p_lon)

        return (
            img_tensor,
            torch.tensor(patch_lats, dtype=torch.float32),
            torch.tensor(patch_lons, dtype=torch.float32),
        )


# ==========================================
# 5. TRAINING PIPELINE
# ==========================================


def train_jepa_pipeline(
    csv_file: str,
    tiles_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-4,
    sigreg_weight: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    dataset = TilePatchDataset(csv_file=csv_file, tiles_dir=tiles_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model = LocationConditionedJEPA(
        sh_lmax=40, latent_dim=64, patch_embed_dim=768
    ).to(device)

    # Optimize both Encoder and Decoder parameters
    trainable_params = list(model.encoder.parameters()) + list(
        model.decoder.parameters()
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)

    sigreg_loss_fn = SIGREGLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for step, (images, patch_lats, patch_lons) in enumerate(dataloader):
            images, patch_lats, patch_lons = (
                images.to(device),
                patch_lats.to(device),
                patch_lons.to(device),
            )

            optimizer.zero_grad()

            pred_embs, target_embs, z_latent = model(
                images, patch_lats, patch_lons
            )

            # 1. MSE Loss on reconstructed patch embeddings
            mse_loss = F.mse_loss(pred_embs, target_embs)

            # 2. SIGREG Loss on bottleneck latent representations (64-dim)
            sigreg_loss = sigreg_loss_fn(z_latent)

            # Total Loss
            loss = mse_loss + (sigreg_weight * sigreg_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(dataloader)}] | "
                    f"MSE: {mse_loss.item():.5f} | SIGREG: {sigreg_loss.item():.5f} | Total: {loss.item():.5f}"
                )

    torch.save(model.state_dict(), "jepa_sh40_encoder_decoder.pth")


# ==========================================
# 6. INFERENCE FUNCTION
# ==========================================


@torch.no_grad()
def infer(
    lat: Union[float, list, np.ndarray, torch.Tensor],
    lon: Union[float, list, np.ndarray, torch.Tensor],
    model: Optional[LocationConditionedJEPA] = None,
    weights_path: Optional[str] = "jepa_sh40_encoder_decoder.pth",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts 64-dim location latent vectors z and reconstructed 768-dim DINO embeddings."""
    if not isinstance(lat, torch.Tensor):
        lat = torch.tensor(
            [lat] if isinstance(lat, (int, float)) else lat,
            dtype=torch.float32,
        )
    if not isinstance(lon, torch.Tensor):
        lon = torch.tensor(
            [lon] if isinstance(lon, (int, float)) else lon,
            dtype=torch.float32,
        )

    lat = lat.view(-1).to(device)
    lon = lon.view(-1).to(device)

    if model is None:
        model = LocationConditionedJEPA(
            sh_lmax=40, latent_dim=64, patch_embed_dim=768
        ).to(device)
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()

    # Get 64-dim bottleneck latent embedding
    z_latent = model.encode_location(lat, lon)

    # Get reconstructed 768-dim DINO embedding
    pred_dino_embs = model.decoder(z_latent)

    return z_latent, pred_dino_embs


if __name__ == "__main__":
    TILES_DIR = r"E:\Data\satclip\world_rgb\tiles"
    CSV_FILE = r"E:\Data\satclip\world_rgb\tiles_META.CSV"

    # Example 1: Train the pipeline
    train_jepa_pipeline(csv_file=CSV_FILE, tiles_dir=TILES_DIR, epochs=100, batch_size=2048, lr=3e-4)

    # # Example 2: Infer embeddings for query coordinates
    # test_lats = [34.0522, 40.7128, 51.5074]  # LA, NYC, London
    # test_lons = [-118.2437, -74.0060, -0.1278]

    # sh_feats, dino_feats = infer(
    #     lat=test_lats,
    #     lon=test_lons,
    #     weights_path="jepa_sh40_reconstructor.pth",
    # )

    # print("\nInference Output Summary:")
    # print(f"Spherical Harmonics Shape: {sh_feats.shape}")  # (3, 1681)
    # print(f"Predicted DINO Embedding Shape: {dino_feats.shape}")  # (3, 768)
