import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


import torch
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        cache_labels=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

    def _multi_positive_loss(self, logits, pos_mask):
        """
        logits:   [N, M]
        pos_mask: [N, M] (multi-hot)
        """

        logp = F.log_softmax(logits, dim=1)

        # only keep rows with at least one positive
        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logp = logp[valid]
        pos_mask = pos_mask[valid]

        loss = -(logp * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)

        return loss.mean()

    def forward(
        self,
        logits_per_image,
        logits_per_coord,
        pos_mask=None,
        output_dict=False,
    ):
        """
        logits_per_image: [N, N]
        logits_per_coord: [N, N]
        pos_mask:         [N, N] multi-hot
        """

        if pos_mask is None:
            # fallback to original behavior
            device = logits_per_image.device
            labels = torch.arange(logits_per_image.shape[0], device=device)

            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_coord, labels)
            ) / 2

        else:
            # ---- multi-positive contrastive ----
            loss_i2c = self._multi_positive_loss(logits_per_image, pos_mask)
            loss_c2i = self._multi_positive_loss(logits_per_coord, pos_mask.t())

            total_loss = (loss_i2c + loss_c2i) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class RelationalLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_feats, coord_feats, output_dict=False):
        # 1. L2 Normalize features to ensure dot product = cosine similarity
        image_feats = F.normalize(image_feats, p=2, dim=-1)
        coord_feats = F.normalize(coord_feats, p=2, dim=-1)

        # 2. Compute self-similarity matrices (Gram Matrices)
        # Shape: [Batch, Batch]
        sim_i = torch.matmul(image_feats, image_feats.t()) / self.temperature
        sim_c = torch.matmul(coord_feats, coord_feats.t()) / self.temperature

        # 3. Optional: Apply softmax if you want to match probability distributions 
        # instead of raw scores. For standard MLD, raw MSE on similarities is common.
        # sim_i = F.softmax(sim_i, dim=-1)
        # sim_c = F.softmax(sim_c, dim=-1)

        # 4. Compute MSE loss between the two topologies
        relational_loss = F.mse_loss(sim_i, sim_c)

        return {"relational_loss": relational_loss} if output_dict else relational_loss

class ReconstructionLoss(nn.Module):
    def __init__(self):
        """
        Args:
            feat_dim (int): The dimensionality of the image_feats (target).
            latent_dim (int): The dimensionality of point_feats. If None, assumes same as feat_dim.
            use_projection (bool): Whether to learn a linear mapping from point to image space.
        """
        super().__init__()

    def forward(self, image_feats, reconstructed_feats, output_dict=False):
        """
        reconstructs image_feats from point_feats
        """
        recon_loss = F.mse_loss(reconstructed_feats, image_feats)

        return {"recon_loss": recon_loss} if output_dict else recon_loss


class GramLoss(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def gram_matrix(self, x):
        """
        x: [B, D]
        Returns: [D, D] Gram matrix (feature correlations)
        """
        # Optionally normalize embeddings first
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        # Compute Gram
        G = x.t() @ x  # [D, D]

        # Normalize by batch size to stabilize scale
        G = G / x.size(0)

        return G

    def forward(self, student_feats, teacher_feats, output_dict=False):
        """
        student_feats: [B, D]
        teacher_feats: [B, D]
        """

        G_s = self.gram_matrix(student_feats)
        G_t = self.gram_matrix(teacher_feats)

        loss = F.mse_loss(G_s, G_t)

        return {"gram_loss": loss} if output_dict else loss


# class ReconstructionLoss(nn.Module):
#     def __init__(self, loss_type='combined', alpha=1.0, beta=0.5):
#         """
#         Args:
#             loss_type (str): 'mse', 'l1', or 'combined'
#             alpha (float): Weight for the primary loss (MSE/L1)
#             beta (float): Weight for additional penalty (e.g., L1 when combined)
#         """
#         super(ReconstructionLoss, self).__init__()
#         self.loss_type = loss_type.lower()
#         self.alpha = alpha
#         self.beta = beta
        
#         self.mse = nn.MSELoss()
#         self.l1 = nn.L1Loss()

#     def forward(self, pred, target):
#         """
#         Calculates the error between the reconstructed output and original input.
#         """
#         if self.loss_type == 'mse':
#             loss = self.mse(pred, target)
        
#         elif self.loss_type == 'l1':
#             loss = self.l1(pred, target)
            
#         elif self.loss_type == 'combined':
#             # Smooth L1 is often more robust to outliers than pure MSE
#             loss = self.alpha * self.mse(pred, target) + self.beta * self.l1(pred, target)
            
#         else:
#             raise ValueError(f"Unknown loss_type: {self.loss_type}")

#         return loss

class SilhouetteLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, coord_feats, pos_mask):
        """
        embeddings: [N, D] (normalized preferred)
        pos_mask:   [N, N] (multi-hot, includes self or not—your choice)
        """
        coord_feats = F.normalize(coord_feats, p=2, dim=-1)

        # cosine distance = 1 - similarity
        sim = coord_feats @ coord_feats.t()          # [N, N]
        dist = 1 - sim                             # convert to distance

        # ---- a(i): intra-cluster distance ----
        pos_sum = (dist * pos_mask).sum(dim=1)
        pos_count = pos_mask.sum(dim=1).clamp_min(1)

        a = pos_sum / pos_count                    # [N]

        # ---- b(i): nearest "other cluster" ----
        neg_mask =  (~pos_mask) if pos_mask.dtype == torch.bool else (1 - pos_mask)

        # average distance to negatives
        neg_sum = (dist * neg_mask).sum(dim=1)
        neg_count = neg_mask.sum(dim=1).clamp_min(1)

        b = neg_sum / neg_count                    # [N]

        # ---- silhouette score ----
        denom = torch.maximum(a, b).clamp_min(self.eps)
        s = (b - a) / denom                        # [N]

        # we want to maximize → minimize negative
        loss = -s.mean()

        return loss


class MarginLoss(nn.Module):
    def __init__(self, margin = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, image_feats, point_feats, pos_mask):
        sim = image_feats @ point_feats.t()  # cosine

        pos = (sim * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)

        neg = sim.masked_fill(pos_mask.bool(), -1e9).max(dim=1).values

        return F.relu(self.margin - pos + neg).mean()

    
