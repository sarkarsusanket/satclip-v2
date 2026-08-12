import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SATCLIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "satclip")
if SATCLIP_DIR not in sys.path:
    sys.path.insert(0, SATCLIP_DIR)

from location_encoder import get_neural_network, get_positional_encoding, LocationEncoder

torch.set_float32_matmul_precision("high")

LAT_NAMES = {"lat", "latitude", "y", "lat_wgs84"}
LON_NAMES = {"lon", "lng", "long", "longitude", "x", "lon_wgs84"}
NAME_NAMES = {"name", "id", "filename", "fn"}
N_SAMPLES = 100_000
VERBOSE = 0


def _find_column(df, aliases):
    lower = {c.lower(): c for c in df.columns}
    return next((lower[a] for a in aliases if a in lower), None)


# ---------------------------------------------------------------- model ----
class ProjLayer(nn.Module):
    def __init__(self, width, output_dim):
        super().__init__()
        scale = width ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        return x @ self.proj


class SatCLIP(nn.Module):
    def __init__(
        self,
        embed_dim=64,
        le_type="sphericalharmonics",
        pe_type="siren",
        frequency_num=16,
        max_radius=0.01,
        min_radius=0.00001,
        legendre_polys=40,
        harmonics_calculation="analytic",
        num_hidden_layers=2,
        capacity=256,
    ):
        super().__init__()
        self.dino_proj = ProjLayer(384, embed_dim)  # Dino v2 is used
        self.posenc = get_positional_encoding(
            name=le_type,
            harmonics_calculation=harmonics_calculation,
            legendre_polys=legendre_polys,
            min_radius=min_radius,
            max_radius=max_radius,
            frequency_num=frequency_num,
        ).double()
        self.nnet = get_neural_network(
            name=pe_type,
            input_dim=self.posenc.embedding_dim,
            num_classes=embed_dim,
            dim_hidden=capacity,
            num_layers=num_hidden_layers,
        ).double()
        self.location = LocationEncoder(self.posenc, self.nnet).double()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.5))

    def encode_image(self, dino_image):
        return self.dino_proj(dino_image)

    def encode_location(self, coords):
        return self.location(coords.double())

    def forward(self, dino_image, coords):
        image_features = self.encode_image(dino_image)
        location_features = self.encode_location(coords).float()

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features = location_features / location_features.norm(dim=1, keepdim=True)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=20)
        logits_per_image = logit_scale * image_features @ location_features.t()
        logits_per_location = logits_per_image.t()
        return image_features, location_features, logits_per_image, logits_per_location, logit_scale


# ---------------------------------------------------------------- losses ---
class ContrastiveLoss(nn.Module):
    def _multi_positive_loss(self, logits, pos_mask):
        logp = F.log_softmax(logits, dim=1)
        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        logp = logp[valid]
        pos_mask = pos_mask[valid]
        return -(logp * pos_mask).sum(dim=1).div(pos_mask.sum(dim=1)).mean()

    def forward(self, logits_per_image, logits_per_coord, pos_mask):
        loss_i2c = self._multi_positive_loss(logits_per_image, pos_mask)
        loss_c2i = self._multi_positive_loss(logits_per_coord, pos_mask.t())
        return (loss_i2c + loss_c2i) / 2


class RelationalLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_feats, coord_feats):
        image_feats = F.normalize(image_feats, p=2, dim=-1)
        coord_feats = F.normalize(coord_feats, p=2, dim=-1)
        sim_i = image_feats @ image_feats.t() / self.temperature
        sim_c = coord_feats @ coord_feats.t() / self.temperature
        return F.mse_loss(sim_i, sim_c)


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


# ---------------------------------------------------------------- dataset --
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
        df = df.sample(N_SAMPLES)
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


# ---------------------------------------------------------------- train ----
def _get_quantile(dist, device):
    flat = dist.float().reshape(-1)
    max_samples = 2_000_000
    if flat.numel() > max_samples:
        idx = torch.randint(0, flat.numel(), (max_samples,), device=device)
        flat = flat[idx]
    q = torch.quantile(flat.cpu(), torch.tensor([0.01])).to(device)
    return q[0].item()


def build_pos_mask(dino_feats):
    rgb_dist = torch.cdist(dino_feats, dino_feats)
    pos_mask = rgb_dist < _get_quantile(rgb_dist, dino_feats.device)
    pos_mask.fill_diagonal_(1.0)
    return pos_mask


def common_step(model, losses, batch):
    dino_image = batch["dino"]
    t_points = batch["point"].float()
    pos_mask = build_pos_mask(dino_image)

    image_feats, point_feats, logits_per_image, logits_per_coord, logit_scale = model(dino_image, t_points)

    contra = losses["contra"](logits_per_image, logits_per_coord, pos_mask)
    relat = losses["relat"](image_feats, point_feats)
    silh = losses["silh"](point_feats, pos_mask)
    logit_loss = logit_scale.mean()
    return contra + 0.3 * relat + 0.3 * silh + 0.01 * logit_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatCLIP(
        embed_dim=args.embed_dim,
        le_type=args.le_type,
        pe_type=args.pe_type,
        frequency_num=args.frequency_num,
        max_radius=args.max_radius,
        min_radius=args.min_radius,
        legendre_polys=args.legendre_polys,
        harmonics_calculation=args.harmonics_calculation,
        num_hidden_layers=args.num_hidden_layers,
        capacity=args.capacity,
    ).to(device)

    losses = {
        "contra": ContrastiveLoss(),
        "relat": RelationalLoss(temperature=0.5),
        "silh": SilhouetteLoss(),
    }

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
    optimizer = torch.optim.AdamW(
        [{"params": gain_or_bias, "weight_decay": 0.0}, {"params": rest, "weight_decay": args.weight_decay}],
        lr=args.lr,
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
            loss = common_step(model, losses, batch)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n += 1
        train_loss = running / max(n, 1)
        val_loss = evaluate()
        if VERBOSE:
            print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        elif epoch%10==0:
            print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            early_stop = args.patience
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


# ---------------------------------------------------------------- eval ----
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

    for target in targets:
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
    parser.add_argument("--csv", default=rf"E:\Data\satclip\data\index.csv", help="index CSV with name/lat/lon columns")
    parser.add_argument("--data_dir", default=rf"E:\Data\satclip\data", help="folder containing the dino .npy files")
    parser.add_argument("--npy_subdir", default=rf"npy_features", help="optional subdir under data_dir for npy files")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience (0 to disable)")
    parser.add_argument("--batch_size", type=int, default=20480)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--le_type", default="sphericalharmonics")
    parser.add_argument("--pe_type", default="siren")
    parser.add_argument("--frequency_num", type=int, default=16)
    parser.add_argument("--max_radius", type=float, default=0.01)
    parser.add_argument("--min_radius", type=float, default=0.00001)
    parser.add_argument("--legendre_polys", type=int, default=40)
    parser.add_argument("--harmonics_calculation", default="analytic")
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--eval_csv", default=rf"E:\Data\satclip\eval\usa-cdchealth.csv", help="CSV for LightGBM 5-fold eval (optional)")
    parser.add_argument("--eval_target", default=None, help="target column name in eval CSV (default: all except lat/lon)")
    parser.add_argument("--task", choices=["regression", "classification"], default=None, help="eval task (default: auto)")
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy("file_system")
    train(args)


if __name__ == "__main__":
    main()
