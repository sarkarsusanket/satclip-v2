from datetime import datetime
from pathlib import Path

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


import lightning.pytorch
import torch
from datamodules.s2geo_dataset import S2GeoDataModule
from lightning.pytorch.cli import LightningCLI
from loss import ContrastiveLoss, RelationalLoss, ReconstructionLoss, SilhouetteLoss, MarginLoss
from model import SatCLIP

torch.set_float32_matmul_precision('high')

class SatCLIPLightningModule(lightning.pytorch.LightningModule):
    def __init__(
        self,
        embed_dim=64,
        image_resolution=256,
        vision_layers=12,
        vision_width=1024,
        vision_patch_size=32,
        in_channels=4,
        le_type="sphericalharmonics",
        pe_type="siren",
        frequency_num=16,
        max_radius=260,
        min_radius=1,
        legendre_polys=40,
        harmonics_calculation="analytic",
        sh_embedding_dims=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_hidden_layers=8,
        capacity=512,
    ) -> None:
        super().__init__()

        self.model = SatCLIP(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            le_type=le_type,
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            sh_embedding_dims=sh_embedding_dims,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
        )

        self.contraloss_fn = ContrastiveLoss()
        self.relatloss_fn = RelationalLoss(temperature=0.5)
        self.silhouette_fn = SilhouetteLoss()
        self.marginloss_fn = MarginLoss()
        self.reconloss_fn = ReconstructionLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def _get_quantile(self, dist):
        flat = dist.float().reshape(-1)
        q = torch.tensor([0.01], device=flat.device)

        max_samples = 2_000_000
        if flat.numel() > max_samples:
            idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device)
            flat = flat[idx]

        # compute on CPU to further reduce GPU pressure (optional)
        quantiles = torch.quantile(flat.cpu(), q.cpu()).to(dist.device)

        return quantiles[0].item()


    def build_pos_mask(self, dino_feats, points):
        # 1. Spatial Distance: Are they physically close?
        # points: [N, D] -> spatial_dist: [N, N]
        # spatial_dist = torch.cdist(points, points)
        # spatial_mask = spatial_dist < self._get_quantile(spatial_dist)

        # 2. Visual Distance: Do they look similar? Right nw only keeping visual similar
        rgb_dist = torch.cdist(dino_feats, dino_feats)
        visual_mask = rgb_dist < self._get_quantile(rgb_dist)

        # 3. Combine: A pair is positive only if it satisfies BOTH criteria
        # This prevents grouping two similar-looking objects that are far apart
        pos_mask = visual_mask#(spatial_mask & visual_mask).float()

        # Self-positive (diagonal)
        pos_mask.fill_diagonal_(1.0)

        return pos_mask

    def common_step(self, batch):
        dino_image = batch['dino']
        t_points = batch["point"].float()
        pos_mask = self.build_pos_mask(dino_image, t_points)

        # Forward Pass
        image_feats, point_feats, logits_per_image, logits_per_coord, logit_scale = self.model(dino_image, t_points)

        contraloss = self.contraloss_fn(logits_per_image, logits_per_coord, pos_mask)
        relatloss = self.relatloss_fn(image_feats, point_feats)
        silhloss = self.silhouette_fn(point_feats, pos_mask)
        # reconloss = self.reconloss_fn(dino_image, reconstructed_image)
        # marginloss = self.marginloss_fn(image_feats, point_feats, pos_mask)
        logit_loss = (logit_scale).mean()

        self.log("contra_loss", contraloss)
        self.log("relational_loss", relatloss)
        self.log("silhoutte_loss", silhloss)

        # self.log("margin_loss", marginloss)
        self.log("logitscale_loss", logit_loss)
        
        self.log("logit_scale", logit_scale)

        loss = contraloss + 0.3*relatloss + 0.3*silhloss + 0.01*logit_loss # Regulatize too high logit scales

        return loss

    def training_step(self, batch):
        loss = self.common_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        loss = self.common_step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },  # specify in configs/default.yaml
            ],
            lr=self.learning_rate,  # specify in configs/default.yaml
        )

        return optimizer


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--watchmodel", action="store_true")


def cli_main(default_config_filename="./configs/default.yaml"):
    save_config_fn = default_config_filename.replace(".yaml", "-latest.yaml")
    # modify configs/default.yaml for learning rate etc.
    cli = MyLightningCLI(
        model_class=SatCLIPLightningModule,
        datamodule_class=S2GeoDataModule,
        save_config_kwargs=dict(
            config_filename=save_config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "accumulate_grad_batches": 1,
            "log_every_n_steps": 1,
            "strategy": "ddp_find_unused_parameters_true",
        },
        parser_kwargs={"default_config_files": [default_config_filename]},
        seed_everything_default=0,
        run=False,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"SatCLIP_S2_{ts}"
    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name
        # this seems to be necessary to force logging of datamodule hyperparams
        cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    # Create folder to log configs
    # NOTE: Lightning does not handle config paths with subfolders
    dirname_cfg = Path(default_config_filename).parent
    dir_log_cfg = Path(cli.trainer.log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )


if __name__ == "__main__":
    config_fn = r"/home/susanket/satclip/sentinel2/satclip/satclip/configs/default.yaml"

    torch.multiprocessing.set_sharing_strategy('file_system')

    if torch.cuda.get_device_name(device=0)=='NVIDIA A100 80GB PCIe':
        torch.set_float32_matmul_precision("highest")
        print('Model go vroom! 🚀')
    else:
        torch.set_float32_matmul_precision("high")
    cli_main(config_fn)