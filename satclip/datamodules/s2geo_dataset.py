from typing import Any, Callable, Dict, Optional

import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


import pandas as pd
import rasterio
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset
import matplotlib.pyplot as plt
import numpy as np
import torch

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .transforms import get_pretrained_s2_train_transform, get_s2_train_transform

CHECK_MIN_FILESIZE = 10000 # 10kb

class S2GeoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = r"/data/susanket/sentinel",
        batch_size: int = 20480,
        num_workers: int = 92,
        crop_size: int = 256,
        val_random_split_fraction: float = 0.1,
        transform: str = 'pretrained',
        mode: str = "both",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if transform=='pretrained':
            self.train_transform = get_pretrained_s2_train_transform(resize_crop_size=crop_size)
        elif transform=='default':
            self.train_transform = get_s2_train_transform()
        else:
            self.train_transform = transform
            
        self.val_random_split_fraction = val_random_split_fraction
        self.mode = mode
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            print("""
            No dataset found. To download, please follow instructions on: https://github.com/microsoft/satclip
            """)

    def setup(self, stage="fit"):
        dataset = S2Geo(root=self.data_dir, transform=self.train_transform, mode=self.mode)

        N_val = int(len(dataset) * self.val_random_split_fraction)
        N_train = len(dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [N_train, N_val])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            #persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        raise NotImplementedError

class S2Geo(NonGeoDataset):
    """S2-100K dataset.

    This dataset contains 100,000 256x256 patches of 12 band Sentinel imagery sampled randomly
    from Sentinel 2 scenes on the Microsoft Planetary Computer that have <20% cloud cover,
    intersect land, and were captured between 2021-01-01 and 2023-05-17 (there are 2,359,972
    such scenes).
    """

    validation_filenames = [
    ]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        mode: Optional[str] = "both",
    ) -> None:
        """Initialize a new S2-100K dataset instance.
        Args:
            root: root directory of S2-100K pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: which data to return (options are "both" or "points"), useful for embedding locations without loading images 
        """
        assert mode in ["both", "points"]
        self.root = root
        self.transform = transform
        self.mode = mode
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        index_fn = r"/data/susanket/sentinel/dinov2-1M/index_600k.csv"

        df = pd.read_csv(index_fn)
        self.filenames = []
        self.points = []

        n_skipped_files = 0
        for i in tqdm.tqdm(range(df.shape[0])):
            # orig_image = os.path.join(r"/data/susanket/sentinel", "images1M", df.iloc[i]["name"]+".npy")
            dino_file = os.path.join(self.root, "dinov2-1M/cls", df.iloc[i]["name"]+".npy")
            # terramind_file = os.path.join(self.root, "terramind/S2L2A", df.iloc[i]["fn"].replace("npy", "tif.npy"))

            # print(dino_file, terramind_file)
            if not os.path.exists(dino_file):
                n_skipped_files += 1
                continue

            self.filenames.append(dino_file)
            self.points.append(
                (df.iloc[i]["lat"], df.iloc[i]["lon"])
            )

        print(f"skipped {n_skipped_files}/{len(df)} images because they were smaller "
              f"than {CHECK_MIN_FILESIZE} bytes... they probably contained nodata pixels")

    def validate_data(self, file_path, threshold=0.75):
        """
        Checks if the file exists, is not empty, and has > threshold valid data.
        """
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return False
        
        try:
            # Load the image (assuming .npy format based on previous context)
            img = np.load(file_path)
            rgb_img = img[..., :3][..., ::-1].astype(np.float32)
            # True (1) means data exists, False (0) means it's empty/black
            mask = (rgb_img.max(axis=-1) > 0)

            # 2. Calculate the percentage
            # Since True=1 and False=0, the mean gives the percentage in decimal form
            percent_data = np.mean(mask)

            # print(f"Area with data: {percent_data:.2f}%")
            
            return percent_data > threshold
        except Exception:
            return False

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            dictionary with "image" and "point" keys where point is in (lon, lat) format
        """
        point = torch.tensor(self.points[index])
        sample = {"point": point}

        # Open the dino file
        image_path = self.filenames[index]
        image = np.load(image_path).astype(np.float32)
        sample["dino"] = image.squeeze(0)
        
        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.
        Returns:
            length of dataset
        """
        return len(self.filenames)

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.
        Returns:
            True if the dataset directories and split files are found, else False
        """
        
        for filename in self.validation_filenames:
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                print(filepath +' missing' )
                return False
        return True

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.
        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        ncols = 1

        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        ax.imshow(image[:, :, [3,2,1]] / 4000)
        ax.axis("off")

        if show_titles:
            ax.set_title(f"({sample['point'][0]:0.4f}, {sample['point'][1]:0.4f})")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig