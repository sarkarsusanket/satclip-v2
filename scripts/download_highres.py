import io
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from shapely.geometry import Point
from torchvision import transforms


def wgs84_to_web_mercator(lon: float, lat: float) -> Tuple[float, float]:
    """Converts (lon, lat) WGS84 to EPSG:3857 (Web Mercator)."""
    lat = max(-85.0511, min(85.0511, lat))
    lon = max(-180.0, min(180.0, lon))
    x = 6378137.0 * math.radians(lon)
    y = 6378137.0 * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


class SamplerPipeline:
    def __init__(
        self,
        vector_path: str,
        image_size: int = 512,
        meters_per_pixel: float = 5.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_workers: int = 16,  # Threads for parallel image downloads
    ):
        self.image_size = image_size
        self.meters_per_pixel = meters_per_pixel
        self.device = device
        self.max_workers = max_workers

        print(f"Loading vector boundary from: {vector_path}")
        gdf = gpd.read_file(vector_path)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        self.polygon = gdf.geometry.union_all()
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.polygon.bounds

        # Thread-safe HTTP Session setup
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2
        )
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        # Load DINOv2 model
        print(f"Loading DINOv2 model on {self.device}...")
        self.dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.dino_model.eval().to(self.device)

        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def sample_random_point(self) -> Tuple[float, float]:
        """Uniformly samples a valid (lat, lon) inside the vector boundary."""
        while True:
            lon = random.uniform(self.min_lon, self.max_lon)
            lat = random.uniform(self.min_lat, self.max_lat)
            point = Point(lon, lat)
            if self.polygon.contains(point):
                return lat, lon

    def fetch_image(self, lon: float, lat: float) -> Optional[Image.Image]:
        """Fetches RGB satellite image from ArcGIS REST endpoint."""
        mx, my = wgs84_to_web_mercator(lon, lat)
        half_span = (self.image_size * self.meters_per_pixel) / 2.0

        xmin, xmax = mx - half_span, mx + half_span
        ymin, ymax = my - half_span, my + half_span
        bbox = f"{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}"

        params = {
            "bbox": bbox,
            "bboxSR": "3857",
            "size": f"{self.image_size},{self.image_size}",
            "imageSR": "3857",
            "format": "jpg",
            "f": "image",
        }
        url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"

        try:
            res = self.session.get(url, params=params, timeout=10)
            if res.status_code == 200 and res.headers.get("Content-Type", "").startswith("image"):
                return Image.open(io.BytesIO(res.content)).convert("RGB")
        except Exception:
            pass
        return None

    def _get_single_sample_task(self, sample_id: int) -> Tuple[int, float, float, Image.Image]:
        """Worker task executed across threads."""
        attempts = 0
        while attempts < 5:
            lat, lon = self.sample_random_point()
            pil_img = self.fetch_image(lon, lat)
            if pil_img is not None:
                return sample_id, lat, lon, pil_img
            attempts += 1

        # Fallback to blank image
        lat, lon = self.sample_random_point()
        return sample_id, lat, lon, Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

    @torch.no_grad()
    def process_batch(self, batch_items: List[Tuple[int, float, float, Image.Image]]) -> List[Tuple[int, float, float, np.ndarray]]:
        """Processes a batch of images through DINO on GPU simultaneously."""
        images = [item[3] for item in batch_items]
        
        # Batch transform & tensor stack
        tensors = torch.stack([self.dino_transform(img) for img in images]).to(self.device)
        
        # Batched GPU forward pass
        cls_embeddings = self.dino_model(tensors).cpu().numpy()

        results = []
        for i, (sample_id, lat, lon, _) in enumerate(batch_items):
            results.append((sample_id, lat, lon, cls_embeddings[i]))
            
        return results

    def process_and_save(
        self,
        num_samples: int,
        output_dir: str,
        csv_filename: str = "index.csv",
        batch_size: int = 64,
    ):
        """Processes dataset using parallel HTTP workers and batched GPU inference."""
        output_path = Path(output_dir)
        npy_dir = output_path / "npy_features"
        npy_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / csv_filename

        start_id = 0
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                if not existing_df.empty and "id" in existing_df.columns:
                    start_id = int(existing_df["id"].max()) + 1
                    print(f"Found checkpoint CSV. Resuming from ID: {start_id}")
            except Exception as e:
                print(f"⚠️ Failed reading CSV: {e}. Starting fresh.")

        if start_id >= num_samples:
            print(f"Target count of {num_samples} already reached. Exiting.")
            return

        print(f"🚀 Processing from ID {start_id} to {num_samples} (Threads: {self.max_workers}, Batch Size: {batch_size})...")

        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        current_id = start_id

        try:
            while current_id < num_samples:
                batch_end = min(current_id + batch_size, num_samples)
                sample_ids = list(range(current_id, batch_end))

                # Step 1: Parallel HTTP downloads
                download_results = list(executor.map(self._get_single_sample_task, sample_ids))

                # Step 2: Batched DINO GPU Inference
                processed_batch = self.process_batch(download_results)

                # Step 3: Save .npy files & append CSV chunk
                chunk_records = []
                for sample_id, lat, lon, dino_cls in processed_batch:
                    filename = f"{sample_id:07d}.npy"
                    np.save(npy_dir / filename, dino_cls)

                    chunk_records.append({
                        "lat": lat,
                        "lon": lon,
                        "filename": filename,
                        "id": sample_id,
                    })

                # Append batch to CSV immediately
                df_chunk = pd.DataFrame(chunk_records)
                file_exists = csv_path.exists()
                df_chunk.to_csv(csv_path, mode="a", index=False, header=not file_exists)

                current_id = batch_end
                print(f"Completed [{current_id}/{num_samples}] samples.")

        finally:
            executor.shutdown(wait=True)

        print(f"🎉 Fully completed {num_samples} samples!")


if __name__ == "__main__":
    VECTOR_FILE = rf"E:\Data\Global\World\land-poly\land_polygons.shp"
    OUTPUT_DIRECTORY = rf"E:\Data\satclip\data"
    NUM_SAMPLES = 100_000

    pipeline = SamplerPipeline(
        vector_path=VECTOR_FILE,
        image_size=512,
        meters_per_pixel=5.0,
        max_workers=20,  # Increase to 24 or 32 if network bandwidth permits
    )

    pipeline.process_and_save(
        num_samples=NUM_SAMPLES,
        output_dir=OUTPUT_DIRECTORY,
        csv_filename="index.csv",
        batch_size=1024,   # Fits well in GPU VRAM
    )