"""
config.py

Declare what to evaluate here, then run:  python run_eval.py

DATASETS
--------
DatasetSpec(name, path, task, pred, target)
  - task:   "reg" or "class"
  - pred:   (start_col, end_col) -- inclusive, 0-indexed column range that
            contains your predictor block. Must include a lat column and a
            lon column somewhere in that range (any of the common name
            variants -- lat/latitude/lon/lng/longitude/x/y -- are
            auto-detected). Other columns in the range are ignored; the
            model only ever sees lon/lat -> embedding.
  - target: (start_col, end_col) -- inclusive, 0-indexed column range. Every
            column in this range becomes its own row in the report, all
            evaluated with the same `task`.

If a single CSV mixes regression and classification targets, just declare
it twice with two different `target` ranges (and two `task` values) -- see
"housing_class" below for the pattern.

MODELS
------
Any SatCLIP checkpoint via make_satclip_loader(name, ckpt_path,
satclip_repo_path). Point `satclip_repo_path` at your local clone (the one
containing satclip/load.py) -- e.g. "./satclip" or an absolute path.

To compare against anything else (your Global Location Encoder, a raw
lat/lon baseline, etc.), use make_custom_loader(name, embed_fn) with any
function (lon: np.ndarray, lat: np.ndarray) -> np.ndarray of shape (N, D).
"""
import numpy as np
from eval_embeddings import DatasetSpec
from models import make_custom_loader, make_satclip_loader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


SATCLIP_REPO_PATH = rf"D:\Code\satclip\satclip"  # path to your cloned+modified satclip repo

DATASETS = [
    # columns: [0]=id, [1]=state, [2]=county, [3..10]=lat/lon + other predictors,
    # [11..23]=regression targets (median income, home value, etc.)
    # DatasetSpec(name="housing", path="data/housing.csv", task="reg", pred=(3, 10), target=(11, 23)),
    # example of a classification target range from a different CSV
    # DatasetSpec(name="landcover", path="data/landcover.csv", task="class", pred=(0, 2), target=(3, 3)),
    # DatasetSpec(name="Global AQI", path=rf"E:\Data\satclip\eval\aqi.csv", task="reg", pred=(0, 1), target=(2, 11)),
    # DatasetSpec(name="Afrobarometer", path=rf"E:\Data\satclip\eval\afrobarometer_location_targets.csv", task="class", pred=(0, 1), target=(2, 24)),
    DatasetSpec(name="Airbnb Price", path=rf"E:\Data\satclip\eval\airbnb-price.csv", task="reg", pred=(1, 18), target=(0, 0)),
    DatasetSpec(name="Crop Yeild", path=rf"E:\Data\satclip\eval\crop-yeild.csv", task="reg", pred=(0, 1), target=(2, 43)),
    DatasetSpec(name="EarthQuake", path=rf"E:\Data\satclip\eval\earthquake.csv", task="reg", pred=(0, 1), target=(2, 3)),
    DatasetSpec(name="EV Charging", path=rf"E:\Data\satclip\eval\ev-charging-cost.csv", task="reg", pred=(0, 1), target=(2, 2)),
    DatasetSpec(name="FEMA", path=rf"E:\Data\satclip\eval\fema.csv", task="reg", pred=(0, 21), target=(22, 23)),
    DatasetSpec(name="Wealth Index", path=rf"E:\Data\satclip\eval\relative-wealth-index.csv", task="reg", pred=(1, 2), target=(3, 3)),
    DatasetSpec(name="USA Health", path=rf"E:\Data\satclip\eval\usa-cdchealth.csv", task="reg", pred=(41, 42), target=(0, 40)),
    DatasetSpec(name="USA Climate", path=rf"E:\Data\satclip\eval\usa-climate.csv", task="reg", pred=(36, 37), target=(0, 35)),

]

MODELS = [
    make_custom_loader(
            name="random-embeddings",
            embed_fn = lambda lon, lat, dim=64: np.random.default_rng(42).standard_normal((len(lon), dim), dtype=np.float32),
        ),
    make_custom_loader(
            name="raw-coords",
            embed_fn=lambda lon, lat: __import__("numpy").stack([lon, lat], axis=1),
        ),
    # make_custom_loader(
    #         name="spherical-harmonics",
    #         embed_fn=lambda lon, lat: __import__("numpy").stack([lon, lat], axis=1),
    #     ),
    # make_custom_loader(
    #         name="microsoft-satclip",
    #         embed_fn=lambda lon, lat: __import__("numpy").stack([lon, lat], axis=1),
    #         satclip_repo_path=SATCLIP_REPO_PATH,
    #     ),
    # make_custom_loader(
    #         name="satclip-v1",
    #         embed_fn=lambda lon, lat: __import__("numpy").stack([lon, lat], axis=1),
    #         satclip_repo_path=SATCLIP_REPO_PATH,
    #     ),
    make_satclip_loader(
            name="satclip-v2-64dim",
            ckpt_path=rf"E:\Weights\satclip\satclip-v2\satclipv2-64dim.ckpt",
            satclip_repo_path=SATCLIP_REPO_PATH,
        ),
    # add more checkpoints to compare, e.g. a larger backbone:
    # make_satclip_loader(
    #     name="satclip-resnet50-l40",
    #     ckpt_path="satclip-resnet50-l40.ckpt",
    #     satclip_repo_path=SATCLIP_REPO_PATH,
    # ),
    # a raw lat/lon baseline, for reference:
    # make_custom_loader(
    #     name="raw-coords",
    #     embed_fn=lambda lon, lat: __import__("numpy").stack([lon, lat], axis=1),
    # ),
    # wire in your own encoder (e.g. Global Location Encoder) the same way:
    # make_custom_loader(
    #     name="global-location-encoder",
    #     embed_fn=lambda lon, lat: my_gle_model.embed(lon, lat),
    # ),
]