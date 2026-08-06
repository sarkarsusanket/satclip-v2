# eval/

Evaluate a set of location-embedding models (SatCLIP checkpoints, your own
encoders, baselines) against a set of CSV datasets, using default-params
LightGBM as the downstream probe.

For every `(dataset, target column, model)` combination:
1. Embed the dataset's lon/lat once per model (reused across all its
   targets).
2. 5-fold CV with a default-params `LGBMRegressor` (task="reg") or
   `LGBMClassifier` (task="class") — **X is the embedding only**, nothing
   else from the CSV.
3. Score: R² for regression, macro-F1 for classification.

Output is one tidy CSV: rows = `(dataset, target)`, columns = model names.

## Files

- **`config.py`** — declare everything here: which CSVs, which target
  columns, which task type, which models. This is the only file you should
  need to touch for a normal run.
- **`eval_embeddings.py`** — the pipeline (`DatasetSpec`, `run_evaluation`).
  You shouldn't need to edit this.
- **`models.py`** — `EmbeddingModel` wrapper + loaders. `make_satclip_loader`
  mirrors the SatCLIP demo notebook's `get_satclip(...)` call exactly (lon
  first, then lat). `make_custom_loader` wraps anything else.
- **`run.py`** — `python run.py [--output eval_report.csv]`.

## Declaring a dataset

```python
DatasetSpec(name="housing", path="data/housing.csv", task="reg", pred=(3, 10), target=(11, 23))
```

- `pred=(3, 10)`: inclusive, 0-indexed column range containing your
  predictor block. It must contain a lat column and a lon column somewhere
  in that range — auto-detected by name (`lat`/`latitude`/`y`,
  `lon`/`lng`/`longitude`/`x`, case-insensitive). Any other columns in the
  range are ignored; only lon/lat ever reach the model.
- `target=(11, 23)`: inclusive column range. **Every column in this range
  becomes its own row** in the report, all evaluated with the same `task`.
- `task="reg"` or `"class"`, applied to the whole `target` range.

If one CSV has both regression and classification targets, declare it
**twice** with two different `target` ranges (and two `DatasetSpec`s) —
see the `housing_class` example commented out in `config.py`.

## Declaring models

```python
make_satclip_loader(name="satclip-resnet18-l10", ckpt_path="satclip-resnet18-l10.ckpt", satclip_repo_path="./satclip")
```

`satclip_repo_path` should point at your cloned+modified SatCLIP repo (the
one containing `satclip/load.py`). Internally this does exactly what the
demo notebook does:

```python
model = get_satclip(ckpt_path, device=device)
emb = model(coords.double().to(device))  # coords is (N, 2) as (lon, lat)
```

To compare against anything else — your Global Location Encoder, a raw
lat/lon baseline, GridAndSphere/SphericalHarmonics, whatever — use:

```python
make_custom_loader(name="my-model", embed_fn=lambda lon, lat: my_model.embed(lon, lat))
```

`embed_fn` takes 1-D numpy arrays `(lon, lat)` and returns an `(N, D)`
numpy array, same row order as input. That's the entire contract — it can
wrap a checkpoint, a formula, or something already loaded in memory.

## Running

```bash
cd eval
python run_eval.py --output eval_report.csv
```

Output columns: `dataset, target, task_type, n_samples, <model_1>, <model_2>, ...`
Each model column holds R² (reg rows) or macro-F1 (class rows). The report
is rewritten incrementally after every target, so a long run can be
interrupted without losing prior results.

## Notes / things you may want to tweak

- **Class imbalance**: a classification target needs at least 5 samples in
  its smallest class (for 5-fold stratified CV) or that combination is
  skipped with a logged error, not a crash — one bad target/model pair
  never kills the whole run.
- **Missing values**: rows with a missing target are dropped per-target
  (not per-dataset), so different targets from the same CSV can use
  different row counts.
- **LightGBM params**: intentionally left at defaults per your ask
  (`LGBMRegressor()` / `LGBMClassifier()`, only `verbose=-1`). If you want
  to sweep params later, that's a small, contained change in
  `_score_regression`/`_score_classification` in `eval_embeddings.py`.