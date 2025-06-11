### I. Data collection - 0_lakeapi.ipynb

This script pulls raw order-book snapshots using **`lakeapi`** rather than native exchange endpoints.

* **Why `lakeapi`?**

  * It is purpose-built for *historical* data without running a live recorder.
  * It provides full-depth LOB at multiple granularities.
  * It is **orders of magnitude cheaper** than most exchange “data shop” APIs.

* **Caveat:** the service returns an *HTTP 413 / payload-too-large* error for very long date ranges, so the script requests **15-day windows** to stay below that limit.

* **Sampling period:** 1 Nov 2024 – 28 Feb 2025 is chosen to capture a clear up-trend, a sideways phase, and a sharp down-move.
  Each 15-day slice is saved as

  ```
  data/book_btc_usdt_YYYYMMDD_YYYYMMDD.parquet
  ```

  (compressed with Snappy via cuDF/pyarrow).

### II. Data normalization (5-day rolling z-score) - `1_preprocessing_normalize_data.py`

This script normalizes the data for the model in a backward looking fashion and creates a new directory with the normalized data.

I rely on **cuDF** to stream large Parquet blocks through GPU memory and write the normalised result back to disk.

* **Input**  
  `book_symbol_YYYYMMDD_YYYYMMDD.parquet` files (6 Hz, 20-level LOB snapshots) located in `./data/`.  
  I scan only those files whose date range overlaps the requested period.

* **Rolling statistics**  
  For each trading day I update a dictionary of **mean/variance per LOB column**.  
  I keep at most **`window_days` = 5** distinct day-entries, so memory stays bounded even for month-long runs.

* **Normalisation**  
  Before writing a file I compute the window-wide weighted mean ± std and apply a **z-score** transform column-wise:
  \[
  x' = \frac{x - \mu_{\text{window}}}{\sigma_{\text{window}} + 10^{-8}}
  \]
  This stabilises the scale drift that creeps in when spreads widen or volumes explode.

* **Output**  
  Each source file becomes  `./data_normalized/norm_book_btc_usdt_YYYYMMDD_YYYYMMDD.parquet`

Again saved with **PyArrow + Snappy**, so I benefit from columnar layout and fast (de)compression on GPU.

* **API quirks**  
The script detects missing days, warns about gaps, and skips normalisation until at least one full day of stats is available—avoiding a divide-by-tiny-sigma spike at the very beginning of a run.

Run it from the CLI:

```bash
python normalize_data.py \
  --start_date 2025-01-01 --end_date 2025-01-15 \
  --symbol BTC-USDT \
  --input_dir ./data \
  --output_dir ./data_normalized \
  --window_days 5
```

### III. Model architecture & training pipeline - `2_single_model_parallelism.py`

The training pipeline optimizes GPU utilization through parallel batch processing. It relies on specialized modules for data handling and model architecture:

1. **gpu_loaders.py**: GPU-accelerated data loading pipeline that minimizes CPU-GPU transfers. Uses PyTorch DataLoaders with a custom `GPUCachedDataset` that loads all data to GPU memory once at initialization, eliminating transfer bottlenecks during training. It handles chunking, windowing, and labeling operations directly on the GPU.

2. **deeplob_optimized.py**: Contains the optimized DeepLOB architecture, a CNN-LSTM hybrid model for predicting price movements. The `DeepLOB` class implements a 3-layer CNN for dimensionality reduction, followed by an Inception module and LSTM layer for temporal feature extraction. The `load_book_chunk` function complements the GPU loaders by finding and organizing normalized parquet files within a specific date range.

**Please see [model_architecture.md](model_architecture.md) for a detailed description of the model architecture.**

3. **Parallel batch computation**: The `SingleModelParallelTrainer` class splits each batch into multiple micro-batches (`split_batches` parameter) that can be processed in parallel, maximizing GPU utilization. It uses mixed-precision training via PyTorch's AMP (Automatic Mixed Precision) with a gradient scaler to maintain numerical stability while reducing memory usage and increasing throughput. The micro-batch gradients are accumulated before updating the model, effectively simulating a larger batch size without memory constraints.

> **The entire pipeline was trained on data from *1 Nov 2024 to 28 Feb 2025*;  
> the **last 10 % of that span** (late-Jan → 28 Feb 2025) was held out as the test set.**

**Run configuration (key hyper-parameters used in my experiments)**  

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `depth` | **10** | Number of LOB price levels per side included (⇒ 10 × (bid+ask) × (price+size) = 40 raw features). |
| `window` | **100** | Length of the look-back window in ticks fed to the network (≈ 16 s at 6 Hz). |
| `horizon` | **100** | Prediction horizon; the label compares prices 100 ticks in the future to the last tick in the window. |
| `batch_size` | **64** | Logical batch presented to the trainer **before** micro-batch splitting. |
| `split_batches` | **3** | Each logical batch is cut into three micro-batches (≈ 21 samples each) that run sequentially under AMP; gradients are accumulated, emulating an effective batch of 192 without exceeding 4 GB GPU RAM. |
| `alpha` | **0.002** | Return threshold (±0.2 %) for converting the continuous price change into the 3-class label {down, flat, up}. |
| `stride` | **5** | Step size between successive training samples; reduces overlap and speeds up an epoch. |
| `epochs` | **40** | Maximum training epochs. |
| `lr` | **1 × 10⁻³** | AdamW learning rate; chosen via a short LR-range test. |
| `patience` | **5** | Early-stopping patience on validation F1 to prevent over-fitting. |

### IV. Validation pipeline — `3_validation.py`

`ModelValidator` lets me stress-test any pre-trained **DeepLOB** checkpoint on an arbitrary date range without touching the training codebase.

| Step | What happens | Key design choice |
|------|--------------|-------------------|
| **1. Locate data** | `load_book_chunk()` scans `data_normalized/` for `norm_book_…` files that overlap the requested period and warns if there are gaps. | Keeps the evaluation window reproducible and transparent. |
| **2. Pair *normalised* + *raw* LOB** | For every normalised file I eagerly load the matching raw Parquet (same date suffix) and keep only `ask_0_price`. | Lets me compare model output against both the z-scored input and the untouched mid-price. |
| **3. One-shot GPU dataloader** | I concat all normalised frames → save a **temporary** Parquet → re-load it with `create_gpu_data_loaders(valid_frac=1.0)`, stride forced to 1. | Simpler than stitching an IterableDataset and avoids CPU ⇄ GPU ping-pong. |
| **4. Forward pass under AMP** | Validation runs with `torch.amp.autocast('cuda')`; no back-prop, so memory stays ~500 MB. | Exact precision isn’t critical; throughput is. |
| **5. Metrics computed** | • cross-entropy loss<br>• macro-F1 + “directional F1” (avg of up & down)<br>• per-class precision/recall/F1<br>• confusion matrix<br>• class distribution | Same headline numbers as in the paper, easy to benchmark. |
| **6. Outputs saved (opt-in)** | • PNG confusion matrix (`results/deeplob/validation_conf_matrix_*.png`)<br>• text report (all metrics)<br>• Parquet file containing **timestamps + raw & normalised price + prediction** for downstream plotting or burst detection | Keeps artefacts neatly versioned alongside the model file. |

**Run example**

```bash
python 3_validation.py \
    --start_date 2025-03-05 \
    --end_date   2025-03-10 \
    --model_path ./models/deeplob_single_parallel_f1_0.4369.pt
```

This will load five days of data, run the network with depth=10, window=100, horizon=100, batch-size 64, and emit the full metric set in a few minutes on a 4 GB RTX A500 laptop GPU.

### V. Trading-simulation script — `5_trading_strategy.py`

A minimal back-test that turns DeepLOB’s **per-tick class predictions** into concrete long/short trades and benchmarks them against a buy-and-hold baseline.

| Stage | What it does | Key tweak-options |
|-------|--------------|-------------------|
| **Load predictions** | Reads a Parquet file produced by the validation pipeline – must contain `received_time`, `prediction`, `raw_price` columns – and indexes it by timestamp. | *File path pattern:*<br>`results/deeplob/predictions_<model>_YYYYMMDD_YYYYMMDD.parquet` |
| **Pre-process** | Maps classes → signed direction **{-1, 0, +1}** and stores the DataFrame. No smoothing or probability thresholding is applied. | — |
| **Signal filter** | Uses a **consecutive-signal counter**. Only opens/closes a position when the same class repeats `signal_threshold` times in a row.<br>`signal_threshold = 1` reproduces the paper’s naïve “trade every tick” rule; higher values cut noise. | `signal_threshold` *(default = 1, demo = 3)* |
| **Daily loop** | Positions are *intraday only* → anything still open is closed at the first price of the next session. | Avoids overnight risk and simplifies PnL. |
| **Trade book** | Each trade dict records entry/exit time, price, direction, %-profit and exit reason (`signal_reversal` or `end_of_day`). | All trades are for **1 unit**; extendable to size scaling. |
| **Performance analytics** | Computes win-rate, total / mean return, annualised Sharpe (252 d), max drawdown, and buy-and-hold return over the same period. | Results are stored in `self.results` for further plotting. |

#### Typical run

```bash
python 5_trading_strategy.py \
    --predictions ./results/deeplob/predictions_deeplob_single_parallel_f1_0_20250305_20250310.parquet \
    --signal_threshold 3
```
With `signal_threshold` = 3 the script places far fewer, higher-confidence trades, making the outcome a clearer gauge of the classifier’s economic value instead of pure tick-by-tick noise.
