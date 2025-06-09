# szakdolgozat-high-freq-btc-prediction



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

### II. Data normalization (5-day rolling z-score) - 1_preprocessing_normalize_data.py

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
  --start_date 2025-01-01 --end_date 2025-03-01 \
  --symbol BTC-USDT \
  --input_dir ./data \
  --output_dir ./data_normalized \
  --window_days 5
```

### III. Model architecture & training pipeline - 2_single_model_parallelism.py

The training pipeline optimizes GPU utilization through parallel batch processing. It relies on specialized modules for data handling and model architecture:

1. **gpu_loaders.py**: GPU-accelerated data loading pipeline that minimizes CPU-GPU transfers. Uses PyTorch DataLoaders with a custom `GPUCachedDataset` that loads all data to GPU memory once at initialization, eliminating transfer bottlenecks during training. It handles chunking, windowing, and labeling operations directly on the GPU.

2. **deeplob_optimized.py**: Contains the optimized DeepLOB architecture, a CNN-LSTM hybrid model for predicting price movements. The `DeepLOB` class implements a 3-layer CNN for dimensionality reduction, followed by an Inception module and LSTM layer for temporal feature extraction. The `load_book_chunk` function complements the GPU loaders by finding and organizing normalized parquet files within a specific date range.

** Please see [model_architecture.md](model_architecture.md) for a detailed description of the model architecture.**

3. **Parallel batch computation**: The `SingleModelParallelTrainer` class splits each batch into multiple micro-batches (`split_batches` parameter) that can be processed in parallel, maximizing GPU utilization. It uses mixed-precision training via PyTorch's AMP (Automatic Mixed Precision) with a gradient scaler to maintain numerical stability while reducing memory usage and increasing throughput. The micro-batch gradients are accumulated before updating the model, effectively simulating a larger batch size without memory constraints.



