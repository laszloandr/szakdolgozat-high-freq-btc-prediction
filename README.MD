# DeepLOB-BTC – project overview

## Motivation  
In high-frequency markets the **limit-order-book (LOB) imbalance** contains most of the actionable information: which side is leaning, how deeply, and for how long.  
Hand-crafted rules rarely capture these micro-patterns, so I built a **deep-learning pipeline** that learns them end-to-end and then shows whether the signal can be monetised in a simple trading loop.

---

## Data  
| Item | Details |
|------|---------|
| Source | Binance spot **BTC-USDT** order-book snapshots |
| Raw granularity | **6 Hz**, **20 levels** per side (price + size ⇒ 40 columns) |
| Modelling input | First **10 levels**, z-scored in a 5-day rolling window |
| Period | **01 Nov 2024 → 28 Feb 2025** |
| Train / Test split | Chronological, last **10 %** of the period held out for test |

That gives nearly four months of data covering a clean up-trend, a sideways regime and a sharp sell-off.

---

## Model architecture  
1. **CNN “compression” block** – three 1 × 2 convolutions reduce the 40-wide LOB slice to a single 32-channel vector per tick while learning price–size interactions and cross-level imbalance.  
2. **Inception @ 32 ch** – parallel 1 × 1, 3 × 1, 5 × 1 and pool-projection branches (`ratio = 0.25 / 0.375 / 0.25 / 0.125`) mine patterns at multiple temporal scales without blowing up parameter count.  
3. **LSTM 64-hid** – integrates those local cues over the full 100-tick window.  
4. **Linear head** – outputs three probabilities: **down / flat / up**.

The target label looks 100 ticks (~16.7 s) ahead and checks if the mid-price moved **±0.2 %**; the network learns to classify every tick into {-1, 0, +1}.  
Main metric: **directional F1** – average of the Up- and Down-class F1 scores.

Please see further details in `.\README\model_architecture.md`
---

## Training & validation  
* **Training window**: 01 Nov 2024 → 17 Febr 2025  
* **Hyper-params**: batch 64, split into 3 micro-batches (effective 192), 40 epochs, AdamW lr = 1e-3, early-stop patience = 5.  
* **Validation pipeline**: loads normalised prices, runs the model under AMP, reports loss, macro F1, directional F1, per-class precision/recall and a PNG confusion matrix.

Please see the created models in `.\models` folder, being the `.\models\deeplob_single_parallel_f1_0.4369.pt` best performing model which I used in the validation. Please see `best_confusion_matrix_single_parallel_100-100-0.002.png` for the confusion matrix of the model validation (last 10% of period, which is 17 Febr 2025 → 28 Febr 2025) in `.\results\deeplob`.

Please see the best performing model's results for period 05 March 2025 → 10 March 2025 in files `validation_conf_matrix_deeplob_single_parallel_f1_0_20250606_091933.png` and `validation_report_deeplob_single_parallel_f1_0_20250606_091933.txt` in `.\results\deeplob`

PLEASE RUN `validation_and_strategy.ipynb` TO RUN THE VALIDATION PIPELINE AND THE TRADING STRATEGY.
---

## Trading simulation  
A deliberately simple rule set converts predictions to P&L:

* map {0, 1, 2} → {-1, 0, +1}.  
* open **long** after `signal_threshold` consecutive +1’s, open **short** after the same number of -1’s (default threshold = 3).  
* close on the first opposite streak or at session end; always flat overnight.  
* size = 1 unit, P&L computed on raw mid-price.  
* performance report: win-rate, total & average return, Sharpe (252d), max drawdown, and buy-and-hold comparison.

The goal isn’t a production strategy but a sanity check: does the classifier generate tradeable directional bias after realistic filtering?

Please see `.\results\visualizations\trading_strategy.html` for the performance of the trading strategy.
Please run the `validation_and_strategy.ipynb` respective cell to see the above written metrics of the trading performance.

---

## Folder map  
README.md ← this file
README/env_setup.md ← step-by-step GPU env guide
README/model_architecture.md← deeper CNN/Inception/LSTM notes
README/pipeline_overview.md ← normalise → train → validate → trade

validation_and_strategy.ipynb← one-click notebook demo


Follow `README/env_setup.md`, and you can reproduce every result on a single NVIDIA GPU computer.
