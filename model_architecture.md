## DeepLOB architecture — what each block does and why I built it this way

| Stage | My goal | Layer stack & activation | Shape flow |
|-------|---------|--------------------------|------------|
| **1. Convolutional “micro-features”** | Let the network **learn one book level in isolation** (price-size interaction) before mixing levels. | `Conv2d 1×2 → LeakyReLU` | (100 × 40) → (100 × 20) |
| | Couple **neighbouring levels** into local imbalance cues. | `Conv2d 1×2 → LeakyReLU` | (100 × 20) → (100 × 10) |
| | Collapse depth so each time-step becomes a **32-D vector**. | `Conv2d 1×depth → LeakyReLU` | (100 × 10) → (100 × 1) |

---

### 2. Inception @ 32 ch — multi-scale pattern mining

*Why I need it*  
Price moves can be driven by **short bursts** or **slow drifts**. A single kernel size would miss one or the other, so I adopt a GoogLeNet-style Inception block: several filter sizes in parallel, cheap 1 × 1 projections, all concatenated into 32 channels.

**My custom branch split**

| Branch | Channels | Captures |
|--------|----------|----------|
| 1 × 1 | 8 (25 %) | identity / cheap mixing |
| 3 × 1 | 12 (37.5 %) | 3-tick bursts |
| 5 × 1 | 8 (25 %) | slightly longer pulses |
| 3 × 1 max-pool → 1 × 1 | 4 (12.5 %) | pooled context |

The `ratio = (0.25, 0.375, 0.25, 0.125)` tuple ensures the four branches **always sum to 32 channels**, keeping downstream tensor shapes stable.

---

### 3. LSTM @ 64 hid — stitching cues into a tradeable story

After Inception, every time-step encodes rich spatial context.  
A 64-unit **LSTM** lets me:

* remember patterns over the full 100-tick window,  
* weigh evidence over time without exploding parameters,  
* keep the whole model end-to-end trainable.

The last hidden state feeds a `Linear(64 → 3)` head that outputs probabilities for **down / flat / up**.

---

### End-to-end tensor sizes

(B, 100, 40) → Conv stack → (B, 100, 1)
↓
Inception (32 ch) → (B, 100, 32)
↓
LSTM(64) → (B, 64)
↓
Linear → (B, 3)

With cuDNN kernels and TF-32 the whole pipeline trains in minutes per epoch and fits easily in a 4 GB laptop GPU.
