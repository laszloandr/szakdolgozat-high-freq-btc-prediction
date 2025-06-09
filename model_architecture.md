## DeepLOB architecture — what each block does and why I built it this way

### 1. CNN block – “compress and clean” the raw LOB slice

| Sub-step | Why I need it | Layer spec (PyTorch) | Shape flow* |
|----------|---------------|----------------------|-------------|
| **1. Halve feature depth** | Capture **pairwise interaction** between adjacent bid/ask levels while expanding to 32 channels. | `Conv2d(1 → 32, kernel=(1,2), stride=(1,2))`<br>`LeakyReLU(0.01)` | (B, 1, 100, 40) → (B, 32, 100, 20) |
| **2. Halve again** | Build **4-level context** (each output cell now “sees” 4 original levels). | `Conv2d(32 → 32, kernel=(1,2), stride=(1,2))`<br>`LeakyReLU(0.01)` | (B, 32, 100, 20) → (B, 32, 100, 10) |
| **3. Collapse width** | Mix all 10 remaining levels into a single **32-D embedding per timestamp**—ready for Inception. | `Conv2d(32 → 32, kernel=(1, depth))` *(= 1×10)*<br>`LeakyReLU(0.01)` | (B, 32, 100, 10) → (B, 32, 100, 1) |

\* The spatial axes are **(time, depth)**; `depth=10` for a 20-level LOB since each level contributes price + size × bid/ask = 4 features.  
After this block I `squeeze` the last dimension and permute to `(B, 100, 32)`, handing a clean 32-feature sequence to the Inception module.


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
