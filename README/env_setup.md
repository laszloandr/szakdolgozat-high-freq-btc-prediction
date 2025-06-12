## ⚙️ Environment setup – run the whole DeepLOB stack locally

> Follow these steps once and you can execute every script **and** the  
> `validation_and_strategy.ipynb` notebook end-to-end on your own machine.

---

### 1. Hardware requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| **GPU** | NVIDIA card, ≥ 4 GB VRAM | cuDF and PyTorch-CUDA require a CUDA device. The repo is tuned on an **RTX A500 4 GB** laptop GPU. |
| **CPU-only / Apple Silicon** | *Not officially supported* | RAPIDS needs CUDA. You can swap cuDF → pandas and install a CPU-only PyTorch build, but training will be 10-100× slower. |

---

### 2. OS & driver

* **Windows 10/11** – install **WSL 2** + Ubuntu 22.04 and enable GPU passthrough  
  <https://learn.microsoft.com/windows/wsl/install>
* **Native Linux** – install the **NVIDIA 550+ driver** and the **CUDA 12.9** runtime.  
  No Java/JDK needed; CUDA ships everything.

> **Check versions**  
> ```bash
> nvcc --version      # should report release 12.9
> nvidia-smi          # driver ≥ 550.xx
> ```

---

### 3. Create the linux virtual environment (open in WSL terminal)

```bash
# inside WSL or your Linux shell
conda create -n rapids-25.06 \
    -c rapidsai -c conda-forge -c nvidia \
    rapids=25.06 python=3.13 cuda-version=12.9
conda activate rapids-25.06
```

### 4. Install Python packages

# from the repository root
pip install -r requirements.txt

### 5. Recommended IDE workflow

Install Visual Studio Code.

Open VS Code → “WSL: Connect to WSL in New Window”.

Open the repo folder – the interpreter selector should list rapids-25.06.

### 6. run `validation_and_strategy.ipynb`