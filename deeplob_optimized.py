"""
DeepLOB model with performance optimization and enhanced progress monitoring.
This file contains optimizations to improve performance and monitoring of the
original deeplob.ipynb notebook.
"""
import os, re, datetime as dt, math, time
from pathlib import Path


import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd
import cudf                 # Fast Parquet loading on GPU
import cupy as cp            # CuPy for GPU-accelerated arrays
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torch.cuda.amp as amp
from sklearn.metrics import f1_score, confusion_matrix                       # Evaluation metric

# PyTorch and CUDA configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# Optimize for CNN with fixed input sizes
torch.backends.cudnn.benchmark = True    # Accelerates CNNs with fixed input dimensions
torch.backends.cuda.matmul.allow_tf32 = True    # Allow TF32 for faster computation


def load_book_chunk(
    start_date: dt.datetime,
    end_date:   dt.datetime,
    symbol: str,
    data_dir: str = "./szakdolgozat-high-freq-btc-prediction/data",
) -> cudf.DataFrame:
    """
    Load LOB (Limit Order Book) Parquet files with GPU acceleration (cudf).
    This function only loads raw data; normalization happens in the Dataset class.
    
    Args:
        start_date: Beginning date for data loading
        end_date: End date for data loading
        symbol: Trading pair symbol (e.g., "BTC-USDT")
        data_dir: Directory containing Parquet files
        
    Returns:
        cudf.DataFrame with loaded order book data
    """
    print(f"Loading data for {symbol} from {start_date} to {end_date}...")
    t_start = time.time()
    
    sym_pat = symbol.lower().replace("-", "_")
    rex = re.compile(rf"book_{sym_pat}_(\d{{8}})_(\d{{8}})\.parquet$", re.I)

    sd = pd.to_datetime(start_date); ed = pd.to_datetime(end_date)
    frames = []

    # Performance improvement: first list all matching files to show progress
    matching_files = []
    for fn in sorted(os.listdir(data_dir)):
        m = rex.match(fn);  fp = Path(data_dir) / fn
        if not m or not fp.is_file(): continue

        f_sd, f_ed = pd.to_datetime(m.group(1)), pd.to_datetime(m.group(2))
        if f_ed < sd or f_sd > ed:         # completely outside our date range
            continue
        matching_files.append(fp)
    
    print(f"Found {len(matching_files)} matching files.")
    
    for i, fp in enumerate(matching_files):
        print(f"Processing file {i+1}/{len(matching_files)}: {fp.name}... ", end="", flush=True)
        t_file_start = time.time()
        
        df = cudf.read_parquet(fp)
        msk = (df["received_time"]>=sd) & (df["received_time"]<=ed)
        
        if bool(msk.any()):
            filtered_df = df[msk]
            frames.append(filtered_df)
            print(f"added {len(filtered_df)} rows. Took {time.time()-t_file_start:.2f}s")
        else:
            print("no matching rows")

    if frames:
        result = cudf.concat(frames, ignore_index=True)
        print(f"Total rows: {len(result)}")
    else:
        result = cudf.DataFrame()
        print("No data found.")
    
    print(f"Data loading complete. Took {time.time()-t_start:.2f}s")
    return result


class LobDataset(Dataset):
    """
    Dataset for Limit Order Book data processing and normalization.
    Handles data preparation for the DeepLOB model, including normalization
    and conversion to PyTorch tensors on GPU.
    """
    def __init__(self, df: cudf.DataFrame,
                 depth:int = 10,          # Number of price levels to use
                 window:int = 100,         # Number of time points to look back
                 horizon:int = 100,        # How far into future to predict
                 alpha:float = 0.002,      # Threshold for price movement classification
                 stride:int = 5):          # Step size between consecutive samples
        
        print("Initializing LobDataset...")
        t_start = time.time()

        self.depth, self.window = depth, window
        self.horizon, self.alpha = horizon, alpha
        self.stride = stride

        # Feature selection - identify relevant price and size columns
        print("Selecting feature columns...")
        t1 = time.time()
        pat = rf'(bid|ask)_[0-9]{{1,2}}_(price|size)'
        feat_cols = [c for c in df.columns
                     if re.match(pat, c) and int(c.split('_')[1]) < depth]
        print(f"Selected {len(feat_cols)} feature columns. Took {time.time()-t1:.2f}s")

        # Z-score normalization - major performance bottleneck in original implementation
        # ---- 5-day normalization optimized on GPU ----
        t1 = time.time()
        print("Performing optimized z-score normalization...")
        
        # cuDF doesn't support the std() method with ewm(), so we use an alternative approach
        # Faster approximation: using fixed window statistics instead of rolling window
        wnd = int(5*24*60*60*6.12)  # 5-day equivalent in data points
        window_size = min(wnd, len(df) // 2)  # Use at most half of the data
        
        print(f"Using fixed window statistics (size={window_size})...")
        t_stat_start = time.time()
        
        # Calculate statistics on a subset of data to avoid data leakage
        sample_df = df[feat_cols].head(window_size)
        mu = sample_df.mean()
        sig = sample_df.std() + 1e-8  # Add small epsilon to avoid division by zero
        
        print(f"Statistics calculation took {time.time()-t_stat_start:.2f}s")
        
        # Apply normalization to all data
        print("Applying normalization...")
        t_norm_start = time.time()
        # cuDF doesn't support float16, use float32 instead
        df[feat_cols] = ((df[feat_cols] - mu) / sig).astype('float32')
        print(f"Normalization application took {time.time()-t_norm_start:.2f}s")
        print(f"Normalization complete. Took {time.time()-t1:.2f}s")

        # Keep data on GPU for better performance instead of transferring to CPU
        print("Converting data to GPU tensors...")
        t1 = time.time()
        
        # Use reshape() instead of view() to avoid memory layout issues
        # Correct parameter order: device, dtype, then non_blocking
        self.X = torch.from_dlpack(df[feat_cols].T.to_dlpack())\
                       .contiguous()\
                       .reshape(-1, depth*4)\
                       .to(device=device, dtype=torch.float16, non_blocking=True)

        # Calculate mid-price for return computation
        mid = (df["bid_0_price"] + df["ask_0_price"]) / 2
        self.mid = torch.from_dlpack(mid.to_dlpack())\
                         .to(device=device, dtype=torch.float16, non_blocking=True)
        print(f"Data transfer complete. Took {time.time()-t1:.2f}s")
        
        # Display information about the dataset dimensions
        print(f"Dataset initialized. X shape: {self.X.shape}, mid shape: {self.mid.shape}")
        print(f"Total initialization time: {time.time()-t_start:.2f}s")
        print(f"Effective samples: {self.__len__()}")

    def __len__(self):
        """Calculate number of samples considering window size, horizon and stride"""
        return (len(self.X) - self.window - self.horizon) // self.stride

    def __getitem__(self, i):
        """Get a single sample: input features and target class"""
        idx = i * self.stride
        j   = idx + self.window

        # Use reshape() instead of view() due to non-contiguous memory layout
        x = self.X[idx:j].contiguous().reshape(self.window, self.depth*4)   # (window, features)
        
        # Calculate price change and classify direction (all on GPU)
        r = (self.mid[j+self.horizon-1] - self.mid[j-1]) / self.mid[j-1]
        y = 2 if r >  self.alpha else 0 if r < -self.alpha else 1  # 0=down, 1=stable, 2=up
        
        # Keep label on GPU as well
        return x, torch.tensor(y, dtype=torch.long, device=device)


def make_loaders(df,
                 depth:   int = 10,
                 window:  int = 100,
                 horizon: int = 100,
                 valid_frac: float = 0.1,
                 batch: int = 32):
    """
    Create training and validation data loaders from the input dataframe.
    
    Args:
        df: cuDF DataFrame with order book data
        depth: Number of price levels to use
        window: Number of time points to use as input
        horizon: How far into future to predict
        valid_frac: Fraction of data to use for validation
        batch: Batch size for training
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    print("Creating data loaders...")
    t_start = time.time()
    
    print("Initializing dataset...")
    ds = LobDataset(df,
                    depth   = depth,
                    window  = window,
                    horizon = horizon,
                    alpha=0.01)

    n = len(ds)
    print(f"Dataset contains {n} samples")
    
    print("Creating train/val split...")
    # Using temporal split instead of random to avoid data leakage
    # Later timepoints are used for validation
    n_val = int(n * valid_frac)
    n_train = n - n_val
    
    train_ds = torch.utils.data.Subset(ds, range(n_train))
    val_ds = torch.utils.data.Subset(ds, range(n_train, n))
    
    print(f"Train samples: {n_train}, Validation samples: {n_val}")

    print("Creating data loaders...")
    # num_workers=0 ensures data stays on GPU and prevents transfer issues
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, 
                           num_workers=0, pin_memory=False)
    
    print(f"Data loaders created. Took {time.time()-t_start:.2f}s")
    return train_loader, val_loader


class InceptionModule(nn.Module):
    """
    Inception module as described in the paper with limited details (only diagrams).
    Filter widths were inferred as: 1×1, 3×1, 5×1 + 3×1 max-pool branch, all with 32 channels.
    """
    def __init__(self, in_ch, out_ch=32):
        super().__init__()
        self.branch1 = nn.Conv1d(in_ch, out_ch, kernel_size=1)  # 1×1 convolution branch

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),  # Dimensionality reduction
            nn.LeakyReLU(0.01, True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),  # 3×1 convolution
            nn.LeakyReLU(0.01, True),
        )

        self.branch5 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1),  # Dimensionality reduction
            nn.LeakyReLU(0.01, True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),  # 5×1 convolution
            nn.LeakyReLU(0.01, True),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),  # Max pooling branch
            nn.Conv1d(in_ch, out_ch, kernel_size=1),  # Projection to reduce channels
            nn.LeakyReLU(0.01, True),
        )

        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):                 # x: (B,100,16)  (batch, time, features)
        x = x.permute(0,2,1)              # (B,16,100)  Conv1d expects channels before time
        y1 = self.branch1(x)              # Output from 1×1 branch
        y2 = self.branch3(x)              # Output from 3×1 branch
        y3 = self.branch5(x)              # Output from 5×1 branch
        y4 = self.branch_pool(x)          # Output from max-pool branch - all (B,32,100)
        y = torch.cat((y1, y2, y3, y4), dim=1)    # Concatenate along channel dim (B,128,100)
        y = y.permute(0,2,1)             # Back to (B,100,128) for LSTM input
        return y[..., :32]     # Keep only 32 channels as per paper


class DeepLOB(nn.Module):
    """
    DeepLOB model architecture as described in the paper.
    Predicts price movement direction (up/stable/down) based on limit order book data.
    """
    def __init__(self, depth: int = 10):
        super().__init__()
        self.depth = depth

        # CNN layers to reduce input dimensionality: 80→40→20→depth (generalized)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (1,2), (1,2)),  # First reduction layer
            nn.LeakyReLU(0.01, True),
            nn.Conv2d(32,32,(1,2),(1,2)),   # Second reduction layer
            nn.LeakyReLU(0.01, True),
            nn.Conv2d(32,32,(1,depth),(1,1)),   # Final layer with kernel=(1,depth)
            nn.LeakyReLU(0.01, True),
        )

        # ——— 2. Inception module with 32 output channels ———
        self.inception = InceptionModule(in_ch=32, out_ch=32)

        # ——— 3. LSTM with 64 hidden units ———
        self.lstm = nn.LSTM(input_size=32,
                    hidden_size=64,
                    batch_first=True,
                    dropout=0.1)      # 0.1 dropout as in the paper's DeepLOB version

        # Output layer for 3-class classification (down/stable/up)
        self.head = nn.Linear(64, 3)

    def forward(self, x):                  # x: (B, 100, 40) - batch, time, features
        # Measure execution time of each component for performance analysis
        t0 = time.time()
        
        x = x.unsqueeze(1)                 # -> (B,1,100,40) - add channel dimension for CNN
        
        # CNN processing
        t1 = time.time()
        x = self.cnn(x)                    # -> (B,32,100,10) - apply CNN layers
        t2 = time.time()
        
        x = x.squeeze(-1).permute(0,2,1)   # -> (B,100,16)
        
        # Inception
        t3 = time.time()
        x = self.inception(x)              # -> (B,100,32)
        t4 = time.time()
        
        # LSTM
        t5 = time.time()
        out, _ = self.lstm(x)              # -> (B,100,64)
        t6 = time.time()
        
        y = self.head(out[:,-1])           # utolsó időlépés
        t7 = time.time()
        
        # Solo reportamos tiempos durante el entrenamiento para no afectar la inferencia
        if self.training and torch.cuda.current_stream().query() and x.size(0) > 1:
            print(f"Forward pass times - CNN: {t2-t1:.4f}s, Inception: {t4-t3:.4f}s, LSTM: {t6-t5:.4f}s, Head: {t7-t6:.4f}s, Total: {t7-t0:.4f}s")
        
        return y


def train(model, train_loader, val_loader,
          epochs=40, lr=1e-3, patience_lim=5,
          accum_steps=2):                     

    print(f"Starting training with {epochs} epochs, lr={lr}, patience={patience_lim}, accum_steps={accum_steps}")
    print(f"Train loader has {len(train_loader.sampler)} samples, Val loader has {len(val_loader.sampler)} samples")
    print(f"Memory usage before training: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    opt = torch.optim.AdamW(model.parameters(),
                           lr=lr, betas=(0.9,0.999),
                           eps=1e-8, weight_decay=1e-4,
                           fused=True)              # ha PyTorch ≥ 2
    scaler = amp.GradScaler()
    ce = nn.CrossEntropyLoss()

    best_f1, wait = 0.0, 0
    
    total_train_time = 0
    
    # Monitorear GPU
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {ep}/{epochs} ---")
        
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0
        batch_times = []
        data_load_times = []
        forward_times = []
        backward_times = []
        opt_step_times = []
        
        print(f"Training: {len(train_loader)} batches")
        for step, (xb, yb) in enumerate(train_loader):
            batch_start = time.time()
            data_load_end = time.time()
            data_load_times.append(data_load_end - batch_start)
            
            # Monitorear memoria GPU
            if step % 50 == 0:
                print(f"GPU memory at step {step}: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                
            if step % 10 == 0:
                print(f"Training batch {step}/{len(train_loader)}")
            
            # A tenzorok már a GPU-n vannak a dataset-ben, csak folytonosság biztosítása szükséges
            xb = xb.contiguous()
            # yb már a GPU-n van, nem kell átmásolni
            
            # Frissített autocast API használata
            with torch.amp.autocast(device_type='cuda'):
                forward_start = time.time()
                logits = model(xb)
                forward_end = time.time()
                forward_times.append(forward_end - forward_start)
                
                loss = ce(logits, yb) / accum_steps   # ➊ skálázott loss

            backward_start = time.time()
            scaler.scale(loss).backward()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)

            # ➋ csak minden accum_steps-edik mini-batchnél frissítünk
            if (step + 1) % accum_steps == 0:
                opt_start = time.time()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                opt_end = time.time()
                opt_step_times.append(opt_end - opt_start)

            running_loss += loss.item() * accum_steps   # eredeti loss
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Mostrar estadísticas cada 50 batches
            if (step + 1) % 50 == 0:
                avg_batch = sum(batch_times[-50:]) / min(50, len(batch_times[-50:]))
                avg_data = sum(data_load_times[-50:]) / min(50, len(data_load_times[-50:]))
                avg_forward = sum(forward_times[-50:]) / min(50, len(forward_times[-50:]))
                avg_backward = sum(backward_times[-50:]) / min(50, len(backward_times[-50:]))
                avg_opt = sum(opt_step_times[-50:]) / min(50, len(opt_step_times[-50:]))
                
                print(f"Batch {step+1}/{len(train_loader)} - Loss: {loss.item()*accum_steps:.4f}")
                print(f"Avg times: Batch={avg_batch:.4f}s, Data={avg_data:.4f}s ({avg_data/avg_batch*100:.1f}%), "
                      f"Forward={avg_forward:.4f}s ({avg_forward/avg_batch*100:.1f}%), "
                      f"Backward={avg_backward:.4f}s ({avg_backward/avg_batch*100:.1f}%), "
                      f"Optim={avg_opt:.4f}s ({avg_opt/avg_batch*100:.1f}%)")
        
        # Mostrar estadísticas de entrenamiento
        avg_batch = sum(batch_times) / len(batch_times)
        avg_data = sum(data_load_times) / len(data_load_times)
        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)
        avg_opt = sum(opt_step_times) / len(opt_step_times) if opt_step_times else 0
        
        print("\nTraining phase statistics:")
        print(f"Avg times: Batch={avg_batch:.4f}s, Data={avg_data:.4f}s ({avg_data/avg_batch*100:.1f}%), "
              f"Forward={avg_forward:.4f}s ({avg_forward/avg_batch*100:.1f}%), "
              f"Backward={avg_backward:.4f}s ({avg_backward/avg_batch*100:.1f}%), "
              f"Optim={avg_opt:.4f}s ({avg_opt/avg_batch*100:.1f}%)")

        # ---------- VALID ----------
        print("\nStarting validation phase...")
        val_start = time.time()
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for i, (xb, yb) in enumerate(val_loader):
                if i % 20 == 0:
                    print(f"Validation batch {i}/{len(val_loader)}")
                    
                t_start = time.time()
                # Az adatok már a GPU-n vannak, csak folytonosság biztosítása szükséges
                logits = model(xb.contiguous())
                batch_preds = logits.argmax(1)
                
                # Címkék és predikciók tárolása a későbbi elemzéshez
                y_true.append(yb)
                y_pred.append(batch_preds)
                
                # Batch előrehaladásának kijelzése
                if i % 20 == 0:
                    print(f"Validation batch {i}/{len(val_loader)}, inference time: {time.time() - t_start:.4f}s")

        val_end = time.time()
        print(f"Validation completed in {val_end - val_start:.2f}s")

        # Összegyűjtjük az összes predikciót és valós címkét
        all_true = torch.cat(y_true)
        all_pred = torch.cat(y_pred)
        
        # F1 score számítása - csak ezt számoljuk minden epoch-nál
        cpu_macro_f1 = f1_score(all_true.cpu().numpy(), all_pred.cpu().numpy(), average='macro')
        macro_f1 = cpu_macro_f1  # Ezt használjuk a továbbiakban

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        total_train_time += epoch_time
        
        print(f"Epoch {ep:02d} completed in {epoch_time:.2f}s")
        print(f"loss={running_loss/len(train_loader.sampler):.4f} F1={macro_f1:.4f}")
        print(f"Average epoch time so far: {total_train_time/ep:.2f}s")
        print(f"Estimated remaining time: {(epochs-ep)*(total_train_time/ep)/60:.2f} minutes")
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

        # ---------- EARLY-STOP ----------
        if macro_f1 > best_f1 + 1e-4:
            best_f1, wait = macro_f1, 0
            print(f"New best F1: {best_f1:.4f}, saving model and calculating confusion matrix...")
            torch.save(model.state_dict(), "best_deeplob.pt")
            
            # Csak a legjobb F1 score esetén számoljuk ki és mentjük el a confusion matrix-ot
            print("Calculating detailed metrics for best model...")
            
            # GPU-n számítjuk a confusion matrix-ot a legjobb modellhez
            conf_matrix = torch.zeros(3, 3, dtype=torch.int, device=device)
            for t in range(3):
                for p in range(3):
                    conf_matrix[t, p] = torch.sum((all_true == t) & (all_pred == p))
            
            # Precision és recall számítása osztályonként
            precision = torch.zeros(3, device=device)
            recall = torch.zeros(3, device=device)
            
            for i in range(3):
                # Precision: TP / (TP + FP)
                precision[i] = conf_matrix[i, i] / conf_matrix[:, i].sum() if conf_matrix[:, i].sum() > 0 else 0
                # Recall: TP / (TP + FN)
                recall[i] = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0
            
            # F1 score: 2 * (precision * recall) / (precision + recall)
            f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Kiírjuk a részletes statisztikákat
            cm = conf_matrix.cpu().numpy()  # CPU-ra konvertálás csak a megjelenítéshez
            print("Best model confusion matrix:")
            print(cm)
            print(f"Per-class F1 scores: {f1_per_class.cpu().numpy()}")
            
            # Confusion matrix vizualizáció és mentés
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Down', 'Stable', 'Up'],
                        yticklabels=['Down', 'Stable', 'Up'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Best Confusion Matrix - Epoch {ep} (F1: {macro_f1:.4f})')
            plt.tight_layout()
            plt.savefig('best_confusion_matrix.png')
            print("Saved confusion matrix to 'best_confusion_matrix.png'")
            
            # Részletesebb statisztikák mentése szöveges fájlba
            with open('best_classification_report.txt', 'w') as f:
                f.write(f"Epoch: {ep}\n")
                f.write(f"F1 Score: {macro_f1:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(str(cm) + "\n\n")
                
                # Osztályonkénti metrikák
                f.write("Per-class metrics:\n")
                class_names = ['Down', 'Stable', 'Up']
                for i, class_name in enumerate(class_names):
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {precision[i].item():.4f}\n")
                    f.write(f"  Recall: {recall[i].item():.4f}\n")
                    f.write(f"  F1: {f1_per_class[i].item():.4f}\n\n")
                
                # Osztályok eloszlása a validációs halmazban
                class_counts = [torch.sum(all_true == i).item() for i in range(3)]
                f.write("Class distribution in validation set:\n")
                for i, class_name in enumerate(class_names):
                    f.write(f"  {class_name}: {class_counts[i]} samples ({class_counts[i]/len(all_true)*100:.2f}%)\n")
            
            print("Saved detailed metrics to 'best_classification_report.txt'")

        else:
            wait += 1
            print(f"No improvement for {wait} epochs. Best F1: {best_f1:.4f}")
            
        if wait >= patience_lim:
            print(f"Early stopping after {ep} epochs.")
            break
    
    print(f"Training completed in {total_train_time:.2f}s ({total_train_time/60:.2f} minutes)")
    return model


def main():
    # Configuraciones
    print("\n=== DeepLOB Performance Optimized Version ===\n")
    print("Starting data loading process...")
    
    # 1. Carga de datos
    t_start = time.time()
    df = load_book_chunk(
            dt.datetime(2025, 2, 20),
            dt.datetime(2025, 2, 28),
            "BTC-USDT")
    print(f"Data loading completed in {time.time()-t_start:.2f}s")
    
    # 2. Creación de loaders
    print("\nCreating data loaders...")
    t_start = time.time()
    train_loader, val_loader = make_loaders(
            df,
            depth   = 10,
            window  = 100,
            horizon = 100,
            batch   = 64)
    print(f"Data loaders created in {time.time()-t_start:.2f}s")
    
    # 3. Creación y compilación del modelo
    print("\nInitializing model...")
    t_start = time.time()
    model = DeepLOB(depth=10).to(
                device,
                memory_format=torch.channels_last)
    print(f"Model initialization completed in {time.time()-t_start:.2f}s")
    
    # Kikapcsoljuk a model compilation-t, mivel nincs C fordító a rendszeren
    # print("\nCompiling model...")
    # t_start = time.time()
    # model = torch.compile(model, mode="reduce-overhead")
    # print(f"Model compilation completed in {time.time()-t_start:.2f}s")
    print("Model compilation skipped (requires C compiler).")
    
    # 4. Entrenamiento
    print("\nStarting training...")
    train(model, train_loader, val_loader,
          epochs       = 40,
          lr           = 1e-3,
          patience_lim = 5,
          accum_steps  = 2)
    
    print("\n=== Training complete! ===")


if __name__ == "__main__":
    main()
