"""
DeepLOB model with performance optimization and enhanced progress monitoring.
This file contains optimizations to improve performance and monitoring of the
original deeplob.ipynb notebook.
"""
import os, re, datetime as dt, math, time
from pathlib import Path

# Helper function for performance logging
def p(msg):
    """Print performance log message"""
    print(msg)


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


def find_normalized_files(
    start_date: dt.datetime,
    end_date: dt.datetime,
    symbol: str,
    data_dir: str = "./szakdolgozat-high-freq-btc-prediction/data_normalized",
) -> list:
    """
    Find normalized parquet files within date range.
    
    Args:
        start_date: Beginning date for data loading
        end_date: End date for data loading
        symbol: Trading pair symbol (e.g., "BTC-USDT")
        data_dir: Directory containing normalized parquet files
        
    Returns:
        List of matching file paths sorted chronologically
    """
    print(f"Finding normalized data for {symbol} from {start_date} to {end_date}...")
    
    sym_pat = symbol.lower().replace("-", "_")
    rex = re.compile(rf"norm_book_{sym_pat}_(\d{{8}})_(\d{{8}})\.parquet$", re.I)

    sd = pd.to_datetime(start_date); ed = pd.to_datetime(end_date)
    matching_files = []
    
    # Check if normalized data directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: Normalized data directory '{data_dir}' not found!")
        return []
        
    # Find all matching files
    for fn in sorted(os.listdir(data_dir)):
        m = rex.match(fn)  
        fp = Path(data_dir) / fn
        if not m or not fp.is_file(): 
            continue

        f_sd, f_ed = pd.to_datetime(m.group(1)), pd.to_datetime(m.group(2))
        if f_ed < sd or f_sd > ed:  # completely outside our date range
            continue
            
        matching_files.append({
            'path': fp,
            'start_date': f_sd,
            'end_date': f_ed,
            'filename': fn
        })
    
    # Sort files chronologically
    matching_files.sort(key=lambda x: x['start_date'])
    
    print(f"Found {len(matching_files)} normalized files.")
    for i, file_info in enumerate(matching_files):
        print(f"  {i+1}. {file_info['filename']} "  
              f"({file_info['start_date'].strftime('%Y-%m-%d')} - {file_info['end_date'].strftime('%Y-%m-%d')})")
    
    return matching_files


def load_book_chunk(
    start_date: dt.datetime,
    end_date: dt.datetime,
    symbol: str,
    data_dir: str = "./szakdolgozat-high-freq-btc-prediction/data_normalized",
) -> list:
    """
    Find normalized LOB (Limit Order Book) Parquet files.
    Instead of loading all files at once, this returns a list of file information
    to be loaded one by one during training.
    
    Args:
        start_date: Beginning date for data loading
        end_date: End date for data loading
        symbol: Trading pair symbol (e.g., "BTC-USDT")
        data_dir: Directory containing normalized parquet files
        raw_data_dir: Directory containing raw data files (for fallback)
        
    Returns:
        List of file information for sequential processing
    """
    # Try to find normalized files first
    file_infos = find_normalized_files(start_date, end_date, symbol, data_dir)
    
    if not file_infos:
        print("No normalized files found. Did you run normalize_data.py first?")
        print("Please run normalize_data.py to prepare normalized data.")
        return []
    
    return file_infos


class StreamingLobDataset(Dataset):
    """
    Dataset for loading pre-normalized Limit Order Book data one file at a time.
    This class loads only one file at a time to save memory.
    """
    def __init__(self, file_infos: list, 
                 depth: int = 10,          # Number of price levels to use
                 window: int = 100,         # Number of time points to look back
                 horizon: int = 100,        # How far into future to predict
                 alpha: float = 0.002,      # Threshold for price movement classification
                 stride: int = 5,          # Step size between consecutive samples
                 current_file_idx: int = 0):  # Index of the current file to load
        
        print("Initializing StreamingLobDataset...")
        t_start = time.time()

        self.file_infos = file_infos
        self.depth = depth
        self.window = window
        self.horizon = horizon
        self.alpha = alpha
        self.stride = stride
        self.current_file_idx = current_file_idx
        
        # Load the current file
        if not file_infos:
            raise ValueError("No files provided to dataset. Did you run normalize_data.py first?")
        
        self._load_current_file()
        
        print(f"Streaming dataset initialized. Current file: {self.file_infos[self.current_file_idx]['filename']}")
        print(f"Total initialization time: {time.time()-t_start:.2f}s")
        print(f"Effective samples: {self.__len__()}")
    
    def _load_current_file(self):
        """Load the current file into memory with selective column loading based on depth"""
        file_info = self.file_infos[self.current_file_idx]
        print(f"Loading file: {file_info['filename']}...")
        t_start = time.time()
        
        # Generate the list of required columns based on the depth parameter
        required_cols = []
        
        # Add all bid/ask columns up to the specified depth
        for side in ['bid', 'ask']:
            for i in range(self.depth):
                required_cols.append(f"{side}_{i}_price")
                required_cols.append(f"{side}_{i}_size")
        
        # We need these columns for the mid-price calculation
        essential_cols = ['bid_0_price', 'ask_0_price']
        
        # Make sure the essential columns are included
        for col in essential_cols:
            if col not in required_cols:
                required_cols.append(col)
        
        print(f"Selectively loading {len(required_cols)} columns based on depth={self.depth}")
        
        # Load only the required columns from the parquet file
        t_load_start = time.time()
        df = cudf.read_parquet(file_info['path'], columns=required_cols)
        print(f"Parquet reading took {time.time()-t_load_start:.4f}s")
        
        # Get feature columns (should match our required_cols, but this is safer)
        t_process_start = time.time()
        pat = rf'(bid|ask)_[0-9]{{1,2}}_(price|size)'
        print(f"Starting column filtering...")
        feat_cols = [c for c in df.columns
                    if re.match(pat, c) and int(c.split('_')[1]) < self.depth]
        t_filter_end = time.time()
        print(f"Column filtering took {t_filter_end-t_process_start:.4f}s, found {len(feat_cols)} feature columns")
        
        # ALTERNATIVE APPROACH: Use NumPy bridge instead of DLPack
        # This avoids the potentially problematic to_dlpack() operation
        print(f"Converting data to CPU then GPU...")
        t_convert_start = time.time()
        
        # Step 1: Convert to numpy (CPU) first
        numpy_data = df[feat_cols].to_pandas().values.T  # Transpose during numpy conversion
        t_numpy_end = time.time()
        print(f"NumPy conversion took {t_numpy_end-t_convert_start:.4f}s")
        
        # Step 2: Create tensor on CPU first then move to GPU
        cpu_tensor = torch.tensor(numpy_data, dtype=torch.float32)
        t_tensor_end = time.time()
        print(f"CPU tensor creation took {t_tensor_end-t_numpy_end:.4f}s")
        
        # Free memory early
        del numpy_data
        
        # Step 3: Reshape on CPU (less memory pressure than on GPU)
        reshaped_tensor = cpu_tensor.reshape(-1, self.depth*4)
        t_reshape_end = time.time()
        print(f"Reshape operation took {t_reshape_end-t_tensor_end:.4f}s")
        
        # Step 4: Transfer to GPU
        self.X = reshaped_tensor.to(device=device, dtype=torch.float16)
        torch.cuda.synchronize()  # Make sure the transfer is complete
        t_gpu_end = time.time()
        print(f"GPU transfer took {t_gpu_end-t_reshape_end:.4f}s")
        
        # Free CPU memory
        del cpu_tensor, reshaped_tensor
        
        # Calculate mid-price
        print(f"Calculating mid-price...")
        t_mid_start = time.time()
        
        # Use the same NumPy approach for mid price
        mid_np = ((df["bid_0_price"] + df["ask_0_price"]) / 2).to_pandas().values
        self.mid = torch.tensor(mid_np, dtype=torch.float32).to(device=device, dtype=torch.float16)
        torch.cuda.synchronize()
        t_mid_end = time.time()
        print(f"Mid-price calculation took {t_mid_end-t_mid_start:.4f}s")
        
        print(f"Total data processing took {time.time()-t_process_start:.4f}s")
        
        # Clear dataframe to free memory
        del df
        
        # Print dataset info
        print(f"File loaded. X shape: {self.X.shape}, mid shape: {self.mid.shape}")
        print(f"Total loading took {time.time()-t_start:.2f}s")
        # Force garbage collection to free memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    def move_to_next_file(self):
        """Move to the next file in the list"""
        if self.current_file_idx < len(self.file_infos) - 1:
            # Free current data
            del self.X
            del self.mid
            torch.cuda.empty_cache()
            
            # Move to next file
            self.current_file_idx += 1
            self._load_current_file()
            return True
        else:
            print("No more files available.")
            return False
    
    def __len__(self):
        """Calculate number of samples considering window size, horizon and stride"""
        return max(0, (len(self.X) - self.window - self.horizon) // self.stride)

    def __getitem__(self, i):
        """Get a single sample: input features and target class"""
        idx = i * self.stride
        j = idx + self.window

        # Use reshape() instead of view() due to non-contiguous memory layout
        x = self.X[idx:j].contiguous().reshape(self.window, self.depth*4)   # (window, features)
        
        # Calculate price change and classify direction (all on GPU)
        r = (self.mid[j+self.horizon-1] - self.mid[j-1]) / self.mid[j-1]
        y = 2 if r > self.alpha else 0 if r < -self.alpha else 1  # 0=down, 1=stable, 2=up
        
        # Keep label on GPU as well
        return x, torch.tensor(y, dtype=torch.long, device=device)


class LobDataset(Dataset):
    """
    Legacy dataset class for backward compatibility.
    This class is kept for compatibility with existing code, but the StreamingLobDataset
    should be used instead for memory-efficient processing.
    """
    def __init__(self, df: cudf.DataFrame,
                 depth: int = 10,          # Number of price levels to use
                 window: int = 100,         # Number of time points to look back
                 horizon: int = 100,        # How far into future to predict
                 alpha: float = 0.002,      # Threshold for price movement classification
                 stride: int = 5):          # Step size between consecutive samples
        
        print("NOTICE: Using legacy LobDataset. Consider using StreamingLobDataset for memory efficiency.")
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

        # Keep data on GPU for better performance instead of transferring to CPU
        print("Converting data to GPU tensors...")
        t1 = time.time()
        
        # Use reshape() instead of view() to avoid memory layout issues
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


def make_streaming_loaders(file_infos,
                         depth: int = 10,
                         window: int = 100,
                         horizon: int = 100,
                         valid_frac: float = 0.1,
                         batch: int = 32):
    """
    Create training and validation data loaders from the first file in the list.
    This function creates loaders for a single file, which can be updated later.
    
    Args:
        file_infos: List of file information dictionaries
        depth: Number of price levels to use
        window: Number of time points to use as input
        horizon: How far into future to predict
        valid_frac: Fraction of data to use for validation
        batch: Batch size for training
        
    Returns:
        train_loader, val_loader, dataset: DataLoader objects and the streaming dataset
    """
    print("Creating streaming data loaders...")
    t_start = time.time()
    
    print("Initializing streaming dataset...")
    streaming_ds = StreamingLobDataset(
                    file_infos,
                    depth=depth,
                    window=window,
                    horizon=horizon,
                    alpha=0.002)

    n = len(streaming_ds)
    print(f"Current file contains {n} samples")
    
    print("Creating train/val split...")
    # Using temporal split instead of random to avoid data leakage
    # Later timepoints are used for validation
    n_val = int(n * valid_frac)
    n_train = n - n_val
    
    train_ds = torch.utils.data.Subset(streaming_ds, range(n_train))
    val_ds = torch.utils.data.Subset(streaming_ds, range(n_train, n))
    
    print(f"Train samples: {n_train}, Validation samples: {n_val}")

    print("Creating data loaders...")
    # num_workers=0 ensures data stays on GPU and prevents transfer issues
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, 
                           num_workers=0, pin_memory=False)
    
    print(f"Data loaders created. Took {time.time()-t_start:.2f}s")
    return train_loader, val_loader, streaming_ds


def make_loaders(df,
                 depth: int = 10,
                 window: int = 100,
                 horizon: int = 100,
                 valid_frac: float = 0.1,
                 batch: int = 32):
    """
    Legacy function for backward compatibility.
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
    print("NOTICE: Using legacy make_loaders. Consider using make_streaming_loaders for memory efficiency.")
    print("Creating data loaders...")
    t_start = time.time()
    
    print("Initializing dataset...")
    ds = LobDataset(df,
                    depth=depth,
                    window=window,
                    horizon=horizon,
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
    """DeepLOB‑style Inception@32 with detailed timing prints.
    Optimized implementation with exact 32-channel output regardless of ratio settings.
    """
    def __init__(self, in_ch: int, out_ch: int = 32,
                 ratio: tuple = (0.25, 0.375, 0.25, 0.125)):
        super().__init__()
        # Calculate each branch's output channels with rounding correction
        raw_channels = [out_ch * r for r in ratio]
        raw_integers = [int(ch) for ch in raw_channels]
        
        # Calculate the total integer channels and the remaining channels to distribute
        total_int = sum(raw_integers)
        remaining = out_ch - total_int
        
        # Distribute the remaining channels based on the fractional parts
        fractional_parts = [ch - int(ch) for ch in raw_channels]
        indices = sorted(range(len(fractional_parts)), key=lambda i: fractional_parts[i], reverse=True)
        
        # Allocate channels, ensuring sum is exactly out_ch
        final_channels = raw_integers.copy()
        for i in range(remaining):
            final_channels[indices[i % len(indices)]] += 1
            
        b1, b3, b5, bp = final_channels
        
        # For debugging
        # print(f"Inception channels: {b1}+{b3}+{b5}+{bp}={sum(final_channels)}")
        
        self.branch1 = nn.Conv1d(in_ch, b1, 1)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, max(1, b3 // 2), 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv1d(max(1, b3 // 2), b3, 3, padding=1)
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_ch, max(1, b5 // 2), 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv1d(max(1, b5 // 2), b5, 5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, bp, 1)
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):               # x: (B, T, C)
        t0 = time.perf_counter()
        x = x.transpose(1, 2).contiguous()
        y = torch.cat((
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch_pool(x)
        ), dim=1)
        y = self.act(y)
        y = y.transpose(1, 2).contiguous()
        
        # Performance timing removed to reduce console output
        pass
        
        return y


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
        self.inception = InceptionModule(in_ch=32, out_ch=32, ratio=(0.2, 0.4, 0.3, 0.1))

        # ——— 3. LSTM with 64 hidden units ———
        self.lstm = nn.LSTM(input_size=32,
                    hidden_size=64,
                    batch_first=True,
                    dropout=0.1)      # 0.1 dropout as in the paper's DeepLOB version

        # Output layer for 3-class classification (down/stable/up)
        self.head = nn.Linear(64, 3)

    def forward(self, x):                  # x: (B, 100, 40) - batch, time, features
        # Add channel dimension for CNN
        x = x.unsqueeze(1)                 # -> (B,1,100,40)
        
        # CNN processing
        x = self.cnn(x)                    # -> (B,32,100,10)
        
        # Reshape for Inception module
        x = x.squeeze(-1).permute(0,2,1)   # -> (B,100,16)
        
        # Inception module
        x = self.inception(x)              # -> (B,100,32)
        
        # LSTM layer
        out, _ = self.lstm(x)              # -> (B,100,64)
        
        # Final prediction from last time step
        y = self.head(out[:,-1])           # utolsó időlépés
        
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
        
        # F1 score számítása minden osztályra külön-külön
        class_f1 = f1_score(all_true.cpu().numpy(), all_pred.cpu().numpy(), average=None)
        
        # Az osztályok f1 értékei: [down, stable, up] (0, 1, 2 osztályok)
        f1_down, f1_stable, f1_up = class_f1
        
        # Az "up" és "down" osztályok F1 értékeinek átlaga - ez lesz az új mérőszám
        directional_f1 = (f1_up + f1_down) / 2
        
        # A sima macro F1-et is kiszámítjuk a diagnosztikához
        cpu_macro_f1 = f1_score(all_true.cpu().numpy(), all_pred.cpu().numpy(), average='macro')
        
        # Az új directional_f1-et használjuk a modell kiválasztásához
        macro_f1 = directional_f1  # A macro_f1 változónév marad a kompatibilitás miatt

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        total_train_time += epoch_time
        
        print(f"Epoch {ep:02d} completed in {epoch_time:.2f}s")
        print(f"Loss: {running_loss/len(train_loader.sampler):.4f}")
        print(f"F1 Scores - Down: {f1_down:.4f}, Stable: {f1_stable:.4f}, Up: {f1_up:.4f}")
        print(f"Directional F1 (Up/Down Avg): {directional_f1:.4f}, Macro F1: {cpu_macro_f1:.4f}")
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


def train_with_streaming(model,
                         file_infos,
                         epochs=40,
                         lr=1e-3,
                         patience_lim=5,
                         accum_steps=2,
                         batch_size=64,
                         depth=10, 
                         window=100, 
                         horizon=100):
    """
    Train the model using streaming data from multiple files.
    This function trains on one file at a time, then moves to the next.
    
    Args:
        model: The DeepLOB model to train
        file_infos: List of file information dictionaries
        epochs: Maximum number of epochs to train
        lr: Learning rate
        patience_lim: Early stopping patience
        accum_steps: Gradient accumulation steps
        batch_size: Batch size
        depth: Number of price levels to use
        window: Number of time points to use as input
        horizon: How far into future to predict
    
    Returns:
        Trained model
    """
    print(f"Starting streaming training with {len(file_infos)} files...")
    print(f"Training parameters: epochs={epochs}, lr={lr}, patience={patience_lim}, batch_size={batch_size}")
    
    # Initialize optimizer and loss function once for all files
    opt = torch.optim.AdamW(model.parameters(),
                          lr=lr, betas=(0.9,0.999),
                          eps=1e-8, weight_decay=1e-4,
                          fused=True)              # ha PyTorch ≥ 2
    scaler = amp.GradScaler()
    ce = nn.CrossEntropyLoss()
    
    # These variables are maintained across all files - for global best model
    best_f1_global = 0.0
    total_train_time = 0
    
    # Iterate over each file
    for file_idx, file_info in enumerate(file_infos):
        print(f"\n=== Training on file {file_idx+1}/{len(file_infos)}: {file_info['filename']} ===\n")
        
        # Reset file-specific variables
        wait = 0                  # Reset patience counter for each file
        best_f1_file = 0.0        # Best F1 for current file
        
        # Create loaders for this file
        t_start = time.time()
        train_loader, val_loader, streaming_ds = make_streaming_loaders(
                [file_info],  # Pass only current file
                depth=depth,
                window=window,
                horizon=horizon,
                batch=batch_size)
        print(f"Loaders created in {time.time()-t_start:.2f}s")
        
        # Clear memory before starting training on this file
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

        # Train on this file
        for ep in range(1, epochs + 1):
            epoch_start = time.time()
            print(f"\n--- File {file_idx+1}/{len(file_infos)}, Epoch {ep}/{epochs} ---")
            
            # Train phase
            model.train()
            running_loss = 0.0
            
            print(f"Training: {len(train_loader)} batches")
            for step, (xb, yb) in enumerate(train_loader):
                if step % 10 == 0:
                    print(f"Training batch {step}/{len(train_loader)}")
                
                xb = xb.contiguous()
                
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(xb)
                    loss = ce(logits, yb) / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                running_loss += loss.item() * accum_steps
            
            # Validation phase
            print("\nStarting validation phase...")
            model.eval()
            y_true, y_pred = [], []
            
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                for i, (xb, yb) in enumerate(val_loader):
                    if i % 20 == 0:
                        print(f"Validation batch {i}/{len(val_loader)}")
                    
                    logits = model(xb.contiguous())
                    batch_preds = logits.argmax(1)
                    
                    y_true.append(yb)
                    y_pred.append(batch_preds)
            
            # Calculate F1 score
            all_true = torch.cat(y_true)
            all_pred = torch.cat(y_pred)
            macro_f1 = f1_score(all_true.cpu().numpy(), all_pred.cpu().numpy(), average='macro')
            
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            
            print(f"Epoch {ep:02d} completed in {epoch_time:.2f}s")
            print(f"loss={running_loss/len(train_loader.sampler):.4f} F1={macro_f1:.4f}")
            print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
            
            # Check for improvement on current file
            file_improved = False
            global_improved = False
            
            if macro_f1 > best_f1_file + 1e-4:
                best_f1_file, wait = macro_f1, 0
                file_improved = True
                print(f"New best F1 for file {file_idx+1}: {best_f1_file:.4f}")
                torch.save(model.state_dict(), f"best_deeplob_file{file_idx+1}.pt")
                
                # Check if this is also the global best model
                if macro_f1 > best_f1_global + 1e-4:
                    best_f1_global = macro_f1
                    global_improved = True
                    print(f"New global best F1: {best_f1_global:.4f}, saving global model...")
                    torch.save(model.state_dict(), "best_deeplob.pt")
            else:
                wait += 1
                print(f"No improvement for {wait} epochs. Best F1 for file: {best_f1_file:.4f}, Global best F1: {best_f1_global:.4f}")
            
            # Generate confusion matrix and reports for improvement
            if file_improved:
                # Calculate confusion matrix for best model
                conf_matrix = torch.zeros(3, 3, dtype=torch.int, device=device)
                for t in range(3):
                    for p in range(3):
                        conf_matrix[t, p] = torch.sum((all_true == t) & (all_pred == p))
                
                # Calculate precision and recall
                precision = torch.zeros(3, device=device)
                recall = torch.zeros(3, device=device)
                
                for i in range(3):
                    precision[i] = conf_matrix[i, i] / conf_matrix[:, i].sum() if conf_matrix[:, i].sum() > 0 else 0
                    recall[i] = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0
                
                f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                # Print detailed statistics
                cm = conf_matrix.cpu().numpy()
                print("Best model confusion matrix:")
                print(cm)
                print(f"Per-class F1 scores: {f1_per_class.cpu().numpy()}")
                
                # Visualization
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Down', 'Stable', 'Up'],
                            yticklabels=['Down', 'Stable', 'Up'])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Best Confusion Matrix - File {file_idx+1}, Epoch {ep} (F1: {macro_f1:.4f})')
                plt.tight_layout()
                
                # Save file-specific confusion matrix
                plt.savefig(f'best_confusion_matrix_file{file_idx+1}.png')
                
                # If this is also a global improvement, save as the global best
                if global_improved:
                    plt.savefig('best_confusion_matrix.png')
                
                # Save detailed statistics
                with open(f'best_classification_report_file{file_idx+1}.txt', 'w') as f:
                    f.write(f"File: {file_info['filename']}\n")
                    f.write(f"Epoch: {ep}\n")
                    f.write(f"F1 Score: {macro_f1:.4f}\n\n")
                    f.write("Confusion Matrix:\n")
                    f.write(str(cm) + "\n\n")
                    
                    class_names = ['Down', 'Stable', 'Up']
                    for i, class_name in enumerate(class_names):
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {precision[i].item():.4f}\n")
                        f.write(f"  Recall: {recall[i].item():.4f}\n")
                        f.write(f"  F1: {f1_per_class[i].item():.4f}\n\n")
            # Early stopping for this file
            if wait >= patience_lim:
                print(f"Early stopping after {ep} epochs for this file.")
                break
        
        # Free memory before moving to next file
        del train_loader
        del val_loader
        del streaming_ds
        torch.cuda.empty_cache()
    
    print(f"\nTraining completed on all files in {total_train_time:.2f}s ({total_train_time/60:.2f} minutes)")
    return model


def main():
    # Basic configuration
    print("\n=== DeepLOB Performance Optimized Version - Streaming Edition ===\n")
    print("Starting data loading process...")
    
    # 1. Find normalized data files
    t_start = time.time()
    file_infos = load_book_chunk(
            dt.datetime(2024, 9, 1),
            dt.datetime(2025, 2, 28),
            "BTC-USDT")
    
    if not file_infos:
        print("No normalized files found. Please run normalize_data.py first.")
        return
    
    print(f"File information loaded in {time.time()-t_start:.2f}s")
    
    # 2. Initialize model
    print("\nInitializing model...")
    t_start = time.time()
    model = DeepLOB(depth=10).to(
                device,
                memory_format=torch.channels_last)
    print(f"Model initialization completed in {time.time()-t_start:.2f}s")
    
    # Skip model compilation
    print("Model compilation skipped (requires C compiler).")
    
    # 3. Train using streaming approach
    print("\nStarting streaming training...")
    train_with_streaming(model, file_infos,
                       epochs=40,
                       lr=1e-3,
                       patience_lim=5,
                       accum_steps=2,
                       batch_size=64,
                       depth=10,
                       window=100,
                       horizon=100)
    
    print("\n=== Training complete! ===")


if __name__ == "__main__":
    main()
