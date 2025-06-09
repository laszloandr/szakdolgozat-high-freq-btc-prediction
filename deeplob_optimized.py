
import os, re, datetime as dt, time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

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

        f_sd = pd.to_datetime(m.group(1))
        f_ed = pd.to_datetime(m.group(2))
        
        # Check if file overlaps with requested time period
        if f_ed < sd or f_sd > ed:  # completely outside our date range
            continue
            
        # Add file info with actual time range
        matching_files.append({
            'path': fp,
            'start_date': max(f_sd, sd),  # Use the later of file start and requested start
            'end_date': min(f_ed, ed),    # Use the earlier of file end and requested end
            'filename': fn
        })
    
    # Sort files chronologically
    matching_files.sort(key=lambda x: x['start_date'])
    
    if not matching_files:
        print(f"No normalized files found for {symbol} between {start_date} and {end_date}")
    else:
        print(f"Found {len(matching_files)} normalized files:")
        for f in matching_files:
            print(f"  {f['filename']}: {f['start_date']} to {f['end_date']}")
    
    return matching_files


def load_book_chunk(
    start_date: dt.datetime,
    end_date: dt.datetime,
    symbol: str,
    data_dir: str = "./szakdolgozat-high-freq-btc-prediction/data_normalized",
) -> list:
    """
    Load normalized book data for the specified time period.
    
    Args:
        start_date: Beginning date for data loading
        end_date: End date for data loading
        symbol: Trading pair symbol (e.g., "BTC-USDT")
        data_dir: Directory containing normalized parquet files
        
    Returns:
        List of file info dictionaries with paths and time ranges
    """
    # Find all matching normalized files
    file_infos = find_normalized_files(start_date, end_date, symbol, data_dir)
    
    if not file_infos:
        print("No normalized files found for the specified time period.")
        return []
    
    return file_infos

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