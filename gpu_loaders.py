"""
GPU-accelerated data loading for the DeepLOB model.
This module enables direct loading and processing of data on the GPU,
avoiding CPU-GPU data transfer costs during training.
"""
import os
import re
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class GPUCachedDataset(Dataset):
    """
    GPU-cached dataset that minimizes CPU-GPU transfers.
    Loads data to GPU once and stores it there throughout the entire training process.
    
    Benefits:
    - No need for CPU-GPU data transfer during training
    - Much faster data loading, especially with multiple epochs
    - Batch creation also happens directly on the GPU
    """
    def __init__(self, 
                 file_paths, 
                 depth=10, 
                 window=100, 
                 horizon=100, 
                 alpha=0.002, 
                 stride=5,
                 device=None):
        """
        Initialize the GPU-cached dataset.
        
        Args:
            file_paths: List of files to process (full paths)
            depth: Number of price levels to use from the order book
            window: Size of the time window (timesteps)
            horizon: Prediction horizon
            alpha: Threshold value for price movement classification
            stride: Step size for sampling
            device: Device on which to store the data (None = current GPU)
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth = depth
        self.window = window
        self.horizon = horizon
        self.alpha = alpha
        self.stride = stride
        
        print(f"Initializing GPU Cached Dataset on {self.device}...")
        
        # Initialize transfer buffers
        self.X_tensor = None  # Features of the entire dataset
        self.mid_tensor = None  # Mid-prices of the entire dataset
        
        # Load all data to GPU (combined)
        self._load_all_files_to_gpu(file_paths)
        
        # Calculate sampling indices
        self._prepare_indices()
        
        print(f"GPU Dataset initialized. Features shape: {self.X_tensor.shape}, Mid price shape: {self.mid_tensor.shape}")
        print(f"Total samples: {len(self.sample_indices)}")
        
    def _load_all_files_to_gpu(self, file_paths):
        """Load all files and combine them into a large GPU tensor."""
        t_start = time.time()
        print(f"Loading {len(file_paths)} files directly to GPU...")
        
        # Data structures for collecting data
        all_features = []
        all_mid_prices = []
        
        for i, file_path in enumerate(file_paths):
            print(f"Processing file {i+1}/{len(file_paths)}: {Path(file_path).name}")
            
            # Load file using pandas (more efficient for parquet files)
            load_start = time.time()
            
            # Prepare columns
            required_cols = []
            for side in ['bid', 'ask']:
                for j in range(self.depth):
                    required_cols.append(f"{side}_{j}_price")
                    required_cols.append(f"{side}_{j}_size")
            
            # Ensure that essential columns are always included
            essential_cols = ['bid_0_price', 'ask_0_price']
            for col in essential_cols:
                if col not in required_cols:
                    required_cols.append(col)
            
            # Load the parquet file with pandas
            try:
                df = pd.read_parquet(file_path, columns=required_cols)
                print(f"File loaded in {time.time() - load_start:.2f}s")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue
            
            # Select feature columns
            pat = rf'(bid|ask)_[0-9]{{1,2}}_(price|size)'
            feat_cols = [c for c in df.columns
                        if re.match(pat, c) and int(c.split('_')[1]) < self.depth]
            
            print(f"Converting to tensors and moving to GPU...")
            tensor_start = time.time()
            
            # Load features from numpy to PyTorch tensor
            # The features are arranged in the format (time, features)
            features = torch.tensor(df[feat_cols].values, dtype=torch.float32)
            
            # Calculate mid-price
            mid_prices = torch.tensor(((df["bid_0_price"] + df["ask_0_price"]) / 2).values, dtype=torch.float32)
            
            # Move data to GPU and convert to float16 for efficiency
            features = features.to(device=self.device, dtype=torch.float16)
            mid_prices = mid_prices.to(device=self.device, dtype=torch.float16)
            
            print(f"Tensor conversion completed in {time.time() - tensor_start:.2f}s")
            print(f"Features shape: {features.shape}, Mid price shape: {mid_prices.shape}")
            
            # Add to the feature lists
            all_features.append(features)
            all_mid_prices.append(mid_prices)
            
            # Free memory
            del df, features, mid_prices
            torch.cuda.empty_cache()
        
        # Combine all data
        if all_features:
            print("\nConcatenating all data...")
            cat_start = time.time()
            
            # Combine the lists into large GPU tensors
            self.X_tensor = torch.cat(all_features, dim=0)
            self.mid_tensor = torch.cat(all_mid_prices, dim=0)
            
            # Free memory
            del all_features, all_mid_prices
            torch.cuda.empty_cache()
            
            print(f"Concatenation completed in {time.time() - cat_start:.2f}s")
            print(f"Final shapes - Features: {self.X_tensor.shape}, Mid prices: {self.mid_tensor.shape}")
        else:
            raise ValueError("No data was loaded from the provided files")
        
        print(f"All data loaded to GPU in {time.time() - t_start:.2f}s")
    
    def _prepare_indices(self):
        """Prepare sampling indices based on window, horizon and stride parameters."""
        print("Preparing sample indices...")
        num_samples = max(0, (len(self.X_tensor) - self.window - self.horizon) // self.stride)
        
        # Create sampling indices
        # Each index is a (start_idx, j_idx) pair, where:
        # - start_idx: the starting point of the sample
        # - j_idx: the prediction point (start_idx + window)
        self.sample_indices = []
        
        for i in range(num_samples):
            start_idx = i * self.stride
            j_idx = start_idx + self.window
            self.sample_indices.append((start_idx, j_idx))
        
        print(f"Created {len(self.sample_indices)} sample indices")
    
    def __len__(self):
        """The size of the dataset (number of samples)."""
        return len(self.sample_indices)
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        All operations happen on the GPU, with no CPU-GPU transfer.
        """
        # Get the start and prediction indices
        start_idx, j_idx = self.sample_indices[index]
        
        # Extract features - already on the GPU
        x = self.X_tensor[start_idx:j_idx].contiguous().reshape(self.window, self.depth*4)
        
        # Calculate price change and classify direction (also on GPU)
        r = (self.mid_tensor[j_idx+self.horizon-1] - self.mid_tensor[j_idx-1]) / self.mid_tensor[j_idx-1]
        y = 2 if r > self.alpha else 0 if r < -self.alpha else 1  # 0=down, 1=stable, 2=up
        
        # The label also stays on the GPU
        return x, torch.tensor(y, dtype=torch.long, device=self.device)

def create_gpu_data_loaders(file_paths, 
                           valid_frac=0.1,
                           depth=10,
                           window=100,
                           horizon=100,
                           batch_size=64,
                           alpha=0.002,
                           stride=5,
                           device=None):
    """
    Create a GPU-accelerated DataLoader for loading data efficiently.
    
    Args:
        file_paths: List of files to process
        valid_frac: Fraction of data to use for validation
        depth: Number of price levels to use from the order book
        window: Size of the time window (timesteps)
        horizon: Prediction horizon
        batch_size: Size of each batch
        alpha: Threshold value for price movement classification
        stride: Step size for sampling
        device: Device on which to store the data
        
    Returns:
        train_loader, val_loader: Training and validation DataLoaders
    """
    print(f"Creating GPU data loaders with {len(file_paths)} files...")
    t_start = time.time()
    
    # Create GPU dataset
    dataset = GPUCachedDataset(
        file_paths=file_paths,
        depth=depth,
        window=window,
        horizon=horizon,
        alpha=alpha,
        stride=stride,
        device=device
    )
    
    # Split data into training and validation sets
    # Validation set is at the end of the data (chronological order)
    n_samples = len(dataset)
    n_val = int(n_samples * valid_frac)
    n_train = n_samples - n_val
    
    print(f"\nSplitting dataset: {n_train} training samples, {n_val} validation samples")
    
    # Create indices - training set at the beginning, validation set at the end
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_samples))
    
    # Create DataLoaders - no need for workers since everything is already on GPU
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=0,  # No need for workers since data is already on GPU
        pin_memory=False  # No need for pin memory since there's no CPU-GPU transfer
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=0,
        pin_memory=False
    )
    
    print(f"GPU DataLoaders created in {time.time() - t_start:.2f}s")
    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader

def process_file_infos(file_infos):
    """
    Converts a file_infos list into a list of absolute paths.
    
    Args:
        file_infos: List of file information returned by the load_book_chunk function
        
    Returns:
        file_paths: List of absolute file paths
    """
    return [str(info['path']) for info in file_infos]
