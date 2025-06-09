"""
GPU-gyorsított adatbetöltés a DeepLOB modellhez.
Ez a modul lehetővé teszi az adatok közvetlen betöltését és feldolgozását a GPU-n,
elkerülve a CPU-GPU adatátviteli költségeket.
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
    GPU-n tárolt adatkészlet, amely minimalizálja a CPU-GPU átvitelt.
    Az adatokat egyszer betölti a GPU-ra és ott tárolja a teljes tanítás során.
    
    Előnyök:
    - Nincs szükség CPU-GPU adatátvitelre a tanítás során
    - Sokkal gyorsabb adatbetöltés, különösen több epoch esetén
    - A batch-ek képzése is a GPU-n történik
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
        Inicializálja a GPU-n tárolt adatkészletet.
        
        Args:
            file_paths: A feldolgozandó fájlok listája (teljes elérési út)
            depth: A használandó árszintek száma
            window: Az időablak mérete (timestep)
            horizon: Az előrejelzési horizont
            alpha: A küszöbérték az árváltozás osztályozásához
            stride: A lépés mérete a mintavételezéshez
            device: Az eszköz, amelyen az adatokat tárolni kell (None = aktuális GPU)
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth = depth
        self.window = window
        self.horizon = horizon
        self.alpha = alpha
        self.stride = stride
        
        print(f"Initializing GPU Cached Dataset on {self.device}...")
        
        # Átviteli pufferek inicializálása
        self.X_tensor = None  # A teljes adatkészlet jellemzői
        self.mid_tensor = None  # A teljes adatkészlet középárai
        
        # Betöltjük az összes adatot a GPU-ra (egyesítve)
        self._load_all_files_to_gpu(file_paths)
        
        # Kiszámoljuk a mintavételezési indexeket
        self._prepare_indices()
        
        print(f"GPU Dataset initialized. Features shape: {self.X_tensor.shape}, Mid price shape: {self.mid_tensor.shape}")
        print(f"Total samples: {len(self.sample_indices)}")
        
    def _load_all_files_to_gpu(self, file_paths):
        """Betölti az összes fájlt és egyesíti őket egy nagy GPU tenszorrá."""
        t_start = time.time()
        print(f"Loading {len(file_paths)} files directly to GPU...")
        
        # Adatstruktúrák az adatok összegyűjtéséhez
        all_features = []
        all_mid_prices = []
        
        for i, file_path in enumerate(file_paths):
            print(f"Processing file {i+1}/{len(file_paths)}: {Path(file_path).name}")
            
            # Betöltjük a fájlt pandas-szal (hatékonyabb a parquet fájlokhoz)
            load_start = time.time()
            
            # Oszlopok előkészítése
            required_cols = []
            for side in ['bid', 'ask']:
                for j in range(self.depth):
                    required_cols.append(f"{side}_{j}_price")
                    required_cols.append(f"{side}_{j}_size")
            
            # Ellenőrizzük, hogy szükséges oszlopok mindig benne legyenek
            essential_cols = ['bid_0_price', 'ask_0_price']
            for col in essential_cols:
                if col not in required_cols:
                    required_cols.append(col)
            
            # Pandas-szal betöltjük a parquet fájlt
            try:
                df = pd.read_parquet(file_path, columns=required_cols)
                print(f"File loaded in {time.time() - load_start:.2f}s")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue
            
            # Feature oszlopok kiválasztása
            pat = rf'(bid|ask)_[0-9]{{1,2}}_(price|size)'
            feat_cols = [c for c in df.columns
                        if re.match(pat, c) and int(c.split('_')[1]) < self.depth]
            
            print(f"Converting to tensors and moving to GPU...")
            tensor_start = time.time()
            
            # Jellemzők betöltése numpy-ból PyTorch tensorra
            # A jellemzőket transzponáljuk, hogy a megfelelő formátumban legyenek (idő, jellemzők)
            features = torch.tensor(df[feat_cols].values, dtype=torch.float32)
            
            # Középár számítása
            mid_prices = torch.tensor(((df["bid_0_price"] + df["ask_0_price"]) / 2).values, dtype=torch.float32)
            
            # Az adatokat áthelyezzük a GPU-ra és float16-ra konvertáljuk
            features = features.to(device=self.device, dtype=torch.float16)
            mid_prices = mid_prices.to(device=self.device, dtype=torch.float16)
            
            print(f"Tensor conversion completed in {time.time() - tensor_start:.2f}s")
            print(f"Features shape: {features.shape}, Mid price shape: {mid_prices.shape}")
            
            # Hozzáadjuk a listához
            all_features.append(features)
            all_mid_prices.append(mid_prices)
            
            # Memória felszabadítása
            del df, features, mid_prices
            torch.cuda.empty_cache()
        
        # Egyesítjük az összes adatot
        if all_features:
            print("\nConcatenating all data...")
            cat_start = time.time()
            
            # Egyesítjük a listákat egy-egy nagy GPU tensorrá
            self.X_tensor = torch.cat(all_features, dim=0)
            self.mid_tensor = torch.cat(all_mid_prices, dim=0)
            
            # Memória felszabadítása
            del all_features, all_mid_prices
            torch.cuda.empty_cache()
            
            print(f"Concatenation completed in {time.time() - cat_start:.2f}s")
            print(f"Final shapes - Features: {self.X_tensor.shape}, Mid prices: {self.mid_tensor.shape}")
        else:
            raise ValueError("No data was loaded from the provided files")
        
        print(f"All data loaded to GPU in {time.time() - t_start:.2f}s")
    
    def _prepare_indices(self):
        """Előkészíti a mintavételezési indexeket a window, horizon és stride alapján."""
        print("Preparing sample indices...")
        num_samples = max(0, (len(self.X_tensor) - self.window - self.horizon) // self.stride)
        
        # Mintavételezési indexek létrehozása
        # Minden index egy (start_idx, j_idx) pár, ahol:
        # - start_idx: a minta kezdőpontja
        # - j_idx: az előrejelzési pont (start_idx + window)
        self.sample_indices = []
        
        for i in range(num_samples):
            start_idx = i * self.stride
            j_idx = start_idx + self.window
            self.sample_indices.append((start_idx, j_idx))
        
        print(f"Created {len(self.sample_indices)} sample indices")
    
    def __len__(self):
        """Az adatkészlet mérete (minták száma)."""
        return len(self.sample_indices)
    
    def __getitem__(self, index):
        """
        Visszaad egy mintát az adatkészletből.
        Minden művelet a GPU-n történik, nincs CPU-GPU átvitel.
        """
        # Lekérjük a kezdő- és az előrejelzési indexet
        start_idx, j_idx = self.sample_indices[index]
        
        # Kinyerjük a jellemzőket - ez már a GPU-n van
        x = self.X_tensor[start_idx:j_idx].contiguous().reshape(self.window, self.depth*4)
        
        # Kiszámoljuk az árváltozást és osztályozzuk az irányt (szintén a GPU-n)
        r = (self.mid_tensor[j_idx+self.horizon-1] - self.mid_tensor[j_idx-1]) / self.mid_tensor[j_idx-1]
        y = 2 if r > self.alpha else 0 if r < -self.alpha else 1  # 0=down, 1=stable, 2=up
        
        # A címke is a GPU-n marad
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
    Létrehoz egy GPU-gyorsított DataLoader-t az adatok betöltéséhez.
    
    Args:
        file_paths: A feldolgozandó fájlok listája
        valid_frac: Az adatok hány százaléka kerüljön a validációs halmazba
        depth: A használandó árszintek száma
        window: Az időablak mérete
        horizon: Az előrejelzési horizont
        batch_size: A batch mérete
        alpha: A küszöbérték az árváltozás osztályozásához
        stride: A lépés mérete a mintavételezéshez
        device: Az eszköz, amelyen az adatokat tárolni kell
        
    Returns:
        train_loader, val_loader: Training és validációs DataLoader-ek
    """
    print(f"Creating GPU data loaders with {len(file_paths)} files...")
    t_start = time.time()
    
    # GPU adatkészlet létrehozása
    dataset = GPUCachedDataset(
        file_paths=file_paths,
        depth=depth,
        window=window,
        horizon=horizon,
        alpha=alpha,
        stride=stride,
        device=device
    )
    
    # Az adatok felosztása képzési és validációs halmazra
    # A validációs halmaz az adatok végén van (időrendi sorrend)
    n_samples = len(dataset)
    n_val = int(n_samples * valid_frac)
    n_train = n_samples - n_val
    
    print(f"\nSplitting dataset: {n_train} training samples, {n_val} validation samples")
    
    # Indexek létrehozása - ezúttal a képzési halmaz elején, a validációs halmaz a végén
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_samples))
    
    # DataLoader-ek létrehozása - szükségtelen munkavállalók, mivel minden már a GPU-n van
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=0,  # Nincs szükség worker-ekre, mivel az adatok már a GPU-n vannak
        pin_memory=False  # Nincs szükség pin memory-ra, mivel nincs CPU-GPU átvitel
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
    Átalakítja a file_infos listát az absolute path-ek listájává.
    
    Args:
        file_infos: A fájl információk listája, amit a load_book_chunk függvény ad vissza
        
    Returns:
        file_paths: Az abszolút fájl utak listája
    """
    return [str(info['path']) for info in file_infos]
