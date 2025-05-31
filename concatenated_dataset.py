"""
StreamingConcatenatedDataset for DeepLOB - hatékony memóriahasználat mellett
lehetővé teszi, hogy a modell a teljes időszakot lássa egy epoch alatt.
"""
import torch
from torch.utils.data import Dataset, ConcatDataset
import time
from pathlib import Path
import cudf
import re
import numpy as np
import pandas as pd

class CachedDataset(Dataset):
    """
    Egyszerű adatkészlet, amely már előre kiszámított adatokat tárol.
    """
    def __init__(self, X, mid, window, horizon, alpha, stride):
        self.X = X
        self.mid = mid
        self.window = window
        self.horizon = horizon
        self.alpha = alpha
        self.stride = stride
        
    def __len__(self):
        """Calculate number of samples considering window size, horizon and stride"""
        return max(0, (len(self.X) - self.window - self.horizon) // self.stride)

    def __getitem__(self, i):
        """Get a single sample: input features and target class"""
        idx = i * self.stride
        j = idx + self.window

        # Use reshape() instead of view() due to non-contiguous memory layout
        x = self.X[idx:j].contiguous()
        
        # Calculate price change and classify direction (all on GPU)
        r = (self.mid[j+self.horizon-1] - self.mid[j-1]) / self.mid[j-1]
        y = 2 if r > self.alpha else 0 if r < -self.alpha else 1  # 0=down, 1=stable, 2=up
        
        # Keep label on GPU as well
        return x, torch.tensor(y, dtype=torch.long, device=self.X.device)


class StreamingConcatenatedDataset:
    """
    Ez az osztály streamingelve tölti be a fájlokat és egyesíti őket egy nagy ConcatDataset-be,
    így a modell egy epoch alatt a teljes időszakot láthatja, miközben a memóriahasználat 
    továbbra is hatékony marad.
    """
    def __init__(self, file_infos, depth=10, window=100, horizon=100, alpha=0.002, stride=5, 
                 valid_frac=0.1, batch_size=32, device=None):
        """
        Inicializálja a StreamingConcatenatedDataset-et.
        
        Args:
            file_infos: Fájl információk listája
            depth: Árszintek száma
            window: Időablakok száma
            horizon: Előrejelzési horizont
            alpha: Küszöbérték a változások osztályozásához
            stride: Lépés mérete
            valid_frac: Validációs adatok aránya
            batch_size: Batch méret
            device: Eszköz, amelyen az adatok feldolgozásra kerülnek
        """
        self.file_infos = file_infos
        self.depth = depth
        self.window = window
        self.horizon = horizon
        self.alpha = alpha
        self.stride = stride
        self.valid_frac = valid_frac
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_datasets = []
        self.val_datasets = []
        self.train_indices = []
        self.val_indices = []
        self.file_boundaries = [0]  # Track where each file's data starts
        
        print(f"Initializing StreamingConcatenatedDataset with {len(file_infos)} files...")
        self._prepare_all_files()
        
    def _prepare_all_files(self):
        """Betölti az összes fájlt és előkészíti a DataLoader-eket"""
        start_time = time.time()
        total_samples = 0
        all_datasets = []
        self.timestamps = []  # Eltároljuk a fájlok időpont adatait
        
        # Először összegyűjtjük az összes adatkészletet és a teljes mintaszámot
        for file_idx, file_info in enumerate(self.file_infos):
            print(f"\nPreparing file {file_idx+1}/{len(self.file_infos)}: {file_info['filename']}...")
            
            # Fájl betöltése és tensorrá alakítása (a StreamingLobDataset logikája alapján)
            X, mid, timestamps = self._load_file_to_tensor(file_info['path'])
            
            # Kiszámoljuk a mintavételezések számát az aktuális fájlra
            file_samples = max(0, (len(X) - self.window - self.horizon) // self.stride)
            print(f"File contains {file_samples} samples after windowing")
            print(f"File timestamp range: {timestamps[0]} to {timestamps[-1]}")
            
            # Adatkészlet létrehozása és tárolása
            dataset = CachedDataset(X, mid, self.window, self.horizon, self.alpha, self.stride)
            all_datasets.append(dataset)
            
            # Minden mintaponthoz kiszámoljuk az időbélyeget és tároljuk
            for i in range(0, file_samples):
                idx = i * self.stride + self.window  # Az előrejelzés kezdőpontja
                if idx < len(timestamps):
                    self.timestamps.append(timestamps[idx])
                    
            # Frissítjük a határokat a következő fájlhoz és a teljes mintaszámot
            self.file_boundaries.append(self.file_boundaries[-1] + file_samples)
            total_samples += file_samples
            
            # Memória felszabadítása
            torch.cuda.empty_cache()
        
        # Most, hogy tudjuk a teljes mintaszámot, felosztjuk az adatokat train és validation halmazra
        # A teljes adathalmaz végéről vesszük a validációs mintákat
        n_val = int(total_samples * self.valid_frac)
        n_train = total_samples - n_val
        
        print(f"\nTotal samples across all files: {total_samples}")
        print(f"Training samples: {n_train}, Validation samples: {n_val}")
        
        # Minden adat train, kivéve az utolsó n_val minta
        self.train_indices = list(range(n_train))
        self.val_indices = list(range(n_train, total_samples))
        
        # Kiírjuk a train/validation határidőpontot, ha elég timestamp van
        if len(self.timestamps) >= n_train:
            print(f"\n=== TRAIN/VALIDATION SPLIT TIMESTAMP ====")
            print(f"Training ends at timestamp: {self.timestamps[n_train-1]}")
            print(f"Validation begins at timestamp: {self.timestamps[n_train]}")
            print("========================================\n")
        
        # Adatkészletek hozzáadása a listákhoz
        self.train_datasets = all_datasets
        self.val_datasets = all_datasets
        
        print(f"\nAll files prepared in {time.time() - start_time:.2f}s")
        print(f"Total training samples: {n_train}")
        print(f"Total validation samples: {n_val}")
        
    def _load_file_to_tensor(self, filepath):
        """Betölti a fájlt és tensorrá alakítja hatékony módon"""
        load_start = time.time()
        print(f"Loading file: {Path(filepath).name}...")
        
        # A szükséges oszlopok generálása a depth paraméter alapján
        required_cols = []
        for side in ['bid', 'ask']:
            for i in range(self.depth):
                required_cols.append(f"{side}_{i}_price")
                required_cols.append(f"{side}_{i}_size")
        
        # Alapvető oszlopok a mid price számításához és az időbélyeghez
        # A parquet fájlokban csak 'received_time' oszlop van
        essential_cols = ['bid_0_price', 'ask_0_price', 'received_time']
        for col in essential_cols:
            if col not in required_cols:
                required_cols.append(col)
        
        print(f"Selectively loading {len(required_cols)} columns...")
        
        # Csak a szükséges oszlopok betöltése
        df = cudf.read_parquet(filepath, columns=required_cols)
        
        # Feature oszlopok kiválasztása
        pat = rf'(bid|ask)_[0-9]{{1,2}}_(price|size)'
        feat_cols = [c for c in df.columns
                    if re.match(pat, c) and int(c.split('_')[1]) < self.depth]
        
        print(f"Converting data to CPU then GPU...")
        
        # 1. lépés: Konvertálás numpy-ra (CPU)
        numpy_data = df[feat_cols].to_pandas().values.T
        
        # Időbélyegek kibontása és kezelése - most már tudjuk, hogy 'received_time' oszlopot használunk
        timestamps = None
        try:
            # Csak a received_time oszlopot használjuk
            if 'received_time' in df.columns:
                print(f"Found 'received_time' column, using for timestamps")
                received_times = df['received_time'].to_pandas()
                
                # Biztonságosabb konverzió pd.Series-en keresztül, a datetime64 kezelésére
                timestamps = list(received_times)
                
                # Diagnosztikai információk kiírása
                if len(timestamps) > 0:
                    print(f"Successfully extracted {len(timestamps)} timestamps")
                    print(f"First timestamp: {timestamps[0]}, Last timestamp: {timestamps[-1]}")
            else:
                print("'received_time' column not found in the data!")
                raise ValueError("Missing required 'received_time' column")
        except Exception as e:
            print(f"Warning: Couldn't extract timestamps: {e}")
            timestamps = None
        
        # Ha nem sikerült kibontani a timestamp-eket, generálunk helyette sorszámokat
        if timestamps is None or len(timestamps) == 0:
            print("No timestamps found, using sequence numbers instead.")
            timestamps = list(range(len(df)))
        
        # 2. lépés: Tensor létrehozása a CPU-n, majd áthelyezés a GPU-ra
        cpu_tensor = torch.tensor(numpy_data, dtype=torch.float32)
        
        # Memória felszabadítása
        del numpy_data
        
        # 3. lépés: Átformálás a CPU-n
        reshaped_tensor = cpu_tensor.reshape(-1, self.depth*4)
        
        # 4. lépés: Áthelyezés a GPU-ra
        X = reshaped_tensor.to(device=self.device, dtype=torch.float16)
        
        # Memória felszabadítása
        del cpu_tensor, reshaped_tensor
        
        # Mid price számítása
        print(f"Calculating mid-price...")
        mid_np = ((df["bid_0_price"] + df["ask_0_price"]) / 2).to_pandas().values
        mid = torch.tensor(mid_np, dtype=torch.float32).to(device=self.device, dtype=torch.float16)
        
        # Memória felszabadítása
        del df, mid_np
        
        print(f"File loaded and processed in {time.time() - load_start:.2f}s")
        print(f"X shape: {X.shape}, mid shape: {mid.shape}, timestamps length: {len(timestamps)}")
        
        # Memória kényszerített felszabadítása
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return X, mid, timestamps
        
    def create_dataloaders(self):
        """Létrehozza a teljes train és validation DataLoader-eket"""
        print("Creating combined DataLoaders...")
        
        # Az összes adatkészlet egyesítése
        train_dataset = ConcatDataset(self.train_datasets)
        val_dataset = ConcatDataset(self.val_datasets)
        
        # Adatbetöltők létrehozása
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(self.train_indices),
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(self.val_indices),
            num_workers=0,
            pin_memory=False
        )
        
        print(f"Train loader created with {len(self.train_indices)} samples")
        print(f"Validation loader created with {len(self.val_indices)} samples")
        
        return train_loader, val_loader
