"""
Párhuzamos modell tréning a GPU kihasználtság maximalizálására.
Ez a modul több DeepLOB modellt treníroz párhuzamosan, hogy jobban kihasználja a GPU erőforrásokat.
"""
import os, re, datetime as dt, math, time
from pathlib import Path
import threading
import concurrent.futures

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda import amp
import torch.cuda.amp as amp
from sklearn.metrics import f1_score, confusion_matrix

# Importáljuk a DeepLOB modellt és az adatbetöltő függvényeket
from deeplob_optimized import (
    DeepLOB, 
    find_normalized_files, 
    load_book_chunk
)

# Importáljuk az új GPU Dataset modult
from gpu_dataset import create_gpu_data_loaders, process_file_infos

# PyTorch és CUDA konfiguráció
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# Optimalizálás
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class ParallelTrainer:
    """
    Párhuzamos modell tréner, amely több modellt képez párhuzamosan a GPU-n.
    Ez maximalizálja a GPU kihasználtságot anélkül, hogy változtatnunk kellene
    a modell architektúráján vagy a batch méreten.
    """
    def __init__(self, 
                 file_paths,
                 num_models=3,
                 depth=10,
                 window=100,
                 horizon=100,
                 batch_size=64,
                 alpha=0.002,
                 stride=5,
                 epochs=40,
                 lr=1e-3,
                 patience=5,
                 accum_steps=4):
        """
        Inicializálja a párhuzamos trénert.
        
        Args:
            file_paths: A feldolgozandó fájlok listája
            num_models: Párhuzamos modellek száma
            depth: Árszintek száma
            window: Időablak mérete
            horizon: Előrejelzési horizont
            batch_size: Batch méret
            alpha: Küszöbérték az árváltozás osztályozásához
            stride: Lépés mérete a mintavételezéshez
            epochs: Maximális epoch-ok száma
            lr: Tanulási ráta
            patience: Early stopping türelmi idő
            accum_steps: Gradiens akkumulációs lépések száma
        """
        self.file_paths = file_paths
        self.num_models = num_models
        self.depth = depth
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.alpha = alpha
        self.stride = stride
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.accum_steps = accum_steps
        
        # Inicializáljuk a modelleket
        self.models = []
        for i in range(num_models):
            print(f"Initializing model {i+1}/{num_models}...")
            model = DeepLOB(depth=depth).to(device, memory_format=torch.channels_last)
            self.models.append(model)
        
        # GPU adatbetöltők létrehozása
        print("Creating GPU dataloaders...")
        self.train_loaders = []
        self.val_loaders = []
        
        # Eltérő mintavételezést használunk minden modellhez a diverzitás érdekében
        for i in range(num_models):
            custom_stride = stride + i  # Minden modellhez kicsit eltérő stride
            train_loader, val_loader = create_gpu_data_loaders(
                file_paths=file_paths,
                valid_frac=0.1,
                depth=depth,
                window=window,
                horizon=horizon,
                batch_size=batch_size,
                alpha=alpha,
                stride=custom_stride,
                device=device
            )
            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)
    
    def train_model(self, model_idx):
        """
        Betanít egy modellt a megadott indexen.
        
        Args:
            model_idx: A betanítandó modell indexe
        
        Returns:
            Betanított modell és az F1 score
        """
        model = self.models[model_idx]
        train_loader = self.train_loaders[model_idx]
        val_loader = self.val_loaders[model_idx]
        
        print(f"\n=== Training Model {model_idx+1}/{self.num_models} ===")
        
        # Optimalizáló és veszteségfüggvény
        opt = torch.optim.AdamW(model.parameters(),
                               lr=self.lr, betas=(0.9,0.999),
                               eps=1e-8, weight_decay=1e-4,
                               fused=True)
        
        # Mixed precision skálázó
        scaler = amp.GradScaler(
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=True
        )
        
        # Veszteségfüggvény
        ce = nn.CrossEntropyLoss()
        
        # Változók inicializálása
        best_f1 = 0.0
        wait = 0
        total_train_time = 0
        
        # Főciklus: epoch-ok
        for ep in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # ------ Tréning fázis ------
            model.train()
            running_loss = 0.0
            
            # A tréning adatokon végigmegyünk
            for step, (xb, yb) in enumerate(train_loader):
                # Minden már a GPU-n van, csak a folytonosságot biztosítjuk
                xb = xb.contiguous()
                
                # Forward pass
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(xb)
                    loss = ce(logits, yb) / self.accum_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradiens akkumuláció
                if (step + 1) % self.accum_steps == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                
                running_loss += loss.item() * self.accum_steps
                
                # Állapot kiírása minden 1000. lépésben
                if step % 1000 == 0:
                    print(f"Model {model_idx+1} - Epoch {ep}/{self.epochs} - Batch {step}/{len(train_loader)} - Loss: {running_loss/(step+1):.4f}")
            
            # ------ Validációs fázis ------
            model.eval()
            y_true, y_pred = [], []
            
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                for xb, yb in val_loader:
                    # Előrejelzés
                    logits = model(xb.contiguous())
                    batch_preds = logits.argmax(1)
                    
                    # Eredmények gyűjtése
                    y_true.append(yb)
                    y_pred.append(batch_preds)
            
            # F1 score számítása
            all_true = torch.cat(y_true)
            all_pred = torch.cat(y_pred)
            
            # CPU-ra kell másolni a scikit-learn függvényhez
            cpu_true = all_true.cpu().numpy()
            cpu_pred = all_pred.cpu().numpy()
            macro_f1 = f1_score(cpu_true, cpu_pred, average='macro')
            
            # Epoch statisztikák
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            
            print(f"Model {model_idx+1} - Epoch {ep:02d} completed in {epoch_time:.2f}s")
            print(f"loss={running_loss/len(train_loader.sampler):.4f} F1={macro_f1:.4f}")
            
            # Ellenőrizzük, hogy javult-e az F1 score
            if macro_f1 > best_f1 + 1e-4:
                best_f1, wait = macro_f1, 0
                print(f"Model {model_idx+1} - New best F1: {best_f1:.4f}, saving model")
                torch.save(model.state_dict(), f"best_deeplob_model_{model_idx+1}.pt")
            else:
                wait += 1
                print(f"Model {model_idx+1} - No improvement for {wait} epochs. Best F1: {best_f1:.4f}")
            
            # Early stopping
            if wait >= self.patience:
                print(f"Model {model_idx+1} - Early stopping after {ep} epochs.")
                break
        
        print(f"\nModel {model_idx+1} training completed in {total_train_time:.2f}s ({total_train_time/60:.2f} minutes)")
        return model, best_f1
    
    def train_all_models(self):
        """
        Betanítja az összes modellt párhuzamosan.
        
        Returns:
            Lista a betanított modellekről és az F1 score-okról
        """
        print(f"\n=== Starting Parallel Training of {self.num_models} Models ===\n")
        start_time = time.time()
        
        # Párhuzamos végrehajtás ThreadPoolExecutor-ral
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_models) as executor:
            # Összes modell párhuzamos betanítása
            future_to_idx = {executor.submit(self.train_model, i): i for i in range(self.num_models)}
            
            # Eredmények összegyűjtése
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    model, f1 = future.result()
                    results.append((model, f1))
                    print(f"Model {idx+1} training completed with F1 score: {f1:.4f}")
                except Exception as e:
                    print(f"Model {idx+1} training failed: {e}")
        
        total_time = time.time() - start_time
        print(f"\n=== Parallel Training Completed in {total_time:.2f}s ({total_time/60:.2f} minutes) ===")
        
        # A legjobb modell kiválasztása
        best_idx = max(range(len(results)), key=lambda i: results[i][1])
        best_model, best_f1 = results[best_idx]
        print(f"\nBest model: Model {best_idx+1} with F1 score: {best_f1:.4f}")
        
        return results
    
    def ensemble_predict(self, x):
        """
        Ensemble előrejelzés az összes betanított modell alapján.
        
        Args:
            x: Bemeneti adat
            
        Returns:
            Az ensemble előrejelzés (többségi szavazás)
        """
        predictions = []
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for model in self.models:
                logits = model(x.contiguous())
                preds = logits.argmax(1)
                predictions.append(preds)
        
        # Stack-eljük a predikciókat és kiszámoljuk a móduszt (többségi szavazás)
        stacked_preds = torch.stack(predictions)
        ensemble_pred = torch.mode(stacked_preds, dim=0).values
        
        return ensemble_pred

def main():
    # Alapvető konfiguráció
    print("\n=== DeepLOB Parallel Training - GPU Utilization Maximizer ===\n")
    
    # 1. Normalizált adatfájlok keresése
    t_start = time.time()
    file_infos = load_book_chunk(
            dt.datetime(2024, 9, 1),
            dt.datetime(2024, 9, 30),
            "BTC-USDT")
    
    if not file_infos:
        print("No normalized files found. Please run normalize_data.py first.")
        return
    
    # Átalakítjuk a fájl információkat abszolút útvonalakká
    file_paths = process_file_infos(file_infos)
    print(f"Found {len(file_paths)} files for processing")
    print(f"File information loaded in {time.time()-t_start:.2f}s")
    
    # 2. Párhuzamos tréner inicializálása
    print("\nInitializing parallel trainer...")
    t_start = time.time()
    
    # 3 modell párhuzamos betanítása a GPU kihasználtság maximalizálására
    trainer = ParallelTrainer(
        file_paths=file_paths,
        num_models=3,        # Ennyi modellt képzünk párhuzamosan
        depth=10,
        window=100,
        horizon=100,
        batch_size=64,       # Az eredeti batch méret
        alpha=0.002,
        stride=5,
        epochs=40,
        lr=1e-3,
        patience=5,
        accum_steps=4        # Gradiens akkumuláció
    )
    
    print(f"Parallel trainer initialization completed in {time.time()-t_start:.2f}s")
    
    # 3. Modellek párhuzamos betanítása
    results = trainer.train_all_models()
    
    print("\n=== Parallel Training Complete! ===")
    
    # 4. Ensemble modell mentése
    torch.save({
        'model_0': trainer.models[0].state_dict(),
        'model_1': trainer.models[1].state_dict(),
        'model_2': trainer.models[2].state_dict(),
        'f1_scores': [f1 for _, f1 in results]
    }, "ensemble_deeplob_models.pt")
    
    print("Ensemble model saved to ensemble_deeplob_models.pt")
    
    return trainer

if __name__ == "__main__":
    main()
