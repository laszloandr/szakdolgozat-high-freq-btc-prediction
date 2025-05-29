"""
DeepLOB modell tréning teljes GPU módban.
Ez a verzió teljesen a GPU-n dolgozik, elkerülve a CPU-GPU adatátvitel költségeit.
"""
import os, re, datetime as dt, math, time
from pathlib import Path

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

# Optimalizálás a fix bemenetméretű CNN-ekhez
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def train_gpu_only(model,
                   train_loader, 
                   val_loader,
                   epochs=40,
                   lr=1e-3,
                   patience=5,
                   accum_steps=2):
    """
    Modell tréning teljesen GPU-n, elkerülve a CPU-GPU adatátvitelt.
    
    Args:
        model: A betanítandó DeepLOB modell
        train_loader: Tréning DataLoader
        val_loader: Validációs DataLoader
        epochs: Az epoch-ok maximális száma
        lr: Learning rate
        patience: Early stopping türelmi idő
        accum_steps: Gradiens akkumulációs lépések száma
    
    Returns:
        Betanított modell
    """
    print(f"Starting GPU-only training for {epochs} epochs...")
    print(f"Training parameters: lr={lr}, patience={patience}, accum_steps={accum_steps}")
    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")
    
    # Időmérés
    start_time = time.time()
    
    # Optimalizáló és veszteségfüggvény inicializálása
    opt = torch.optim.AdamW(model.parameters(),
                           lr=lr, betas=(0.9,0.999),
                           eps=1e-8, weight_decay=1e-4,
                           fused=True)
    
    # Mixed precision skálázó speciális beállításokkal a jobb GPU kihasználtságért
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
    
    print(f"GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    # Főciklus: epoch-ok
    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {ep}/{epochs} ---")
        
        # ------ Tréning fázis ------
        model.train()
        running_loss = 0.0
        
        # Batch időmérés diagnosztikához
        batch_times = []
        forward_times = []
        backward_times = []
        
        print(f"Training: {len(train_loader)} batches")
        
        # A tréning adatokon végigmegyünk
        for step, (xb, yb) in enumerate(train_loader):
            batch_start = time.time()
            
            if step % 50 == 0:
                print(f"Training batch {step}/{len(train_loader)}")
            
            # Minden már a GPU-n van, csak a folytonosságot biztosítjuk
            xb = xb.contiguous()
            
            # Forward pass - időmérés
            forward_start = time.time()
            with torch.amp.autocast(device_type='cuda'):
                logits = model(xb)
                loss = ce(logits, yb) / accum_steps
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)
            
            # Backward pass - időmérés
            backward_start = time.time()
            scaler.scale(loss).backward()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)
            
            # Gradiens akkumuláció - csak minden accum_steps-edik lépésben frissítünk
            if (step + 1) % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            
            running_loss += loss.item() * accum_steps
            
            # Batch idő mérése
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Diagnosztika minden 50. batch után
            if (step + 1) % 50 == 0:
                avg_batch = sum(batch_times[-50:]) / min(50, len(batch_times[-50:]))
                avg_forward = sum(forward_times[-50:]) / min(50, len(forward_times[-50:]))
                avg_backward = sum(backward_times[-50:]) / min(50, len(backward_times[-50:]))
                
                print(f"Avg times: Batch={avg_batch:.4f}s, Forward={avg_forward:.4f}s ({avg_forward/avg_batch*100:.1f}%), "
                      f"Backward={avg_backward:.4f}s ({avg_backward/avg_batch*100:.1f}%)")
        
        # ------ Validációs fázis ------
        print("\nStarting validation phase...")
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for i, (xb, yb) in enumerate(val_loader):
                if i % 50 == 0:
                    print(f"Validation batch {i}/{len(val_loader)}")
                
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
        
        print(f"Epoch {ep:02d} completed in {epoch_time:.2f}s")
        print(f"loss={running_loss/len(train_loader.sampler):.4f} F1={macro_f1:.4f}")
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        # Ellenőrizzük, hogy javult-e az F1 score
        if macro_f1 > best_f1 + 1e-4:
            best_f1, wait = macro_f1, 0
            print(f"New best F1: {best_f1:.4f}, saving model and calculating confusion matrix...")
            torch.save(model.state_dict(), "best_deeplob_gpu.pt")
            
            # Konfúziós mátrix számítása
            conf_matrix = confusion_matrix(cpu_true, cpu_pred)
            
            # Részletes metrikák számítása a konfúziós mátrixból
            precision = np.zeros(3)
            recall = np.zeros(3)
            
            for i in range(3):
                precision[i] = conf_matrix[i, i] / conf_matrix[:, i].sum() if conf_matrix[:, i].sum() > 0 else 0
                recall[i] = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0
            
            # F1 osztályonként
            f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Statisztikák kiírása
            print("Best model confusion matrix:")
            print(conf_matrix)
            print(f"Per-class F1 scores: {f1_per_class}")
            
            # Konfúziós mátrix vizualizáció
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Down', 'Stable', 'Up'],
                      yticklabels=['Down', 'Stable', 'Up'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Best Confusion Matrix - Epoch {ep} (F1: {macro_f1:.4f})')
            plt.tight_layout()
            plt.savefig('best_confusion_matrix_gpu.png')
            
            # Részletes statisztikák mentése
            with open('best_classification_report_gpu.txt', 'w') as f:
                f.write(f"Epoch: {ep}\n")
                f.write(f"F1 Score: {macro_f1:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(str(conf_matrix) + "\n\n")
                
                class_names = ['Down', 'Stable', 'Up']
                for i, class_name in enumerate(class_names):
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {precision[i]:.4f}\n")
                    f.write(f"  Recall: {recall[i]:.4f}\n")
                    f.write(f"  F1: {f1_per_class[i]:.4f}\n\n")
                
                # Osztályok eloszlása
                class_counts = [np.sum(cpu_true == i) for i in range(3)]
                f.write("Class distribution in validation set:\n")
                for i, class_name in enumerate(class_names):
                    f.write(f"  {class_name}: {class_counts[i]} samples ({class_counts[i]/len(cpu_true)*100:.2f}%)\n")
        else:
            wait += 1
            print(f"No improvement for {wait} epochs. Best F1: {best_f1:.4f}")
        
        # Early stopping
        if wait >= patience:
            print(f"Early stopping after {ep} epochs.")
            break
    
    print(f"\nTraining completed in {total_train_time:.2f}s ({total_train_time/60:.2f} minutes)")
    return model

def main():
    # Alapvető konfiguráció
    print("\n=== DeepLOB Training - GPU-Only Mode ===\n")
    
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
    
    # 2. Modell inicializálása
    print("\nInitializing model...")
    t_start = time.time()
    model = DeepLOB(depth=10).to(
                device,
                memory_format=torch.channels_last)
    print(f"Model initialization completed in {time.time()-t_start:.2f}s")
    
    # 3. GPU adatbetöltők létrehozása
    # Ez a lépés betölti az összes adatot a GPU-ra, és ott tartja
    train_loader, val_loader = create_gpu_data_loaders(
        file_paths=file_paths,
        valid_frac=0.1,
        depth=10,
        window=100,
        horizon=100,
        batch_size=64,    # Eredeti batch méret a túltanulás elkerülése érdekében
        alpha=0.002,
        stride=5,
        device=device
    )
    
    # 4. Modell tréningje
    print("\nStarting GPU-only training...")
    train_gpu_only(
        model,
        train_loader,
        val_loader,
        epochs=40,
        lr=1e-3,
        patience=5,
        accum_steps=4      # Növelt gradiens akkumuláció a jobb GPU kihasználtságért
    )
    
    print("\n=== Training complete! ===")

if __name__ == "__main__":
    main()
