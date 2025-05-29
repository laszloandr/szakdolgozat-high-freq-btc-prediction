"""
DeepLOB model training with full period processing in a single epoch.
Streaming data loading but unified training over all files.
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

# Importáljuk az optimalizált LSTM-et
try:
    from optimized_lstm import OptimizedLSTM
    print("Optimized LSTM module imported successfully!")
    USE_OPTIMIZED_LSTM = True
except ImportError as e:
    print(f"Warning: Could not import optimized LSTM: {e}")
    print("Falling back to standard PyTorch LSTM")
    USE_OPTIMIZED_LSTM = False

# Importáljuk a DeepLOB modellt és az adatbetöltő függvényeket
from deeplob_optimized import (
    DeepLOB, 
    find_normalized_files, 
    load_book_chunk
)

# Importáljuk az új StreamingConcatenatedDataset osztályt
from concatenated_dataset import StreamingConcatenatedDataset

# PyTorch és CUDA konfiguráció
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# Optimalizálás a fix bemenetméretű CNN-ekhez
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True    # Engedélyezzük a TF32-t a konvolúciókhoz is

# Beállítunk magasabb intra-op párhuzamosságot
torch.set_num_threads(4)  # Állítsd be a CPU magok számának megfelelően

def train_entire_period(model,
                       file_infos,
                       epochs=40,
                       lr=1e-3,
                       patience=5,
                       accum_steps=2,
                       batch_size=64,
                       depth=10, 
                       window=100, 
                       horizon=100):
    """
    Train the model on the entire period in a single epoch.
    
    Args:
        model: The DeepLOB model to train
        file_infos: List of file information dictionaries
        epochs: Maximum number of epochs to train
        lr: Learning rate
        patience: Early stopping patience
        accum_steps: Gradient accumulation steps
        batch_size: Batch size
        depth: Number of price levels to use
        window: Number of time points to use as input
        horizon: How far into future to predict
    
    Returns:
        Trained model
    """
    print(f"Starting training on entire period with {len(file_infos)} files...")
    print(f"Training parameters: epochs={epochs}, lr={lr}, patience={patience}, batch_size={batch_size}")
    
    # Időmérés
    start_time = time.time()
    
    # Adatkészlet előkészítése
    print("\nPreparing concatenated dataset...")
    dataset = StreamingConcatenatedDataset(
        file_infos=file_infos,
        depth=depth,
        window=window,
        horizon=horizon,
        alpha=0.002,
        stride=5,
        valid_frac=0.1,
        batch_size=batch_size,
        device=device
    )
    
    # DataLoader-ek létrehozása
    train_loader, val_loader = dataset.create_dataloaders()
    
    # Optimalizáló és veszteségfüggvény inicializálása - speciális beállításokkal
    opt = torch.optim.AdamW(model.parameters(),
                          lr=lr, betas=(0.9,0.999),
                          eps=1e-8, weight_decay=1e-4,
                          fused=True)  # A fused=True gyorsabb GPU végrehajtást eredményez
    
    # Jobb mixed precision beállítások - gyorsabb konvergencia és végrehajtás
    # Egységesített formátum, amely minden PyTorch verzióval működik
    scaler = amp.GradScaler(
        growth_factor=2.0,     # Gyorsabb scaling növekedés
        backoff_factor=0.5,   # Konzerváltabb csökkentés
        growth_interval=100   # Gyakoribb skálás
    )
    
    # Keresztentrópia veszteségfüggvény, készülve az automatikus mixedprecision-re
    ce = nn.CrossEntropyLoss()
    
    # Változók inicializálása
    best_f1 = 0.0
    wait = 0
    total_train_time = 0
    
    print(f"\nGPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    # Tanítás epochonként
    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {ep}/{epochs} ---")
        
        # ------ Tanítási fázis ------
        model.train()
        running_loss = 0.0
        
        print(f"Training: {len(train_loader)} batches")
        for step, (xb, yb) in enumerate(train_loader):
            if step % 50 == 0:
                print(f"Training batch {step}/{len(train_loader)}")
                print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
            
            # Folytonosság biztosítása
            xb = xb.contiguous()
            
            # Automatikus mixed precision használata
            with torch.amp.autocast(device_type='cuda'):
                logits = model(xb)
                loss = ce(logits, yb) / accum_steps
            
            # Gradiens számítása és skálázása
            scaler.scale(loss).backward()
            
            # Gradiens akkumuláció - csak minden accum_steps-edik lépésnél frissítünk
            if (step + 1) % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            
            running_loss += loss.item() * accum_steps
        
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
        macro_f1 = f1_score(all_true.cpu().numpy(), all_pred.cpu().numpy(), average='macro')
        
        # Epoch statisztikák
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        
        print(f"Epoch {ep:02d} completed in {epoch_time:.2f}s")
        print(f"loss={running_loss/len(train_loader.dataset):.4f} F1={macro_f1:.4f}")
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        # Ellenőrizzük, hogy javult-e az F1 score
        if macro_f1 > best_f1 + 1e-4:
            best_f1, wait = macro_f1, 0
            print(f"New best F1: {best_f1:.4f}, saving model and calculating confusion matrix...")
            torch.save(model.state_dict(), "best_deeplob_unified.pt")
            
            # Konfúziós mátrix számítása
            conf_matrix = torch.zeros(3, 3, dtype=torch.int, device=device)
            for t in range(3):
                for p in range(3):
                    conf_matrix[t, p] = torch.sum((all_true == t) & (all_pred == p))
            
            # Precision és recall számítása
            precision = torch.zeros(3, device=device)
            recall = torch.zeros(3, device=device)
            
            for i in range(3):
                precision[i] = conf_matrix[i, i] / conf_matrix[:, i].sum() if conf_matrix[:, i].sum() > 0 else 0
                recall[i] = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0
            
            # F1 osztályonként
            f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Statisztikák kiírása
            cm = conf_matrix.cpu().numpy()
            print("Best model confusion matrix:")
            print(cm)
            print(f"Per-class F1 scores: {f1_per_class.cpu().numpy()}")
            
            # Konfúziós mátrix vizualizáció
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Down', 'Stable', 'Up'],
                      yticklabels=['Down', 'Stable', 'Up'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Best Confusion Matrix - Epoch {ep} (F1: {macro_f1:.4f})')
            plt.tight_layout()
            plt.savefig('best_confusion_matrix_unified.png')
            
            # Részletes statisztikák mentése
            with open('best_classification_report_unified.txt', 'w') as f:
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
                
                # Osztályok eloszlása
                class_counts = [torch.sum(all_true == i).item() for i in range(3)]
                f.write("Class distribution in validation set:\n")
                for i, class_name in enumerate(class_names):
                    f.write(f"  {class_name}: {class_counts[i]} samples ({class_counts[i]/len(all_true)*100:.2f}%)\n")
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
    print("\n=== DeepLOB Training - Full Period Processing ===\n")
    
    # 1. Normalizált adatfájlok keresése
    t_start = time.time()
    file_infos = load_book_chunk(
            dt.datetime(2024, 9, 1),
            dt.datetime(2024, 9, 30),
            "BTC-USDT")
    
    if not file_infos:
        print("No normalized files found. Please run normalize_data.py first.")
        return
    
    print(f"File information loaded in {time.time()-t_start:.2f}s")
    
    # 2. Modell inicializálása
    print("\nInitializing model...")
    t_start = time.time()
    model = DeepLOB(depth=10).to(
                device,
                memory_format=torch.channels_last)
    
    # Torch compile használata a modell gyorsításához (PyTorch 2.0+ esetén)
    try:
        print("Compiling model with torch.compile() for acceleration...")
        # Az eredeti modellt elmentjük biztongsági okokból
        original_model = model
        
        # Konfiguráljuk a dynamo-t, hogy elnyomja a hibákat
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.suppress_errors = True
        
        # Modell fordítása opimizált futtatáshoz - biztonságosabb beállításokkal
        model = torch.compile(model, mode='reduce-overhead', backend='eager')
        print("Model compilation successful!")
    except Exception as e:
        print(f"Warning: Could not compile model: {e}")
        print("Continuing with uncompiled model. For better performance, upgrade to PyTorch 2.0+")
        # Visszaállítjuk az eredeti modellt
        model = original_model
    
    print(f"Model initialization completed in {time.time()-t_start:.2f}s")
    
    # 3. Tanítás a teljes időszakra - optimalizált paraméterekkel
    print("\nStarting training on the entire period...")
    train_entire_period(
        model, 
        file_infos,
        epochs=40,
        lr=1e-3,
        patience=5,
        accum_steps=4,          # Növeltük a gradiens akkumulációt a jobb GPU kihasználtságért
        batch_size=64,          # Batch méret változatlan marad (ahogy kérted)
        depth=10,
        window=100,
        horizon=100
    )
    
    print("\n=== Training complete! ===")

if __name__ == "__main__":
    main()
