"""
Többprocesszos modell tréning a GPU kihasználtság maximalizálására.
Ez a modul valódi párhuzamosságot biztosít több DeepLOB modell betanításához.
"""
import os, datetime as dt, time
from pathlib import Path
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from deeplob_optimized import (
    DeepLOB, 
    load_book_chunk
)

# Importáljuk az új GPU Dataset modult
from gpu_dataset import process_file_infos

def train_model_process(rank, batch_size, gpu_id, file_paths, gpu_ready_event):
    """
    Külön folyamatban futó modell tréning.
    
    Args:
        rank: A folyamat rangja
        batch_size: Batch méret
        gpu_id: A használandó GPU azonosítója
        file_paths: A feldolgozandó fájlok listája
        gpu_ready_event: Esemény a szinkronizációhoz
    """
    # Várjuk meg, amíg a GPU adatok betöltése megtörténik
    gpu_ready_event.wait()
    
    # Importok
    import torch
    from torch import nn
    from torch.cuda import amp
    import torch.cuda.amp as amp
    
    from deeplob_optimized import DeepLOB
    from gpu_dataset import create_gpu_data_loaders
    
    # GPU konfigurálása
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Process {rank} using device: {device}")
    
    # Optimalizálás
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Mindenképp 64-es batch méretet használunk
    custom_batch_size = batch_size  # Mindig 64-es batch méret, ahogy kérted
    custom_stride = 5 + rank  # Csak a stride változik a különböző adatok látásához
    
    print(f"Process {rank} - Loading data with batch_size={custom_batch_size}, stride={custom_stride}")
    
    # Adatbetöltők létrehozása
    train_loader, val_loader = create_gpu_data_loaders(
        file_paths=file_paths,
        valid_frac=0.1,
        depth=10,
        window=100,
        horizon=100,
        batch_size=custom_batch_size,
        alpha=0.002,
        stride=custom_stride,
        device=device
    )
    
    # Modell inicializálása
    model = DeepLOB(depth=10).to(device)
    
    # Optimalizáló és veszteségfüggvény
    opt = torch.optim.AdamW(model.parameters(),
                           lr=1e-3, betas=(0.9,0.999),
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
    
    # Tréning
    print(f"Process {rank} - Starting training")
    
    # Modell betanítása
    for ep in range(1, 40 + 1):
        epoch_start = time.time()
        
        # Tréning fázis
        model.train()
        running_loss = 0.0
        
        # Megmérjük a batch/forward/backward időket
        batch_times = []
        forward_times = []
        backward_times = []
        
        for step, (xb, yb) in enumerate(train_loader):
            batch_start = time.time()
            
            # Forward pass
            forward_start = time.time()
            with torch.amp.autocast(device_type='cuda'):
                logits = model(xb.contiguous())
                loss = ce(logits, yb) / 4  # akkumulálunk 4 lépést
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)
            
            # Backward pass
            backward_start = time.time()
            scaler.scale(loss).backward()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)
            
            # Gradiens akkumuláció
            if (step + 1) % 4 == 0:  # minden 4. lépésben frissítünk
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            
            running_loss += loss.item() * 4
            
            # Batch idő mérése
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Diagnosztika minden 500. batch után
            if (step + 1) % 500 == 0:
                avg_batch = sum(batch_times[-500:]) / min(500, len(batch_times[-500:]))
                avg_forward = sum(forward_times[-500:]) / min(500, len(forward_times[-500:]))
                avg_backward = sum(backward_times[-500:]) / min(500, len(backward_times[-500:]))
                
                print(f"Process {rank} - Epoch {ep} - Batch {step+1}/{len(train_loader)} - "
                      f"Loss: {running_loss/(step+1):.4f} - "
                      f"Times: Batch={avg_batch:.4f}s, Forward={avg_forward:.4f}s ({avg_forward/avg_batch*100:.1f}%), "
                      f"Backward={avg_backward:.4f}s ({avg_backward/avg_batch*100:.1f}%)")
        
        # Kiírjuk az epoch statisztikát
        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        print(f"Process {rank} - Epoch {ep} completed in {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f}")
        
        # ---------- VALID ----------
        print(f"Process {rank} - Starting validation phase...")
        val_start = time.time()
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for i, (xb, yb) in enumerate(val_loader):
                if i % 20 == 0:
                    print(f"Process {rank} - Validation batch {i}/{len(val_loader)}")
                
                # Az adatok már a GPU-n vannak, csak folytonosság biztosítása szükséges
                logits = model(xb.contiguous())
                loss = ce(logits, yb)
                val_loss += loss.item()
                batch_preds = logits.argmax(1)
                
                # Címkék és predikciók tárolása a későbbi elemzéshez
                y_true.append(yb)
                y_pred.append(batch_preds)
        
        val_end = time.time()
        print(f"Process {rank} - Validation completed in {val_end - val_start:.2f}s")
        
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
        
        # Validációs veszteség átlagolása
        val_loss = val_loss / len(val_loader)
        
        print(f"Process {rank} - Validation - Loss: {val_loss:.4f}")
        print(f"Process {rank} - F1 Scores - Down: {f1_down:.4f}, Stable: {f1_stable:.4f}, Up: {f1_up:.4f}")
        print(f"Process {rank} - Directional F1: {directional_f1:.4f}, Macro F1: {cpu_macro_f1:.4f}")
        
        # Classification report kiírása időnként
        if ep % 10 == 0 or ep == 1:
            report = classification_report(
                all_true.cpu().numpy(), 
                all_pred.cpu().numpy(), 
                target_names=['Down', 'Stable', 'Up'], 
                digits=4
            )
            print(f"Process {rank} - Classification Report:\n{report}")
        
        # ---------- EARLY-STOP ----------
        if macro_f1 > best_f1 + 1e-4:
            best_f1, wait = macro_f1, 0
            print(f"Process {rank} - New best F1: {best_f1:.4f}, saving model...")
            
            # Könyvtár létrehozása, ha nem létezik
            model_dir = Path(f"models")
            model_dir.mkdir(exist_ok=True)
            
            # Modell mentése
            checkpoint_path = Path(f"models/deeplob_mp_rank_{rank}_f1_{best_f1:.4f}.pt")
            
            model_state = {
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
                'epoch': ep,
                'f1': best_f1,
                'directional_f1': directional_f1,
                'macro_f1': cpu_macro_f1
            }
            
            torch.save(model_state, checkpoint_path)
            print(f"Process {rank} - Model saved to {checkpoint_path}")
            
            # Confusion matrix számítása a legjobb modellhez
            print(f"Process {rank} - Calculating confusion matrix for best model...")
            
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
            print(f"Process {rank} - Best model confusion matrix:\n{cm}")
            
            # Confusion matrix vizualizáció és mentés
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Down', 'Stable', 'Up'],
                       yticklabels=['Down', 'Stable', 'Up'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Process {rank} - Best Confusion Matrix (F1: {macro_f1:.4f})')
            plt.tight_layout()
            plt.savefig(f'best_confusion_matrix_rank_{rank}.png')
            print(f"Process {rank} - Saved confusion matrix to 'best_confusion_matrix_rank_{rank}.png'")
            
            # Részletesebb statisztikák mentése szöveges fájlba
            report_path = f"models/classification_report_rank_{rank}.txt"
            with open(report_path, 'w') as f:
                f.write(f"Process: {rank}\n")
                f.write(f"Epoch: {ep}\n")
                f.write(f"F1 Score: {macro_f1:.4f}\n\n")
                f.write(f"Classification Report:\n{report}\n\n")
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
            
            print(f"Process {rank} - Saved detailed metrics to '{report_path}'")
            
        else:
            wait += 1
            print(f"Process {rank} - No improvement for {wait} epochs. Best F1: {best_f1:.4f}")
            
        if wait >= 5:  # Patience = 5
            print(f"Process {rank} - Early stopping after {ep} epochs.")
            break
    
    print(f"Process {rank} - Training completed! Best F1: {best_f1:.4f}")
    return

def main():
    # Alapvető konfiguráció
    print("\n=== DeepLOB MultiProcess Training - True Parallelism ===\n")
    
    # CUDA inicializálása
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires GPU.")
        return
    
    # 1. Normalizált adatfájlok keresése
    t_start = time.time()
    file_infos = load_book_chunk(
            dt.datetime(2024, 11, 1),
            dt.datetime(2025, 2, 28),
            "BTC-USDT")
    
    if not file_infos:
        print("No normalized files found. Please run normalize_data.py first.")
        return
    
    # Átalakítjuk a fájl információkat abszolút útvonalakká
    file_paths = process_file_infos(file_infos)
    print(f"Found {len(file_paths)} files for processing")
    print(f"File information loaded in {time.time()-t_start:.2f}s")
    
    # 2. Multiprocessing inicializálása
    # A Pytorch multiprocessing indítása
    mp.set_start_method('spawn')
    
    # Esemény a GPU adatok betöltésének jelzésére
    gpu_ready_event = mp.Event()
    
    # Folyamatok száma (3 különböző modellt tanítunk párhuzamosan)
    num_processes = 3
    
    # Batch méretek a különböző folyamatokhoz
    batch_size = 64
    
    # Egyetlen GPU-t használunk
    gpu_id = 0
    
    # Folyamatok indítása
    print(f"Starting {num_processes} training processes...")
    processes = []
    
    for rank in range(num_processes):
        p = mp.Process(
            target=train_model_process,
            args=(rank, batch_size, gpu_id, file_paths, gpu_ready_event)
        )
        p.start()
        processes.append(p)
    
    # Jelezzük, hogy a GPU adatok betöltése megtörtént
    print("Setting GPU ready event...")
    gpu_ready_event.set()
    
    # Megvárjuk, amíg minden folyamat befejeződik
    for p in processes:
        p.join()
    
    print("\n=== MultiProcess Training Complete! ===")

if __name__ == "__main__":
    main()
