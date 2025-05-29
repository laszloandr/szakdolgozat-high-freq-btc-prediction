"""
Single Model Parallelism - Egy modell tréningje több GPU részlet párhuzamos használatával.
Ez a modul egyetlen DeepLOB modellt tanít, de több adatparallelizmus technikával.
"""
import os, datetime as dt, time
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
import torch.cuda.amp as amp
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import torch.nn.functional as F

# Importáljuk a DeepLOB modellt és az adatbetöltő függvényeket
from deeplob_optimized import (
    DeepLOB, 
    load_book_chunk
)

# Importáljuk az GPU Dataset modult
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

class SingleModelParallelTrainer:
    """
    Egyetlen modell tréner, amely a modell adatparalelizmusát használja
    a GPU kihasználtság maximalizálására.
    """
    def __init__(self, 
                 file_paths,
                 depth=10,
                 window=100,
                 horizon=100,
                 batch_size=64,
                 alpha=0.002,
                 stride=5,
                 epochs=40,
                 lr=1e-3,
                 patience=5,
                 split_batches=3):  # Hány párhuzamos micro-batch-re osszuk a batch-et
        """
        Inicializálja a párhuzamos adatfeldolgozású trénert.
        
        Args:
            file_paths: A feldolgozandó fájlok listája
            depth: Árszintek száma
            window: Időablak mérete
            horizon: Előrejelzési horizont
            batch_size: Batch méret
            alpha: Küszöbérték az árváltozás osztályozásához
            stride: Lépés mérete a mintavételezéshez
            epochs: Maximális epoch-ok száma
            lr: Tanulási ráta
            patience: Early stopping türelmi idő
            split_batches: Hány párhuzamos micro-batch-re osszuk a batch-et
        """
        self.file_paths = file_paths
        self.depth = depth
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.alpha = alpha
        self.stride = stride
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.split_batches = split_batches
        
        # Inicializáljuk a modellt
        print("Initializing model...")
        self.model = DeepLOB(depth=depth).to(
            device, 
            memory_format=torch.channels_last
        )
        
        # GPU adatbetöltők létrehozása
        print("Creating GPU dataloaders...")
        
        # A teljes adathalmazt betöltjük a GPU-ra
        self.train_loader, self.val_loader = create_gpu_data_loaders(
            file_paths=file_paths,
            valid_frac=0.1,
            depth=depth,
            window=window,
            horizon=horizon,
            batch_size=batch_size * split_batches,  # Nagyobb batch-ek, amiket aztán felosztunk
            alpha=alpha,
            stride=stride,
            device=device
        )
        
        # Optimalizáló és veszteségfüggvény
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr, 
            betas=(0.9,0.999),
            eps=1e-8, 
            weight_decay=1e-4,
            fused=True
        )
        
        # Mixed precision skálázó
        self.scaler = amp.GradScaler(
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=True
        )
        
        # Veszteségfüggvény
        self.criterion = nn.CrossEntropyLoss()
    
    def _parallel_forward(self, x, y):
        """
        Párhuzamos forward pass több micro-batch-en.
        
        Args:
            x: Input batch (már a GPU-n)
            y: Target batch (már a GPU-n)
            
        Returns:
            Teljes loss és logits
        """
        # Felosztjuk a batch-et micro-batch-ekre
        micro_batch_size = x.size(0) // self.split_batches
        
        # Veszteségek és predikcók tárolása
        losses = []
        all_logits = []
        
        # Párhuzamos forward pass minden micro-batch-re
        with torch.amp.autocast(device_type='cuda'):
            for i in range(self.split_batches):
                # Kiválasztjuk a micro-batch-et
                start_idx = i * micro_batch_size
                end_idx = (i + 1) * micro_batch_size if i < self.split_batches - 1 else x.size(0)
                
                micro_x = x[start_idx:end_idx].contiguous()
                micro_y = y[start_idx:end_idx]
                
                # Forward pass a micro-batch-en
                logits = self.model(micro_x)
                loss = self.criterion(logits, micro_y) / self.split_batches
                
                # Eredmények tárolása
                losses.append(loss)
                all_logits.append(logits)
        
        # Összegezzük a veszteségeket
        total_loss = sum(losses)
        
        # Összefűzzük a logits-okat
        all_logits = torch.cat(all_logits, dim=0)
        
        return total_loss, all_logits
    
    def train_one_epoch(self, epoch):
        """
        Egy epoch tréningje párhuzamos adatfeldolgozással.
        
        Args:
            epoch: Az aktuális epoch száma
            
        Returns:
            Átlagos veszteség
        """
        self.model.train()
        running_loss = 0.0
        
        # Batch időmérés
        batch_times = []
        forward_times = []
        backward_times = []
        
        # Gradiens akkumuláció számláló
        accum_steps = 4  # 4 lépésenként frissítünk
        
        print(f"Training: {len(self.train_loader)} batches with split_batches={self.split_batches}")
        
        for step, (xb, yb) in enumerate(self.train_loader):
            batch_start = time.time()
            
            # Párhuzamos forward pass
            forward_start = time.time()
            loss, _ = self._parallel_forward(xb, yb)
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)
            
            # Backward pass
            backward_start = time.time()
            self.scaler.scale(loss).backward()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)
            
            # Gradiens akkumuláció
            if (step + 1) % accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            running_loss += loss.item() * accum_steps
            
            # Batch idő mérése
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Állapot kiírása minden 50. batch után
            if (step + 1) % 50 == 0:
                avg_batch = sum(batch_times[-50:]) / min(50, len(batch_times[-50:]))
                
                # Becsült epoch futási idő (hátralévő batchek * átlagos batch idő)
                remaining_batches = len(self.train_loader) - (step + 1)
                estimated_remaining_time = remaining_batches * avg_batch
                elapsed_time = time.time() - batch_start + sum(batch_times[:-1])
                estimated_total_time = elapsed_time + estimated_remaining_time
                
                print(f"Epoch {epoch} - Batch {step+1}/{len(self.train_loader)} - "
                      f"Loss: {running_loss/(step+1):.4f} - "
                      f"Becsült epoch idő: {estimated_total_time/60:.2f} perc")
                
                # GPU kihasználtság ellenőrzése
                print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / "
                      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        return running_loss / len(self.train_loader.sampler)
    
    def validate(self):
        """
        Validáció párhuzamos adatfeldolgozással, részletes metrikák számításával.
        
        Returns:
            directional_f1: Az irányos F1 score (up és down osztályok átlaga)
            val_loss: A validációs veszteség
            metrics: Dict a részletes metrikákkal
        """
        print("Starting validation phase...")
        val_start = time.time()
        self.model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for i, (xb, yb) in enumerate(self.val_loader):
                if i % 20 == 0:
                    print(f"Validation batch {i}/{len(self.val_loader)}")
                    
                # Párhuzamos forward pass validációnál is
                loss, logits = self._parallel_forward(xb, yb)
                val_loss += loss.item()
                batch_preds = logits.argmax(1)
                
                # Eredmények gyűjtése
                y_true.append(yb)
                y_pred.append(batch_preds)
        
        val_end = time.time()
        print(f"Validation completed in {val_end - val_start:.2f}s")
        
        # Összegyűjtjük az összes predikciót és valós címkét
        all_true = torch.cat(y_true)
        all_pred = torch.cat(y_pred)
        
        # CPU-ra kell másolni a scikit-learn függvényhez
        cpu_true = all_true.cpu().numpy()
        cpu_pred = all_pred.cpu().numpy()
        
        # F1 score számítása minden osztályra külön-külön
        class_f1 = f1_score(cpu_true, cpu_pred, average=None)
        
        # Az osztályok f1 értékei: [down, stable, up] (0, 1, 2 osztályok)
        f1_down, f1_stable, f1_up = class_f1
        
        # Az "up" és "down" osztályok F1 értékeinek átlaga - ez lesz az új mérőszám
        directional_f1 = (f1_up + f1_down) / 2
        
        # A sima macro F1-et is kiszámítjuk a diagnosztikához
        macro_f1 = f1_score(cpu_true, cpu_pred, average='macro')
        
        # Validációs veszteség átlagolása
        val_loss = val_loss / len(self.val_loader)
        
        # Konfúziós mátrix számítása
        conf_matrix = confusion_matrix(cpu_true, cpu_pred)
        
        # Precision és recall számítása osztályonként GPU-n
        conf_tensor = torch.zeros(3, 3, dtype=torch.int, device=device)
        for t in range(3):
            for p in range(3):
                conf_tensor[t, p] = torch.sum((all_true == t) & (all_pred == p))
        
        precision = torch.zeros(3, device=device)
        recall = torch.zeros(3, device=device)
        
        for i in range(3):
            # Precision: TP / (TP + FP)
            precision[i] = conf_tensor[i, i] / conf_tensor[:, i].sum() if conf_tensor[:, i].sum() > 0 else 0
            # Recall: TP / (TP + FN)
            recall[i] = conf_tensor[i, i] / conf_tensor[i, :].sum() if conf_tensor[i, :].sum() > 0 else 0
        
        # F1 score: 2 * (precision * recall) / (precision + recall)
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Osztályok eloszlása a validációs halmazban
        class_counts = [torch.sum(all_true == i).item() for i in range(3)]
        class_distribution = [count / len(all_true) * 100 for count in class_counts]
        
        # Részletes classification report a scikit-learn-től
        report = classification_report(
            cpu_true, 
            cpu_pred, 
            target_names=['Down', 'Stable', 'Up'], 
            digits=4,
            output_dict=True
        )
        
        # Eredmények összegyűjtése egy szótárban
        metrics = {
            'directional_f1': directional_f1,
            'macro_f1': macro_f1,
            'class_f1': class_f1,
            'f1_per_class': f1_per_class.cpu().numpy(),
            'precision': precision.cpu().numpy(),
            'recall': recall.cpu().numpy(),
            'conf_matrix': conf_matrix,
            'class_counts': class_counts,
            'class_distribution': class_distribution,
            'report': report,
            'val_loss': val_loss,
            'all_true': cpu_true,
            'all_pred': cpu_pred
        }
        
        return directional_f1, val_loss, metrics
    
    def train(self):
        """
        Modell tréningje.
        
        Returns:
            Betanított modell
        """
        print(f"\n=== Starting Training with Parallel Data Processing ===\n")
        start_time = time.time()
        
        # Változók inicializálása
        best_f1 = 0.0
        wait = 0
        total_train_time = 0
        
        # Főciklus: epoch-ok
        for ep in range(1, self.epochs + 1):
            epoch_start = time.time()
            print(f"\n--- Epoch {ep}/{self.epochs} ---")
            
            # Tréning
            loss = self.train_one_epoch(ep)
            
            # Validáció
            directional_f1, val_loss, metrics = self.validate()
            
            # Epoch statisztikák
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time
            macro_f1 = metrics['macro_f1']
            class_f1 = metrics['class_f1']
            conf_matrix = metrics['conf_matrix']
            f1_down, f1_stable, f1_up = class_f1
            
            # Eredmények kijelzése
            print(f"Epoch {ep} - Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"F1 Scores - Down: {f1_down:.4f}, Stable: {f1_stable:.4f}, Up: {f1_up:.4f}")
            print(f"Directional F1: {directional_f1:.4f}, Macro F1: {macro_f1:.4f}")
            
            # Classification report kiírása időnként
            if ep % 10 == 0 or ep == 1:
                report_str = classification_report(
                    metrics['all_true'],
                    metrics['all_pred'],
                    target_names=['Down', 'Stable', 'Up'],
                    digits=4
                )
                print(f"Classification Report:\n{report_str}")
            
            # Early stopping ellenőrzése
            if directional_f1 > best_f1 + 1e-4:  # A directional F1-et használjuk az értékeléshez
                best_f1 = directional_f1
                wait = 0
                
                # Könyvtár létrehozása, ha nem létezik
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                
                # Modell mentése
                checkpoint_path = Path(f"models/deeplob_single_parallel_f1_{best_f1:.4f}.pt")
                
                model_state = {
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': ep,
                    'f1': best_f1,
                    'directional_f1': directional_f1,
                    'macro_f1': macro_f1
                }
                
                torch.save(model_state, checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
                
                # Konfúziós mátrix mentése és vizualizációja
                plt.figure(figsize=(10, 8))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                           xticklabels=['Down', 'Stable', 'Up'],
                           yticklabels=['Down', 'Stable', 'Up'])
                plt.title(f'Best Confusion Matrix (F1: {directional_f1:.4f})')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig('best_confusion_matrix_single_parallel.png')
                plt.close()
                print(f"Saved confusion matrix to 'best_confusion_matrix_single_parallel.png'")
                
                # Részletesebb statisztikák mentése szöveges fájlba
                report_path = f"models/classification_report_single_parallel.txt"
                with open(report_path, 'w') as f:
                    f.write(f"Epoch: {ep}\n")
                    f.write(f"Directional F1: {directional_f1:.4f}\n")
                    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
                    f.write(f"Classification Report:\n{report_str}\n\n")
                    f.write("Confusion Matrix:\n")
                    f.write(str(conf_matrix) + "\n\n")
                    
                    # Osztályonkénti metrikák
                    f.write("Per-class metrics:\n")
                    class_names = ['Down', 'Stable', 'Up']
                    for i, class_name in enumerate(class_names):
                        f.write(f"{class_name}:\n")
                        f.write(f"  Precision: {metrics['precision'][i]:.4f}\n")
                        f.write(f"  Recall: {metrics['recall'][i]:.4f}\n")
                        f.write(f"  F1: {metrics['f1_per_class'][i]:.4f}\n\n")
                    
                    # Osztályok eloszlása a validációs halmazban
                    f.write("Class distribution in validation set:\n")
                    for i, class_name in enumerate(class_names):
                        f.write(f"  {class_name}: {metrics['class_counts'][i]} samples "
                                f"({metrics['class_distribution'][i]:.2f}%)\n")
                
                print(f"Saved detailed metrics to '{report_path}'")
                
            else:
                wait += 1
                print(f"No improvement for {wait} epochs. Best F1: {best_f1:.4f}")
                
                if wait >= self.patience:
                    print(f"Early stopping after {ep} epochs")
                    break
        
        print(f"\nTraining completed in {total_train_time:.2f}s ({total_train_time/60:.2f} minutes)")
        print(f"Best F1 score: {best_f1:.4f}")
        
        # Nem térünk vissza a modellel, mivel a trainer objektum tartalmazza azt
        return

def main():
    # Alapvető konfiguráció
    print("\n=== DeepLOB Training - Single Model Parallelism ===\n")
    
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
    
    # 2. Párhuzamos tréner inicializálása
    print("\nInitializing single model parallel trainer...")
    t_start = time.time()
    
    # Egy modell, de párhuzamos adatfeldolgozással
    trainer = SingleModelParallelTrainer(
        file_paths=file_paths,
        depth=10,
        window=100,
        horizon=100,
        batch_size=64,        # Az eredeti batch méret
        alpha=0.002,
        stride=5,
        epochs=40,
        lr=1e-3,
        patience=5,
        split_batches=4       # Hány párhuzamos micro-batch-re osszuk a batch-et
    )
    
    print(f"Trainer initialization completed in {time.time()-t_start:.2f}s")
    
    # 3. Modell tréningje párhuzamos adatfeldolgozással
    trainer.train()
    
    print("\n=== Training Complete! ===")
    
    return trainer

if __name__ == "__main__":
    main()
