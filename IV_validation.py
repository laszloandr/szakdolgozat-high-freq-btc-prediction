"""
Model Validation Pipeline - Előtanított DeepLOB modellek tesztelésére.
Ez a modul lehetővé teszi egy előtanított DeepLOB modell tesztelését egy adott időszakra.
"""
import os, datetime as dt, time
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Importáljuk a DeepLOB modellt és az adatbetöltő függvényeket
from deeplob_optimized import (
    DeepLOB, 
    load_book_chunk
)

# Importáljuk az GPU Dataset modult
from gpu_loaders import create_gpu_data_loaders

# PyTorch és CUDA konfiguráció
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# Optimalizálás
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class ModelValidator:
    """
    Egy előtanított DeepLOB modell validálására szolgáló osztály.
    """
    def __init__(self, 
                 file_paths,
                 model_path,
                 depth=10,
                 window=100,
                 horizon=100,
                 batch_size=64,
                 alpha=0.002,
                 stride=1):
        """
        Inicializálja a modell validáló osztályt.
        
        Args:
            file_paths: A feldolgozandó fájlok listája (file info dictionaries)
            model_path: Az előtanított modell elérési útja (.pt fájl)
            depth: Árszintek száma
            window: Időablak mérete
            horizon: Előrejelzési horizont
            batch_size: Batch méret
            alpha: Küszöbérték az árváltozás osztályozásához
            stride: Lépés mérete a mintavételezéshez (alapértelmezett: 1 a validációhoz)
        """
        self.file_paths = file_paths
        self.model_path = model_path
        self.depth = depth
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.alpha = alpha
        self.stride = stride
        
        # Ellenőrizzük, hogy a modell fájl létezik
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"A megadott modell fájl nem található: {model_path}")
        
        # Inicializáljuk a modellt
        print(f"Initializing model from {model_path}...")
        self.model = DeepLOB(depth=depth).to(
            device, 
            memory_format=torch.channels_last
        )
        
        # Betöltjük a modell súlyait - explicit weights_only=False a kompatibilitás miatt
        try:
            # Először próbáljuk a biztonságosabb módban
            checkpoint = torch.load(model_path, weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(f"Biztonságos betöltés sikertelen, teljes objektum betöltése: {e}")
            # Ha nem sikerül, használjuk a kevésbé biztonságos, de kompatibilis módot
            checkpoint = torch.load(model_path, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
        
        print(f"Model loaded successfully. Original training metrics:")
        if 'directional_f1' in checkpoint:
            print(f"- Directional F1: {checkpoint['directional_f1']:.4f}")
        if 'f1' in checkpoint:
            print(f"- Global F1: {checkpoint['f1']:.4f}")
        if 'epoch' in checkpoint:
            print(f"- Trained for {checkpoint['epoch']} epochs")
        
        # Adatok betöltése a validáláshoz
        print("\nLoading data for validation...")
        self.data = []
        self.raw_prices = []  # Új lista a nyers áraknak
        
        for file_info in self.file_paths:
            print(f"Reading normalized file: {file_info['filename']}")
            print(f"Time range: {file_info['start_date']} to {file_info['end_date']}")
            
            # Betöltjük a normalizált adatokat
            df = pd.read_parquet(file_info['path'])
            
            # Szűrjük az adatokat a megfelelő időszakra
            mask = (df['received_time'] >= file_info['start_date']) & \
                   (df['received_time'] <= file_info['end_date'])
            df = df[mask]
            
            if len(df) > 0:
                self.data.append(df)
                print(f"Loaded {len(df)} normalized records")
                
                # Betöltjük a nyers adatokat
                raw_file_path = os.path.join(
                    'szakdolgozat-high-freq-btc-prediction/data_raw',
                    f"book_btc_usdt_{file_info['start_date'].strftime('%Y%m%d')}_{file_info['end_date'].strftime('%Y%m%d')}.parquet"
                )
                
                if os.path.exists(raw_file_path):
                    print(f"Reading raw file: {raw_file_path}")
                    raw_df = pd.read_parquet(raw_file_path)
                    
                    # Szűrjük a nyers adatokat ugyanarra az időszakra
                    raw_mask = (raw_df['received_time'] >= file_info['start_date']) & \
                             (raw_df['received_time'] <= file_info['end_date'])
                    raw_df = raw_df[raw_mask]
                    
                    if len(raw_df) > 0:
                        # Csak az ask_0_price oszlopot tartjuk meg
                        raw_prices = raw_df[['received_time', 'ask_0_price']]
                        self.raw_prices.append(raw_prices)
                        print(f"Loaded {len(raw_prices)} raw price records")
                    else:
                        print(f"No raw data found in time range {file_info['start_date']} to {file_info['end_date']}")
                else:
                    print(f"Raw file not found: {raw_file_path}")
            else:
                print(f"No normalized data found in time range {file_info['start_date']} to {file_info['end_date']}")
        
        if not self.data:
            raise ValueError("No data found in the specified time ranges")
        
        # Összefűzzük a normalizált adatokat
        self.combined_data = pd.concat(self.data, ignore_index=True)
        print(f"\nTotal normalized data points loaded: {len(self.combined_data)}")
        
        # Összefűzzük a nyers árakat
        if self.raw_prices:
            self.combined_raw_prices = pd.concat(self.raw_prices, ignore_index=True)
            print(f"Total raw price data points loaded: {len(self.combined_raw_prices)}")
        else:
            self.combined_raw_prices = None
            print("Warning: No raw price data loaded")
        
        if len(self.combined_data) > 0:
            print(f"Time range: {self.combined_data['received_time'].min()} to {self.combined_data['received_time'].max()}")
        
        # GPU adatbetöltő létrehozása az egész teszthalmazra
        print("\nCreating GPU test dataloader...")
        
        # Create a temporary filtered parquet file for GPU loading
        temp_dir = Path("./szakdolgozat-high-freq-btc-prediction/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique filename for the filtered data
        temp_file = temp_dir / f"filtered_data_{int(time.time())}.parquet"
        
        # Save the filtered data to a temporary parquet file
        print(f"Saving filtered data to temporary file: {temp_file}")
        self.combined_data.to_parquet(temp_file, index=False)
        
        # Create GPU data loader with the filtered file
        _, self.test_loader = create_gpu_data_loaders(
            file_paths=[str(temp_file)],  # Use the temporary filtered file
            valid_frac=1.0,  # A teljes adathalmaz tesztként lesz használva
            depth=depth,
            window=window,
            horizon=horizon,
            batch_size=batch_size,
            alpha=alpha,
            stride=1,  # Force stride=1 for validation to keep all data points
            device=device
        )
        
        # Clean up the temporary file
        try:
            os.remove(temp_file)
            print(f"Temporary file removed: {temp_file}")
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
        
        # Veszteségfüggvény
        self.criterion = nn.CrossEntropyLoss()
    
    def validate(self):
        """
        A modell validálása a teljes teszthalmazon.
        
        Returns:
            metrics: Dict a részletes metrikákkal
        """
        print("Starting validation phase...")
        val_start = time.time()
        self.model.eval()
        y_true, y_pred = [], []
        test_loss = 0.0
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for i, (xb, yb) in enumerate(self.test_loader):
                if i % 20 == 0:
                    print(f"Validation batch {i}/{len(self.test_loader)}")
                
                # Forward pass
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                test_loss += loss.item()
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
        
        # Létrehozzuk a metrics szótárat az összes eredménnyel
        metrics = {'raw_predictions': cpu_pred}
        
        # Időbélyegek és árak hozzáadása a validált adatokból
        try:
            print("\nAdding timestamps and prices to metrics...")
            # A validált adatok számának megfelelően mintavételezünk
            num_predictions = len(cpu_pred)
            metrics['timestamps'] = self.combined_data['received_time'].values[:num_predictions]
            metrics['normalized_prices'] = self.combined_data['ask_0_price'].values[:num_predictions]
            
            # Nyers árak hozzáadása
            if self.combined_raw_prices is not None:
                # Merge the raw prices with the normalized data based on timestamps
                merged_data = pd.merge(
                    pd.DataFrame({'received_time': metrics['timestamps']}),
                    self.combined_raw_prices,
                    on='received_time',
                    how='left'
                )
                metrics['raw_prices'] = merged_data['ask_0_price'].values
                print(f"Added raw prices to metrics. Shape: {len(metrics['raw_prices'])}")
            else:
                print("Warning: No raw prices available to add to metrics")
                metrics['raw_prices'] = np.zeros_like(metrics['normalized_prices'])
            
            print(f"Data shapes: timestamps={len(metrics['timestamps'])}, "
                  f"normalized_prices={len(metrics['normalized_prices'])}, "
                  f"raw_prices={len(metrics['raw_prices'])}, "
                  f"predictions={len(cpu_pred)}")
            
        except Exception as e:
            print(f"Error adding timestamps and prices: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # F1 score számítása minden osztályra külön-külön
        class_f1 = f1_score(cpu_true, cpu_pred, average=None)
        
        # Az osztályok f1 értékei: [down, stable, up] (0, 1, 2 osztályok)
        f1_down, f1_stable, f1_up = class_f1
        
        # Az "up" és "down" osztályok F1 értékeinek átlaga - ez lesz az directional F1
        directional_f1 = (f1_up + f1_down) / 2
        
        # A sima macro F1-et is kiszámítjuk a diagnosztikához
        macro_f1 = f1_score(cpu_true, cpu_pred, average='macro')
        
        # Validációs veszteség átlagolása
        test_loss = test_loss / len(self.test_loader)
        
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
        metrics.update({
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
            'test_loss': test_loss,
            'all_true': cpu_true,
            'all_pred': cpu_pred
        })
        
        return metrics
    
    def plot_confusion_matrix(self, metrics, save_path=None):
        """
        Konfúziós mátrix vizualizációja.
        
        Args:
            metrics: A validate() által visszaadott metrikák
            save_path: Opcionális elérési út a konfúziós mátrix mentéséhez
        """
        conf_matrix = metrics['conf_matrix']
        directional_f1 = metrics['directional_f1']
        macro_f1 = metrics['macro_f1']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                   xticklabels=['Down', 'Stable', 'Up'],
                   yticklabels=['Down', 'Stable', 'Up'])
        plt.title(f'Confusion Matrix (Directional F1: {directional_f1:.4f}, Macro F1: {macro_f1:.4f})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        if save_path:
            # Biztositjuk, hogy a szulő könyvtár létezik
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Saved confusion matrix to '{save_path}'")
        
        plt.show()
    
    def save_report(self, metrics, save_path):
        """
        Részletes jelentés mentése fájlba.
        
        Args:
            metrics: A validate() által visszaadott metrikák
            save_path: Elérési út a jelentés mentéséhez
        """
        with open(save_path, 'w') as f:
            # Alap metrikák
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Directional F1: {metrics['directional_f1']:.4f}\n")
            f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n\n")
            
            # Classification report
            report_str = classification_report(
                metrics['all_true'],
                metrics['all_pred'],
                target_names=['Down', 'Stable', 'Up'],
                digits=4
            )
            f.write(f"Classification Report:\n{report_str}\n\n")
            
            # Konfúziós mátrix
            f.write("Confusion Matrix:\n")
            f.write(str(metrics['conf_matrix']) + "\n\n")
            
            # Osztályonkénti metrikák
            f.write("Per-class metrics:\n")
            class_names = ['Down', 'Stable', 'Up']
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {metrics['precision'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['recall'][i]:.4f}\n")
                f.write(f"  F1: {metrics['f1_per_class'][i]:.4f}\n\n")
            
            # Osztályok eloszlása a teszthalmazban
            f.write("Class distribution in test set:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"  {class_name}: {metrics['class_counts'][i]} samples "
                        f"({metrics['class_distribution'][i]:.2f}%)\n")
        
        print(f"Saved detailed report to '{save_path}'")
        


def validate_model(start_date, end_date, model_path, 
                symbol='BTC-USDT', depth=10, window=100, horizon=100, 
                batch_size=64, alpha=0.002, stride=5, save_output=True,
                data_dir="./data_normalized"):
    """
    Validál egy előtanított DeepLOB modellt a megadott paraméterekkel.
    
    Args:
        start_date: Kezdő dátum (dt.datetime vagy string 'YYYY-MM-DD' formátumban)
        end_date: Befejező dátum (dt.datetime vagy string 'YYYY-MM-DD' formátumban)
        model_path: Az előtanított modell elérési útja (.pt fájl)
        symbol: Kereskedési szimbólum (alapértelmezett: 'BTC-USDT')
        depth: Árszintek száma (alapértelmezett: 10)
        window: Időablak mérete (alapértelmezett: 100)
        horizon: Előrejelzési horizont (alapértelmezett: 100)
        batch_size: Batch méret (alapértelmezett: 64)
        alpha: Küszöbérték az árváltozás osztályozásához (alapértelmezett: 0.002)
        stride: Lépés mérete a mintavételezéshez (alapértelmezett: 5)
        save_output: Ha True, a konfúziós mátrix és a jelentés mentésre kerül (alapértelmezett: True)
        
    Returns:
        metrics: Dict a validálási metrikákkal
        validator: A ModelValidator példány
    """
    # Dátumok konvertálása, ha string formátumban érkeznek
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
    
    print(f"\n=== DeepLOB Model Validation ===")
    print(f"Time period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Symbol: {symbol}")
    print(f"Model: {model_path}")
    
    # Normalizált adatfájlok keresése
    t_start = time.time()
    file_infos = load_book_chunk(
        start_date,
        end_date,
        symbol,
        data_dir=data_dir
    )
    
    if not file_infos:
        print("No normalized files found for the specified time period.")
        return None, None
    
    print(f"Found {len(file_infos)} files for processing")
    print(f"File information loaded in {time.time()-t_start:.2f}s")
    
    # Ellenőrizzük, hogy a fájlok lefedik-e a teljes időszakot
    total_start = min(info['start_date'] for info in file_infos)
    total_end = max(info['end_date'] for info in file_infos)
    
    if total_start > start_date or total_end < end_date:
        print(f"Warning: Available data only covers {total_start} to {total_end}")
        print("This is less than the requested time period.")
        response = input("Do you want to continue with the available data? (y/n): ")
        if response.lower() != 'y':
            print("Validation cancelled.")
            return None, None
    
    # Validator inicializálása
    validator = ModelValidator(
        file_paths=file_infos,
        model_path=model_path,
        depth=depth,
        window=window,
        horizon=horizon,
        batch_size=batch_size,
        alpha=alpha,
        stride=stride
    )
    
    # Validálás futtatása
    metrics = validator.validate()
    
    # Eredmények kiírása
    print("\n=== Validation Results ===")
    print(f"Directional F1 Score (average of down and up): {metrics['directional_f1']:.4f}")
    print(f"Global F1 Score (macro average): {metrics['macro_f1']:.4f}")
    print(f"Class F1 Scores - Down: {metrics['class_f1'][0]:.4f}, Stable: {metrics['class_f1'][1]:.4f}, Up: {metrics['class_f1'][2]:.4f}")
    
    if save_output:
        # Könyvtár struktura biztosítása
        output_dir = Path("./szakdolgozat-high-freq-btc-prediction/results/deeplob")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Fájlnevek generálása
        model_name = os.path.basename(model_path).split('.')[0]
        date_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Konfúziós mátrix mentése
        conf_matrix_path = output_dir / f"validation_conf_matrix_{model_name}_{date_str}.png"
        validator.plot_confusion_matrix(metrics, save_path=str(conf_matrix_path))
        
        # Részletes jelentés mentése
        report_path = output_dir / f"validation_report_{model_name}_{date_str}.txt"
        validator.save_report(metrics, str(report_path))
        
        # Előrejelzések mentése parquet fájlba
        print("\nChecking metrics for parquet file creation...")
        print(f"Available keys in metrics: {list(metrics.keys())}")
        
        if all(key in metrics for key in ['timestamps', 'normalized_prices', 'raw_prices', 'raw_predictions']):
            print(f"Found required data for parquet file:")
            print(f"- Timestamps shape: {len(metrics['timestamps'])}")
            print(f"- Normalized prices shape: {len(metrics['normalized_prices'])}")
            print(f"- Raw prices shape: {len(metrics['raw_prices'])}")
            print(f"- Predictions shape: {len(metrics['raw_predictions'])}")
            
            predictions_df = pd.DataFrame({
                'received_time': metrics['timestamps'],
                'prediction': metrics['raw_predictions'],
                'normalized_price': metrics['normalized_prices'],
                'raw_price': metrics['raw_prices']
            })
            
            # Parquet fájl mentése
            predictions_path = output_dir / f"predictions_{model_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
            predictions_df.to_parquet(predictions_path, index=False)
            print(f"Saved predictions to '{predictions_path}'")
        else:
            print("Missing required data for parquet file:")
            missing_keys = [key for key in ['timestamps', 'normalized_prices', 'raw_prices', 'raw_predictions'] 
                          if key not in metrics]
            print(f"Missing keys: {missing_keys}")
    
    print(f"\nValidation completed successfully.")
    if save_output:
        print(f"Confusion matrix saved to '{conf_matrix_path}'")
        print(f"Detailed report saved to '{report_path}'")
        
    return metrics, validator

if __name__ == "__main__":
    # Példa a használatra
    validate_model(
        start_date="2025-03-05",
        end_date="2025-03-10",
        model_path="./szakdolgozat-high-freq-btc-prediction/models/deeplob_single_parallel_f1_0.4369.pt"
    )
