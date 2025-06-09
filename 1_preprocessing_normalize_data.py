"""
Normalizálási pipeline LOB (Limit Order Book) adatokhoz.

Ez a script sorban olvassa be a parquet fájlokat és normalizálja őket
visszatekintő 5 napos statisztikák alapján, memória-hatékony módon.
A normalizált adatokat külön parquet fájlokba menti.

Használat:
    python normalize_data.py --start_date 2025-01-01 --end_date 2025-03-01 
                           --symbol BTC-USDT --input_dir ./data --output_dir ./data_normalized
"""
import os
import re
import argparse
import datetime as dt
from pathlib import Path
import time
import pandas as pd
import numpy as np
import cudf
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional, Any

# Típus definíciók
Stats = Dict[str, Dict[str, float]]  # {date: {column: value}}
FileInfo = Dict[str, Any]  # Parquet fájl információi

def parse_args():
    """Parancssori argumentumok feldolgozása"""
    parser = argparse.ArgumentParser(description='Normalizálási pipeline LOB adatokhoz')
    
    parser.add_argument('--start_date', type=str, required=True,
                        help='Kezdő dátum (YYYY-MM-DD formátumban)')
    parser.add_argument('--end_date', type=str, required=True,
                        help='Végső dátum (YYYY-MM-DD formátumban)')
    parser.add_argument('--symbol', type=str, default='BTC-USDT',
                        help='Kereskedési pár (pl. BTC-USDT)')
    parser.add_argument('--input_dir', type=str, default='./data',
                        help='Input könyvtár a parquet fájlokkal')
    parser.add_argument('--output_dir', type=str, default='./data_normalized',
                        help='Output könyvtár a normalizált parquet fájlokhoz')
    parser.add_argument('--window_days', type=int, default=5,
                        help='Visszatekintő statisztikák számításának ablaka (napokban)')
    parser.add_argument('--stats_update_freq', type=int, default=1,
                        help='Statisztikák frissítésének gyakorisága (napokban)')
    
    return parser.parse_args()

def find_parquet_files(input_dir: str, symbol: str, start_date: dt.datetime, 
                       end_date: dt.datetime) -> List[FileInfo]:
    """
    Megkeresi a megfelelő parquet fájlokat a megadott időszakra.
    
    Args:
        input_dir: Input könyvtár a parquet fájlokkal
        symbol: Kereskedési pár (pl. BTC-USDT)
        start_date: Kezdő dátum
        end_date: Végső dátum
        
    Returns:
        Talált fájlok listája kronológikus sorrendben
    """
    print(f"Parquet fájlok keresése {symbol} szimbólumhoz {start_date} és {end_date} között...")
    
    # Regex minta a fájlnevekhez
    sym_pat = symbol.lower().replace("-", "_")
    rex = re.compile(rf"book_{sym_pat}_(\d{{8}})_(\d{{8}})\.parquet$", re.I)
    
    matching_files = []
    
    for fn in sorted(os.listdir(input_dir)):
        fp = Path(input_dir) / fn
        if not fp.is_file():
            continue
        
        match = rex.match(fn)
        if not match:
            continue
        
        # Dátumok kinyerése a fájlnévből
        file_start = dt.datetime.strptime(match.group(1), "%Y%m%d")
        file_end = dt.datetime.strptime(match.group(2), "%Y%m%d")
        
        # Ellenőrizzük, hogy a fájl időtartama átfed-e a kért időszakkal
        if file_end < start_date or file_start > end_date:
            continue
        
        matching_files.append({
            'path': fp,
            'start_date': file_start,
            'end_date': file_end,
            'filename': fn
        })
    
    # Időrendi sorrendbe rendezés
    matching_files.sort(key=lambda x: x['start_date'])
    
    # Ellenőrizzük a hiányzó időszakokat
    if matching_files:
        # Keressünk lyukakat a dátumokban
        current_date = start_date
        for file_info in matching_files:
            if current_date < file_info['start_date']:
                missing_period = f"{current_date.strftime('%Y-%m-%d')} - {file_info['start_date'].strftime('%Y-%m-%d')}"
                print(f"FIGYELMEZTETÉS: Hiányzó adatok a következő időszakra: {missing_period}")
            current_date = max(current_date, file_info['end_date'] + dt.timedelta(days=1))
        
        # Ellenőrizzük, hogy van-e hiányzó adat a végén
        if current_date <= end_date:
            missing_period = f"{current_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
            print(f"FIGYELMEZTETÉS: Hiányzó adatok a következő időszakra: {missing_period}")
    
    print(f"Összesen {len(matching_files)} parquet fájl található a megadott időszakra")
    for i, file_info in enumerate(matching_files):
        print(f"  {i+1}. {file_info['filename']} "
              f"({file_info['start_date'].strftime('%Y-%m-%d')} - {file_info['end_date'].strftime('%Y-%m-%d')})")
    
    if not matching_files:
        print("HIBA: Nem található parquet fájl a megadott időszakra!")
    
    return matching_files

def get_feature_columns(df: cudf.DataFrame, depth: int = 20) -> List[str]:
    """
    Kiválasztja a releváns LOB feature oszlopokat.
    
    Args:
        df: A DataFrame
        depth: Mélység szintje
        
    Returns:
        Feature oszlopok listája
    """
    pat = rf'(bid|ask)_[0-9]{{1,2}}_(price|size)'
    feat_cols = [c for c in df.columns
                 if re.match(pat, c) and int(c.split('_')[1]) < depth]
    
    return feat_cols

def update_rolling_stats(stats: Stats, df: cudf.DataFrame, feat_cols: List[str], 
                         window_days: int) -> Stats:
    """
    Frissíti a gördülő statisztikákat az új adatokkal.
    
    Args:
        stats: Korábbi statisztikák szótára {date: {column: value}}
        df: Új adatok DataFrame
        feat_cols: Feature oszlopok listája
        window_days: Visszatekintő ablak napokban
        
    Returns:
        Frissített statisztikák
    """
    # Frissítsük a statisztikákat az új adatokkal
    if 'dates' not in stats:
        stats['dates'] = []
    if 'means' not in stats:
        stats['means'] = {col: [] for col in feat_cols}
    if 'counts' not in stats:
        stats['counts'] = []
    if 'vars' not in stats:
        stats['vars'] = {col: [] for col in feat_cols}
    
    # Dátum szerinti csoportosítás - cuDF kompatibilis módon
    # Kivonjuk a dátumrészt a datetime oszlopból
    df['date_str'] = df['received_time'].dt.strftime('%Y-%m-%d')
    dates = df['date_str'].unique().to_pandas().tolist()
    
    for date in dates:
        day_df = df[df['date_str'] == date]
        
        # Napi statisztikák számítása
        means = day_df[feat_cols].mean().to_pandas().to_dict()
        vars_dict = day_df[feat_cols].var().to_pandas().to_dict()
        count = len(day_df)
        
        # Statisztikák tárolása
        stats['dates'].append(date)
        for col in feat_cols:
            stats['means'][col].append(means[col])
            stats['vars'][col].append(vars_dict[col])
        stats['counts'].append(count)
    
    # Korlátozzuk a tárolt napok számát az ablak méretére
    if len(stats['dates']) > window_days:
        excess = len(stats['dates']) - window_days
        stats['dates'] = stats['dates'][excess:]
        stats['counts'] = stats['counts'][excess:]
        for col in feat_cols:
            stats['means'][col] = stats['means'][col][excess:]
            stats['vars'][col] = stats['vars'][col][excess:]
    
    return stats

def calculate_window_stats(stats: Stats, feat_cols: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Kiszámítja az ablakon belüli összesített statisztikákat.
    
    Args:
        stats: Statisztikák szótára
        feat_cols: Feature oszlopok listája
        
    Returns:
        (átlag, szórás) szótárak oszloponként
    """
    window_means = {}
    window_stds = {}
    
    # Ha nincs elég adat, használjuk ami van
    if not stats['dates']:
        return {}, {}
    
    # Súlyozott átlag számítása minden oszlopra
    total_count = sum(stats['counts'])
    
    for col in feat_cols:
        # Súlyozott átlag
        weighted_mean = sum(stats['means'][col][i] * stats['counts'][i] 
                           for i in range(len(stats['dates']))) / total_count
        
        # Súlyozott variancia (Welford-féle algoritmus alapján)
        # Először a napi varianciák súlyozott átlaga
        weighted_var = sum(stats['vars'][col][i] * stats['counts'][i] 
                          for i in range(len(stats['dates']))) / total_count
        
        # Korrekció a napi átlagok közötti különbséggel
        mean_var = sum(((stats['means'][col][i] - weighted_mean) ** 2) * stats['counts'][i]
                      for i in range(len(stats['dates']))) / total_count
        
        # Teljes variancia = napi varianciák átlaga + napi átlagok varianciája
        total_var = weighted_var + mean_var
        
        window_means[col] = weighted_mean
        window_stds[col] = np.sqrt(total_var) + 1e-8  # Kis epsilon a nullával való osztás elkerülésére
    
    return window_means, window_stds

def normalize_and_save(file_info: FileInfo, means: Dict[str, float], stds: Dict[str, float], 
                       output_dir: str, feat_cols: List[str]):
    """
    Normalizálja a parquet fájlt és elmenti az eredményt.
    
    Args:
        file_info: Fájl információk
        means: Átlagok oszloponként
        stds: Szórások oszloponként
        output_dir: Output könyvtár
        feat_cols: Feature oszlopok
    """
    if not means or not stds:
        print(f"FIGYELMEZTETÉS: Nincs elég statisztika a normalizáláshoz: {file_info['filename']}")
        return
    
    print(f"Normalizálás: {file_info['filename']}...")
    t_start = time.time()
    
    # Biztosítsuk, hogy a kimeneti könyvtár létezik
    os.makedirs(output_dir, exist_ok=True)
    
    # Fájlnév előállítása a normalizált adatokhoz
    output_path = Path(output_dir) / f"norm_{file_info['filename']}"
    
    # Beolvassuk a fájlt
    print(f"  Fájl beolvasása: {file_info['path']}...")
    df = cudf.read_parquet(file_info['path'])
    
    # Normalizálás
    print(f"  Z-score normalizálás {len(feat_cols)} oszlopra...")
    for col in feat_cols:
        if col in means and col in stds:
            df[col] = ((df[col] - means[col]) / stds[col]).astype('float32')
    
    # Normalizált adatok mentése
    print(f"  Normalizált adatok mentése: {output_path}...")
    df.to_parquet(output_path)
    
    # Memória felszabadítása
    del df
    
    print(f"Normalizálás kész: {file_info['filename']} -> {output_path} ({time.time()-t_start:.2f}s)")

def process_parquet_files(file_infos: List[FileInfo], args):
    """
    Feldolgozza a parquet fájlokat és normalizálja őket.
    
    Args:
        file_infos: Fájl információk listája
        args: Parancssori argumentumok
    """
    if not file_infos:
        return
    
    # Statisztikák tárolója
    stats = {}
    
    # Visszatekintő statisztikák napokban
    window_days = args.window_days
    
    # Statisztikák frissítésének gyakorisága
    stats_update_freq = args.stats_update_freq
    
    # Feature oszlopok meghatározása az első fájlból
    print(f"Feature oszlopok meghatározása az első fájlból: {file_infos[0]['filename']}...")
    sample_df = cudf.read_parquet(file_infos[0]['path'])
    feat_cols = get_feature_columns(sample_df, depth=20)
    print(f"Talált {len(feat_cols)} feature oszlop")
    del sample_df
    
    # Időrendi sorrendben dolgozzuk fel a fájlokat
    for i, file_info in enumerate(file_infos):
        print(f"\nFeldolgozás: {i+1}/{len(file_infos)} - {file_info['filename']}...")
        
        # Beolvassuk a fájlt és frissítjük a statisztikákat
        t_start = time.time()
        df = cudf.read_parquet(file_info['path'])
        
        # Statisztikák frissítése
        stats = update_rolling_stats(stats, df, feat_cols, window_days)
        
        # Statisztikák számítása az ablakra
        means, stds = calculate_window_stats(stats, feat_cols)
        
        # Memória felszabadítása
        del df
        
        # Csak akkor normalizáljuk, ha már elég adatunk van (legalább 1 nap)
        if stats['dates']:
            # Normalizáljuk és mentsük az adatokat
            normalize_and_save(file_info, means, stds, args.output_dir, feat_cols)
        else:
            print(f"FIGYELMEZTETÉS: Kihagyjuk a normalizálást: {file_info['filename']} - nincs elég statisztika")
        
        print(f"Fájl feldolgozása kész: {file_info['filename']} ({time.time()-t_start:.2f}s)")
    
    print("\nAz összes fájl feldolgozása befejeződött!")

def main():
    """Fő függvény"""
    # Parancssori argumentumok feldolgozása
    args = parse_args()
    
    # Dátumok konvertálása
    start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Fájlok keresése
    file_infos = find_parquet_files(args.input_dir, args.symbol, start_date, end_date)
    
    # Fájlok feldolgozása
    process_parquet_files(file_infos, args)

if __name__ == "__main__":
    main()
