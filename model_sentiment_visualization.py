"""
Model Sentiment Visualization - DeepLOB modell előrejelzés vizualizáció.
Ez a modul lehetővé teszi a DeepLOB modell előrejelzéseinek vizualizációját árfolyam adatokkal,
a különböző előrejelzési kategóriák (le, stabil, fel) arányainak időszakonkénti megjelenítésével oszlopdiagramok formájában.
"""
import os
import time
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from glob import glob

def plot_model_sentiment(
    timestamps, 
    prices, 
    predictions, 
    window_size='15min',
    title="Price vs. DeepLOB predictions",
    save_path=None
):
    """
    Plotly alapú interaktív idősoros ábra létrehozása az árfolyammal és időszakonkénti predikciós arányokkal.
    
    Args:
        timestamps: Időbélyegek sorozata (datetime64)
        prices: Árfolyam értékek (ask_0_price)
        predictions: Modell előrejelzések (0=down, 1=stable, 2=up)
        window_size: Időszakok mérete (default: '15min')
        title: Diagram címe
        save_path: Mentési útvonal (ha meg van adva)
        
    Returns:
        plotly.graph_objects.Figure: Az elkészített interaktív ábra
    """
    # Paraméterek ellenőrzése
    if len(timestamps) != len(prices) or len(timestamps) != len(predictions):
        raise ValueError("Az időbélyegek, árak és előrejelzések hosszának egyeznie kell")
    
    # DataFrame létrehozása
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'prediction': predictions
    })
    df.set_index('timestamp', inplace=True)
    
    # Időszakonkénti aggregálás
    print("Időszakonkénti arányok számítása...")
    t_start = time.time()
    
    # Más megközelítés az időszakonkénti arányok számolására
    # Elem dictionaryt hozunk létre az eredmények tárolására
    resampled_data = {}
    
    # Időszakok generálása a teljes tartományra
    min_time = df.index.min()
    max_time = df.index.max()
    intervals = pd.date_range(start=min_time, end=max_time, freq=window_size)
    
    # Create an empty DataFrame to store prediction counts for each interval
    prediction_counts = pd.DataFrame(index=intervals[:-1], columns=[0, 1, 2])
    prediction_counts.fillna(0, inplace=True)
    
    # Iterate through intervals and count predictions
    for i in range(len(intervals) - 1):
        start_time = intervals[i]
        end_time = intervals[i+1]
        
        # Szűrés az aktuális időintervallumra
        interval_data = df[(df.index >= start_time) & (df.index < end_time)]
        
        # Számolás kategóriák szerint
        if not interval_data.empty:
            counts = interval_data['prediction'].value_counts()
            
            # Hozzáadjuk a számokat a megfelelő oszlopokhoz
            for pred_class in counts.index:
                prediction_counts.at[start_time, pred_class] = counts[pred_class]
    
    # Átlagos árfolyam időszakonként
    price_resampled = df['price'].resample(window_size).mean()
    
    # Arányok kiszámítása (sorokra normalizálva)
    row_sums = prediction_counts.sum(axis=1)
    row_sums_nonzero = row_sums.replace(0, 1)  # Elkerüli a nullával való osztást
    prediction_props = prediction_counts.div(row_sums_nonzero, axis=0)
    
    print(f"Időszakonkénti arányok számítási idő: {time.time() - t_start:.2f}s")
    
    # Ábra létrehozása két panellel
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    
    # Árfolyam vonal hozzáadása (felső panel)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1),
        ),
        row=1, col=1
    )
    
    # Bar chart adatok előkészítése
    bar_x = prediction_props.index
    
    # Down (0) predictions
    fig.add_trace(
        go.Bar(
            x=bar_x,
            y=prediction_props[0] if 0 in prediction_props.columns else [],
            name='Down (0)',
            marker_color='#d73027',  # Piros szín
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Stable (1) predictions
    fig.add_trace(
        go.Bar(
            x=bar_x,
            y=prediction_props[1] if 1 in prediction_props.columns else [],
            name='Stable (1)',
            marker_color='#4575b4',  # Kék szín
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Up (2) predictions
    fig.add_trace(
        go.Bar(
            x=bar_x,
            y=prediction_props[2] if 2 in prediction_props.columns else [],
            name='Up (2)',
            marker_color='#1a9850',  # Zöld szín
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Átállítás halmozott bar chartra
    fig.update_layout(
        barmode='stack'
    )
    
    # Diagram formázása
    fig.update_layout(
        title=title,
        hovermode="closest",  # Gyorsabb hover
        template="plotly_white",
        height=800,  # Nem kell olyan magas
        width=1200,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        # HTML méret csökkentés
        modebar_remove=['lasso', 'select']
    )
    
    # Y-tengelyek címkézése
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    # Alsó panel Y-tengely formázása (0-100% skála)
    fig.update_yaxes(
        title_text="Prediction Proportions (%)",
        row=2, col=1,
        range=[0, 1],  # 0-100% skála
        tickformat='.0%'  # Százalék formátum
    )
    
    # X-tengely formázása
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=5, label="5m", step="minute", stepmode="backward"),
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=30, label="30m", step="minute", stepmode="backward"),
                dict(count=1, label="1h", step="hour", stepmode="backward"),
                dict(count=6, label="6h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(step="all")
            ]),
            x=0.01,  # Balra igazítás
            y=1.1,   # Felső panel fölé helyezés
            xanchor="left",
            yanchor="bottom"
        ),
        row=2, col=1  # Csak az alsó panelen jelenik meg a rangeslider
    )
    
    # Mentés, ha meg van adva az útvonal
    if save_path:
        t_save = time.time()
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            save_path_html = save_path if save_path.endswith('.html') else save_path + '.html'
            fig.write_html(save_path_html)
        print(f"Interaktív ábra mentve: {save_path} ({time.time() - t_save:.2f}s)")
    
    return fig

def find_prediction_files(prediction_dir="results/deeplob", pattern="*.parquet"):
    """
    Megkeresi a prediction fájlokat a megadott mappában
    
    Args:
        prediction_dir: A mappa, ahol a fájlokat keressük
        pattern: A fájlok mintája (default: *.parquet)
        
    Returns:
        list: A talált fájlok listája
    """
    search_pattern = os.path.join(prediction_dir, pattern)
    files = glob(search_pattern)
    return sorted(files)

def list_available_predictions(prediction_dir="results/deeplob"):
    """
    Kilistázza az elérhető előrejelzés fájlokat
    
    Args:
        prediction_dir: A mappa, ahol a fájlokat keressük
        
    Returns:
        None
    """
    files = find_prediction_files(prediction_dir)
    if not files:
        print(f"Nem található előrejelzés fájl a {prediction_dir} mappában.")
        return
    
    print(f"Elérhető előrejelzés fájlok ({len(files)}):\n")
    for i, file in enumerate(files, 1):
        # Csak a fájlnevet jelezzük ki, kiterjesztés nélkül
        filename = os.path.basename(file)
        filename_no_ext = os.path.splitext(filename)[0]
        print(f"{i}. {filename_no_ext}")

def load_predictions_from_parquet(file_path):
    """
    Betölti a parquet fájlból az időbélyegeket, árfolyamokat és előrejelzéseket
    
    Args:
        file_path: A parquet fájl elérési útja
        
    Returns:
        tuple: (timestamps, prices, predictions)
    """
    # Fájl betöltése
    df = pd.read_parquet(file_path)
    
    # Ellenőrizzük a szükséges oszlopokat
    required_columns = ['received_time', 'prediction', 'ask_0_price']
    
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Hiányzó oszlop(ok) a parquet fájlban: {missing}")
    
    # Időbélyegek, árfolyamok és előrejelzések kiválasztása
    timestamps = df['received_time'].values
    prices = df['ask_0_price'].values
    predictions = df['prediction'].values
    
    return timestamps, prices, predictions

if __name__ == "__main__":
    # Példa a használatra - debug módban futtatható
    prediction_dir = "./szakdolgozat-high-freq-btc-prediction/results/deeplob"
    
    # Elérhető fájlok listázása
    list_available_predictions(prediction_dir)
    
    # Felhasználói input a fájl kiválasztásához
    print("\nKérlek add meg a kiválasztott fájl nevét (pl. predictions_deeplob_single_parallel_f1_0_20250303_20250307.parquet):")
    selected_filename = input().strip()
    
    # Teljes útvonal összeállítása
    selected_file = os.path.join(prediction_dir, selected_filename)
    
    if os.path.exists(selected_file):
        print(f"\nKiválasztott fájl: {selected_file}")
        
        # Adatok betöltése
        timestamps, prices, predictions = load_predictions_from_parquet(selected_file)
        print(f"Betöltött adatpontok száma: {len(timestamps)}")
        
        # Fájlnév kiolvasása kiterjesztés nélkül
        file_basename = os.path.basename(selected_file)
        model_name = os.path.splitext(file_basename)[0]  # Teljes fájlnév kiterjesztés nélkül
        title = f"Price vs. DeepLOB predictions - {model_name}"
        
        # Vizualizáció létrehozása
        fig = plot_model_sentiment(
            timestamps=timestamps,
            prices=prices,
            predictions=predictions,
            window_size='1min',  # 15 perces gördülő ablak
            title=title
        )
        
        # Megjelenítés
        fig.show()
        
        # Automatikus mentés HTML formátumban
        output_dir = "./szakdolgozat-high-freq-btc-prediction/results/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # HTML mentés (interaktív)
        html_path = os.path.join(output_dir, f"prediction_proportions_{model_name}.html")
        fig.write_html(html_path)
        print(f"\nInteraktív ábra mentve: {html_path}")
        
    else:
        print(f"A megadott fájl nem található: {selected_file}")
        print("Kérlek ellenőrizd a fájlnevet és próbáld újra.")
        

