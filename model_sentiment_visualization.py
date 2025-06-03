"""
Model Sentiment Visualization - DeepLOB modell előrejelzés vizualizáció.
Ez a modul lehetővé teszi a DeepLOB modell előrejelzéseinek vizualizációját árfolyam adatokkal.
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
    window_size='5min',
    title="Price vs. DeepLOB predictions",
    save_path=None
):
    """
    Plotly alapú interaktív idősoros ábra létrehozása az árfolyammal és a gördülő átlaggal.
    
    Args:
        timestamps: Időbélyegek sorozata (datetime64)
        prices: Árfolyam értékek (ask_0_price)
        predictions: Modell előrejelzések (0=down, 1=stable, 2=up)
        window_size: Gördülő ablak mérete (default: '5min')
        title: Diagram címe
        save_path: Mentési útvonal (ha meg van adva)
        
    Returns:
        plotly.graph_objects.Figure: Az elkészített interaktív ábra
    """
    # Paraméterek ellenőrzése
    if len(timestamps) != len(prices) or len(timestamps) != len(predictions):
        raise ValueError("Az időbélyegek, árak és előrejelzések hosszának egyeznie kell")
    
    # DataFrame létrehozása a gördülő átlag számításához
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'prediction': predictions
    })
    df.set_index('timestamp', inplace=True)
    
    # Gördülő átlag számítása
    print("Gördülő átlag számítása...")
    t_start = time.time()
    rolling_avg = df['prediction'].rolling(window=window_size, min_periods=1).mean()
    print(f"Gördülő átlag számítási idő: {time.time() - t_start:.2f}s")
    
    # Ábra létrehozása két y-tengellyel
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
    
    # Gördülő átlag hozzáadása (alsó panel)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rolling_avg,
            mode='lines',
            name='Prediction Rolling Avg',
            line=dict(color='blue', width=1),
            fill='tozeroy',  # Töltés a nulláig
        ),
        row=2, col=1
    )
    
    # Diagram formázása
    fig.update_layout(
        title=title,
        hovermode="x unified",  # Egységes hover mindkét panelen
        template="plotly_white",
        height=1000,  # Magasabb ábra
        width=1200,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Y-tengelyek címkézése
    fig.update_yaxes(title_text="Price", row=1, col=1)
    
    # Dinamikus skála az alsó panelen (5-95% percentilis alapján)
    p05 = np.percentile(rolling_avg, 0.1)
    p95 = np.percentile(rolling_avg, 99.9)
    
    fig.update_yaxes(
        title_text="Prediction Rolling Avg",
        row=2, col=1,
        range=[p05, p95]  # Dinamikus skála az értékek 5-95%-os tartományára
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
        filename = os.path.basename(file)
        model_name = filename.split('_')[0]
        dates = "_".join(filename.split('_')[1:]).replace('.parquet', '')
        print(f"{i}. {model_name} - {dates}")

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
    
    # Példa fájl kiválasztása
    prediction_files = find_prediction_files(prediction_dir)
    if prediction_files:
        # Legfrissebb fájl kiválasztása
        selected_file = prediction_files[-1]
        print(f"\nKiválasztott fájl: {selected_file}")
        
        # Adatok betöltése
        timestamps, prices, predictions = load_predictions_from_parquet(selected_file)
        print(f"Betöltött adatpontok száma: {len(timestamps)}")
        
        # Modell név kiolvasása a fájlnévből
        model_name = os.path.basename(selected_file).split("_")[0]
        title = f"Price vs. DeepLOB predictions - {model_name}"
        
        # Vizualizáció létrehozása
        fig = plot_model_sentiment(
            timestamps=timestamps,
            prices=prices,
            predictions=predictions,
            window_size='15min',  # 5 perces gördülő ablak
            title=title
        )
        
        # Megjelenítés
        fig.show()
        
        # Automatikus mentés HTML formátumban
        output_dir = "./szakdolgozat-high-freq-btc-prediction/results/visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # HTML mentés (interaktív)
        html_path = os.path.join(output_dir, f"prediction_rolling_avg_{model_name}.html")
        fig.write_html(html_path)
        print(f"\nInteraktív ábra mentve: {html_path}")
        
    else:
        print(f"Nem található előrejelzés fájl a {prediction_dir} mappában.")
        

