#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepLOB kereskedési stratégia rugalmas futtatása (debugging-ra optimalizálva)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trading_strategy import DeepLOBTradingStrategy

def plot_strategy_results(strategy, output_path=None):
    """
    Vizualizálja a stratégia eredményeit
    """
    if not strategy.trades or not strategy.data is not None:
        print("Nincs mit ábrázolni")
        return
    
    # Create a basic performance plot
    trades_df = pd.DataFrame(strategy.trades)
    
    # Create interactive plotly figure
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Árfolyam és Kereskedések', 'Z-Score (Irányjelző)', 'Kumulatív Hozam'),
                        row_heights=[0.5, 0.25, 0.25])
                        
    # Add price data
    fig.add_trace(
        go.Scatter(x=strategy.data.index, y=strategy.data['price'],
                   name='Árfolyam', line=dict(color='royalblue', width=1)),
        row=1, col=1
    )
    
    # Add trades as markers
    for _, trade in trades_df.iterrows():
        color = 'green' if trade['profit_pct'] > 0 else 'red'
        
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_time']], 
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(symbol='triangle-up' if trade['direction']=='long' else 'triangle-down', 
                            size=10, color=color),
                name=f"{trade['direction'].capitalize()} Entry",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Exit marker
        fig.add_trace(
            go.Scatter(
                x=[trade['exit_time']], 
                y=[trade['exit_price']],
                mode='markers',
                marker=dict(symbol='x', size=8, color=color),
                name=f"Exit ({trade['exit_reason']})",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add z-score
    fig.add_trace(
        go.Scatter(x=strategy.data.index, y=strategy.data['z_score'],
                   name='Z-Score', line=dict(color='purple', width=1)),
        row=2, col=1
    )
    
    # Add threshold lines
    threshold = strategy.z_score_threshold
    fig.add_trace(
        go.Scatter(x=strategy.data.index, y=[threshold] * len(strategy.data),
                   name=f'Threshold (+{threshold})', line=dict(color='green', width=1, dash='dash')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=strategy.data.index, y=[-threshold] * len(strategy.data),
                   name=f'Threshold (-{threshold})', line=dict(color='red', width=1, dash='dash')),
        row=2, col=1
    )
    
    # Add cumulative returns
    if len(trades_df) > 0:
        cumulative_returns = (1 + trades_df['profit_pct']).cumprod() - 1
        
        fig.add_trace(
            go.Scatter(x=trades_df['exit_time'], y=cumulative_returns,
                    name='Kumulatív Hozam', line=dict(color='green', width=2)),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='DeepLOB Burst Detection Kereskedési Stratégia',
        height=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save the figure if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Ábra mentve: {output_path}")
    
    return fig

def run_strategy(strategy_params=None):
    """Stratégia futtatása flexibilis paraméterekkel debugging-hoz"""
    # Alapértelmezett stratégia paraméterek
    default_params = {
        # Adatbetöltés
        'predictions_file': 'predictions_deeplob_single_parallel_f1_0_20250303_20250307.parquet',
        'output_dir': './strategy_results',
        
        # Stratégia paraméterek
        'ema_span': 2000,                # EMA ablakméret tickekben
        'z_score_threshold': 1.5,        # Z-score küszöbérték
        'kleinberg_gamma': 1.5,          # Kleinberg gamma paraméter
        'max_holding_ticks': 100,        # Maximum tartási időszak tickekben
        'atr_window': 1000,              # ATR számítási ablak
        'sl_atr_factor': 1.2,            # Stop-loss ATR szorzó
        'tp_atr_factor': 2.4,            # Take-profit ATR szorzó
        'commission_per_trade': 0.001,   # Jutalék kereskedésenként
        'slippage_ticks': 1              # Slippage tickekben
    }
    
    # Felülírás a megadott paraméterekkel ha van
    if strategy_params:
        default_params.update(strategy_params)
    
    params = default_params
    
    # Könyvtár létrehozása
    os.makedirs(params['output_dir'], exist_ok=True)
    
    print(f"Stratégia indítása: {params['predictions_file']}")
    
    # Stratégia inicializálása - csak a stratégia-specifikus paramétereket adjuk át
    strategy_specific_params = {k: v for k, v in params.items() 
                               if k not in ['predictions_file', 'output_dir']}
    
    strategy = DeepLOBTradingStrategy(**strategy_specific_params)
    
    # Adatok betöltése és előfeldolgozása
    data = strategy.load_predictions(params['predictions_file'])
    strategy.preprocess_data(data)
    
    # Burst detektálás
    up_bursts, down_bursts = strategy.detect_bursts()
    
    # Kereskedések generálása
    trades = strategy.generate_trades(up_bursts, down_bursts)
    
    # Teljesítmény elemzése
    results = strategy.analyze_performance()
    
    # Eredmények kiírása
    results_path = os.path.join(params['output_dir'], 'results.txt')
    with open(results_path, 'w') as f:
        f.write("DeepLOB Kereskedési Stratégia Eredmények\n")
        f.write("=====================================\n\n")
        f.write(f"Előrejelzések fájlja: {params['predictions_file']}\n")
        f.write(f"EMA ablak: {params['ema_span']}\n")
        f.write(f"Z-score küszöb: {params['z_score_threshold']}\n")
        f.write(f"Kleinberg gamma: {params['kleinberg_gamma']}\n")
        f.write(f"Max tartási idő: {params['max_holding_ticks']}\n\n")
        
        f.write("Kereskedési eredmények:\n")
        f.write(f"Összes kereskedés: {results.get('total_trades', 0)}\n")
        f.write(f"Nyertes kereskedések: {results.get('winning_trades', 0)} ({results.get('win_rate', 0)*100:.2f}%)\n")
        f.write(f"Teljes hozam: {results.get('total_return', 0)*100:.2f}%\n")
        f.write(f"Átlagos hozam kereskedésenként: {results.get('avg_return', 0)*100:.2f}%\n")
        f.write(f"Maximum drawdown: {results.get('max_drawdown', 0)*100:.2f}%\n\n")
        
        f.write("Long pozíciók nyerési aránya: {:.2f}%\n".format(results.get('long_win_rate', 0)*100))
        f.write("Short pozíciók nyerési aránya: {:.2f}%\n\n".format(results.get('short_win_rate', 0)*100))
        
        f.write("Kilépési statisztikák:\n")
        exit_stats = results.get('exit_stats', {})
        for reason, stats in exit_stats.items():
            f.write(f"  {reason}: {stats.get('count', 0)} db, átlagos hozam: {stats.get('mean', 0)*100:.2f}%, összhozam: {stats.get('sum', 0)*100:.2f}%\n")
    
    print(f"Eredmények mentve: {results_path}")
    
    # Ábrák készítése
    viz_path = os.path.join(params['output_dir'], 'strategy_visualization.html')
    plot_strategy_results(strategy, viz_path)
    
    print("Kereskedési stratégia futtatás kész!")
    
    # Visszaadjuk a stratégia objektumot és az eredményeket további elemzéshez
    return strategy, results, up_bursts, down_bursts, trades

# Ha közvetlenül futtatjuk a scriptet (nem importáljuk)
if __name__ == "__main__":
    # Itt állíthatod be a paramétereket közvetlenül a kódban
    # Ez a rész könnyen debuggolható
    my_params = {
        # Fő paraméterek
        'predictions_file': '/home/andras/btc-project/szakdolgozat-high-freq-btc-prediction/results/deeplob/predictions_deeplob_single_parallel_f1_0_20250303_20250307.parquet',
        'output_dir': '/home/andras/btc-project/szakdolgozat-high-freq-btc-prediction/strategy_results',
        
        # Stratégia finomhangolása
        'ema_span': 2000,
        'z_score_threshold': 1.5,
        'kleinberg_gamma': 1.5,
        'max_holding_ticks': 100,
        
        # Kockázatkezelés
        'sl_atr_factor': 1.2,
        'tp_atr_factor': 2.4
    }
    
    # Stratégia futtatása
    strategy, results, up_bursts, down_bursts, trades = run_strategy(my_params)
    
    # Itt további elemzéseket végezhetsz a visszaadott objektumokon
    # Például összes kereskedés és nyereségesek aránya
    win_rate = results.get('win_rate', 0) * 100
    print(f"\nÖsszes kereskedés: {results.get('total_trades', 0)}")
    print(f"Nyerési arány: {win_rate:.2f}%")
    print(f"Teljes hozam: {results.get('total_return', 0) * 100:.2f}%")
    
    # A debugoláshoz itt hozzáférsz minden objektumhoz
