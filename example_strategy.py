#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Példa a DeepLOB kereskedési stratégia használatára
"""

import os
import pandas as pd
from trading_strategy import DeepLOBTradingStrategy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parquet fájl elérési útvonala
predictions_file = "predictions_deeplob_single_parallel_f1_0_20250303_20250307.parquet"

# Kimeneti könyvtár
output_dir = "strategy_results"
os.makedirs(output_dir, exist_ok=True)

# Stratégia paraméterek
params = {
    'ema_span': 2000,           # EMA ablak mérete tickekben
    'z_score_threshold': 1.5,   # Z-score küszöbérték
    'kleinberg_gamma': 1.5,     # Kleinberg gamma paraméter 
    'max_holding_ticks': 100,   # Maximum tartási idő
    'atr_window': 1000,         # ATR számítási ablak
    'sl_atr_factor': 1.2,       # Stop-loss ATR szorzó
    'tp_atr_factor': 2.4,       # Take-profit ATR szorzó
    'commission_per_trade': 0.001,  # Jutalék kereskedésenként
    'slippage_ticks': 1         # Slippage tickekben
}

# Stratégia inicializálása
strategy = DeepLOBTradingStrategy(**params)

# Adatok betöltése és előfeldolgozása
data = strategy.load_predictions(predictions_file)
processed_data = strategy.preprocess_data(data)

# Burst detektálás
print("\nBurst detektálás...")
up_bursts, down_bursts = strategy.detect_bursts()

# Burst-ök ábrázolása (opcionális)
def visualize_bursts(data, up_bursts, down_bursts, output_file=None):
    """Burst-ök vizualizálása"""
    fig = go.Figure()
    
    # Árfolyam
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['price'],
        name='Árfolyam',
        line=dict(color='royalblue', width=1)
    ))
    
    # UP bursts
    for burst in up_bursts:
        burst_data = data.iloc[burst['start']:burst['end']]
        opacity = 0.15 + 0.15 * burst['intensity'] if burst['intensity'] <= 3 else 0.6
        
        fig.add_trace(go.Scatter(
            x=burst_data.index,
            y=burst_data['price'],
            mode='lines',
            fill='tozeroy',
            fillcolor=f'rgba(26, 152, 80, {opacity})',  # Zöld
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hovertemplate=f"UP Burst (Intensity: {burst['intensity']:.1f})<br>%{{'xaxis.title.text'}}: %{{x}}<br>%{{'yaxis.title.text'}}: %{{y:.2f}}"
        ))
    
    # DOWN bursts
    for burst in down_bursts:
        burst_data = data.iloc[burst['start']:burst['end']]
        opacity = 0.15 + 0.15 * burst['intensity'] if burst['intensity'] <= 3 else 0.6
        
        fig.add_trace(go.Scatter(
            x=burst_data.index,
            y=burst_data['price'],
            mode='lines',
            fill='tozeroy',
            fillcolor=f'rgba(215, 48, 39, {opacity})',  # Piros
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hovertemplate=f"DOWN Burst (Intensity: {burst['intensity']:.1f})<br>%{{'xaxis.title.text'}}: %{{x}}<br>%{{'yaxis.title.text'}}: %{{y:.2f}}"
        ))
    
    # Layout beállítások
    fig.update_layout(
        title="Ár és Detektált Burst-ök",
        xaxis_title="Idő",
        yaxis_title="Árfolyam",
        hovermode="closest",
        legend_title="Típus"
    )
    
    if output_file:
        fig.write_html(output_file)
        print(f"Burst ábra mentve: {output_file}")
    
    return fig

# Burst-ök vizualizálása
burst_viz_file = os.path.join(output_dir, "bursts.html")
visualize_bursts(strategy.data, up_bursts, down_bursts, burst_viz_file)

# Kereskedések generálása
print("\nKereskedések generálása...")
trades = strategy.generate_trades(up_bursts, down_bursts)

# Teljesítmény elemzése
print("\nTeljesítmény elemzése...")
results = strategy.analyze_performance()

# Eredmények kiírása konzolra
print("\n====== Kereskedési Stratégia Eredmények ======")
print(f"Összes kereskedés: {results.get('total_trades', 0)}")
print(f"Nyertes kereskedések: {results.get('winning_trades', 0)} ({results.get('win_rate', 0)*100:.2f}%)")
print(f"Teljes hozam: {results.get('total_return', 0)*100:.2f}%")
print(f"Átlagos hozam kereskedésenként: {results.get('avg_return', 0)*100:.2f}%")
print(f"Maximum drawdown: {results.get('max_drawdown', 0)*100:.2f}%")

print("\nLong pozíciók nyerési aránya: {:.2f}%".format(results.get('long_win_rate', 0)*100))
print("Short pozíciók nyerési aránya: {:.2f}%".format(results.get('short_win_rate', 0)*100))

print("\nKilépési statisztikák:")
exit_stats = results.get('exit_stats', {})
for reason, stats in exit_stats.items():
    count = stats.get('count', 0)
    mean = stats.get('mean', 0) * 100
    sum_return = stats.get('sum', 0) * 100
    print(f"  {reason}: {count} db, átlagos hozam: {mean:.2f}%, összhozam: {sum_return:.2f}%")

# Kereskedési vizualizáció 
def visualize_trades(strategy, trades, output_file=None):
    """Kereskedések vizualizálása"""
    if not trades:
        print("Nincs mit ábrázolni")
        return
    
    trades_df = pd.DataFrame(trades)
    
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
    for i, trade in trades_df.iterrows():
        color = 'green' if trade['profit_pct'] > 0 else 'red'
        
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_time']], 
                y=[trade['entry_price']],
                mode='markers',
                marker=dict(symbol='triangle-up' if trade['direction']=='long' else 'triangle-down', 
                            size=10, color=color),
                name=f"Trade {i+1}: {trade['direction'].capitalize()} Entry",
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
                name=f"Trade {i+1}: Exit ({trade['exit_reason']})",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Connect entry and exit with a line
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_time'], trade['exit_time']], 
                y=[trade['entry_price'], trade['exit_price']],
                mode='lines',
                line=dict(color=color, width=1, dash='dot'),
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
        trades_df['cum_return'] = (1 + trades_df['profit_pct']).cumprod() - 1
        
        fig.add_trace(
            go.Scatter(x=trades_df['exit_time'], y=trades_df['cum_return'],
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
    
    if output_file:
        fig.write_html(output_file)
        print(f"Kereskedési ábra mentve: {output_file}")
    
    return fig

# Kereskedések vizualizálása
trades_viz_file = os.path.join(output_dir, "trades.html")
visualize_trades(strategy, trades, trades_viz_file)

print("\nA stratégia futtatása befejeződött!")
print(f"Az eredmények és vizualizációk a '{output_dir}' könyvtárban találhatók.")

# Paraméter-érzékenység vizsgálat (opcionális)
def parameter_sensitivity_analysis():
    """
    Paraméter-érzékenység vizsgálat különböző paraméter kombinációkra
    """
    # Paraméter grid definíciója
    z_thresholds = [1.2, 1.5, 1.8, 2.0]
    gammas = [1.3, 1.5, 1.7]
    
    results = []
    
    for z in z_thresholds:
        for g in gammas:
            print(f"\nTeszt: Z-threshold = {z}, Gamma = {g}")
            
            # Stratégia inicializálása az aktuális paraméterekkel
            test_strategy = DeepLOBTradingStrategy(
                ema_span=params['ema_span'],
                z_score_threshold=z,
                kleinberg_gamma=g,
                max_holding_ticks=params['max_holding_ticks']
            )
            
            # Adatok betöltése és előfeldolgozása
            test_strategy.preprocess_data(data.copy())
            
            # Burst detektálás
            up_b, down_b = test_strategy.detect_bursts()
            
            # Kereskedések generálása
            test_trades = test_strategy.generate_trades(up_b, down_b)
            
            # Teljesítmény elemzése
            perf = test_strategy.analyze_performance()
            
            # Eredmény mentése
            results.append({
                'z_threshold': z,
                'gamma': g,
                'total_trades': perf.get('total_trades', 0),
                'win_rate': perf.get('win_rate', 0),
                'total_return': perf.get('total_return', 0),
                'max_drawdown': perf.get('max_drawdown', 0)
            })
    
    # Eredmények DataFrame-be konvertálása
    results_df = pd.DataFrame(results)
    
    # Eredmények mentése
    results_file = os.path.join(output_dir, "parameter_sensitivity.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nParaméter-érzékenység eredmények mentve: {results_file}")
    
    return results_df

# Paraméter-érzékenységi elemzés futtatása (megjegyzésbe téve, mert időigényes lehet)
# sensitivity_results = parameter_sensitivity_analysis()
