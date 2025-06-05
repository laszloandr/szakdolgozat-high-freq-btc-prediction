# trading_strategy.py - Robust trading strategy using DeepLOB predictions and burst detection

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepLOB Trading Strategy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from burst_detection import burst_detection, enumerate_bursts
import os
import warnings
warnings.filterwarnings('ignore')

class DeepLOBTradingStrategy:
    """
    Trading strategy based on DeepLOB model predictions using Kleinberg burst detection.
    """
    
    def __init__(
        self,
        ema_span: int = 2000,           # EMA window size in ticks
        z_score_threshold: float = 1.5,  # Threshold for considering significant z-scores
        kleinberg_gamma: float = 1.5,    # Gamma parameter for Kleinberg burst detection
        max_holding_ticks: int = 100,    # Maximum holding period in ticks
        atr_window: int = 1000,          # ATR calculation window
        sl_atr_factor: float = 1.2,      # Stop-loss as multiplier of ATR
        tp_atr_factor: float = 2.4,      # Take-profit as multiplier of ATR
        commission_per_trade: float = 0.001,  # Commission per trade in percentage
        slippage_ticks: int = 1          # Slippage in ticks
    ):
        self.ema_span = ema_span
        self.z_score_threshold = z_score_threshold
        self.kleinberg_gamma = kleinberg_gamma
        self.max_holding_ticks = max_holding_ticks
        self.atr_window = atr_window
        self.sl_atr_factor = sl_atr_factor
        self.tp_atr_factor = tp_atr_factor
        self.commission_per_trade = commission_per_trade
        self.slippage_ticks = slippage_ticks
        
        # Store processed data
        self.data = None
        self.trades = []
        self.results = {}
        
    def load_predictions(self, file_path: str) -> pd.DataFrame:
        """
        Load DeepLOB predictions from parquet file
        
        Args:
            file_path: Path to the parquet file containing predictions
            
        Returns:
            DataFrame with predictions and prices
        """
        print(f"Loading predictions from {file_path}")
        df = pd.read_parquet(file_path)
        
        # Print basic information about the loaded data
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Time range: {df.index.min()} - {df.index.max()}")
        print(f"Prediction distribution: \n{df['prediction'].value_counts(normalize=True)}")
        
        # Rename price column if needed - supporting both 'price' and 'ask_0_price' formats
        if 'ask_0_price' in df.columns and 'price' not in df.columns:
            print("Renaming 'ask_0_price' column to 'price'")
            df['price'] = df['ask_0_price']
        
        return df
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data to calculate the necessary indicators
        
        Args:
            df: DataFrame with predictions and prices
            
        Returns:
            Processed DataFrame with additional indicators
        """
        print("Preprocessing data...")
        
        # Convert predictions to direction strength: 0->-1, 1->0, 2->1
        df['direction'] = df['prediction'].replace({0: -1, 1: 0, 2: 1})
        
        # Calculate EMA for noise filtering
        df['direction_ema'] = df['direction'].ewm(span=self.ema_span, adjust=False).mean()
        
        # Calculate rolling statistics for z-score normalization (using 3-day window)
        # Assuming we have timestamps in the index
        # Calculate approximate ticks in 3 days (6Hz = 6*60*60*24*3 = 1,555,200 ticks)
        window_size = 6 * 60 * 60 * 24 * 3  # 3 days at 6Hz
        
        # Use appropriate window size based on data length
        actual_window = min(window_size, len(df) // 2)
        
        # Calculate rolling mean and std
        df['ema_mean'] = df['direction_ema'].rolling(window=actual_window, min_periods=100).mean()
        df['ema_std'] = df['direction_ema'].rolling(window=actual_window, min_periods=100).std()
        
        # Handle first rows where rolling window doesn't have enough data
        df['ema_mean'] = df['ema_mean'].fillna(df['direction_ema'].iloc[:actual_window].mean())
        df['ema_std'] = df['ema_std'].fillna(df['direction_ema'].iloc[:actual_window].std())
        
        # Avoid division by zero
        df['ema_std'] = df['ema_std'].replace(0, df['ema_std'].mean())
        
        # Calculate z-score
        df['z_score'] = (df['direction_ema'] - df['ema_mean']) / df['ema_std']
        
        # Calculate ATR (Average True Range)
        if 'high' in df.columns and 'low' in df.columns:
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    np.abs(df['high'] - df['close'].shift(1)),
                    np.abs(df['low'] - df['close'].shift(1))
                )
            )
        else:
            # If we don't have high/low data, use a simplification based on price changes
            df['tr'] = np.abs(df['price'] - df['price'].shift(1))
            
        df['atr'] = df['tr'].rolling(window=self.atr_window).mean()
        df['atr'] = df['atr'].fillna(df['tr'].iloc[:self.atr_window].mean())
        
        # Mark potential up and down events based on z-score threshold
        df['up_signal'] = df['z_score'] > self.z_score_threshold
        df['down_signal'] = df['z_score'] < -self.z_score_threshold
        
        # Store the processed data
        self.data = df
        
        print("Data preprocessing complete.")
        
        return df
    
    def detect_bursts(self) -> Tuple[List[dict], List[dict]]:
        """
        Detect bursts in up and down signals using Kleinberg algorithm
        
        Returns:
            Tuple of (up_bursts, down_bursts)
        """
        print("Detecting bursts...")
        
        if self.data is None:
            raise ValueError("Data not processed yet. Call preprocess_data first.")
        
        # ---------- 1. Egyszerűsített megközelítés, mint a példakódban ----------
        prediction = self.data['direction'].astype(int)
        
        # Az indexeket használjuk események helyett (ahogy a példakódban is tettük)
        up_indices = np.where((prediction == 1) & (self.data['up_signal']))[0]
        down_indices = np.where((prediction == -1) & (self.data['down_signal']))[0]
        
        print(f"Found {len(up_indices)} up signals and {len(down_indices)} down signals")
        
        up_bursts = []
        down_bursts = []
        
        # Segédfüggvény: csoportosítja a közel eső indexeket burst-ökké
        # Ez egy nagyon egyszerű megoldás, ami lecseréli a Kleinberg algoritmust
        def group_into_bursts(indices, min_gap=5):
            if len(indices) < 3:  # Túl kevés esemny
                return []
                
            bursts = []
            current_burst = {'start': indices[0], 'end': indices[0], 'intensity': 1}
            
            for i in range(1, len(indices)):
                idx = indices[i]
                
                # Ha a következő index elég közel van, kiterjesztünk a burst-öt
                if idx - current_burst['end'] <= min_gap:
                    current_burst['end'] = idx
                    current_burst['intensity'] = max(current_burst['intensity'], 1)
                else:
                    # Mentünk az előző burst-öt, és kezdünk egy újat
                    if current_burst['end'] - current_burst['start'] >= min_gap:
                        bursts.append(dict(current_burst))
                    current_burst = {'start': idx, 'end': idx, 'intensity': 1}
                    
            # Az utolsó burst-öt is hozzáadjuk, ha elég hosszú
            if current_burst['end'] - current_burst['start'] >= min_gap:
                bursts.append(current_burst)
                
            return bursts
        
        # UP burst-ök detektálása
        if len(up_indices) > 3:
            print("Detecting UP bursts using simplified approach...")
            up_bursts_raw = group_into_bursts(up_indices, min_gap=self.kleinberg_gamma * 10) 
            print(f"Detected {len(up_bursts_raw)} UP bursts")
            
            # Konvertálás a teljes stratégiával kompatibilis dict formátumba
            for burst in up_bursts_raw:
                up_bursts.append({
                    'start': burst['start'],
                    'end': burst['end'],
                    'intensity': burst['intensity'],
                    'direction': 'up',
                    'start_time': self.data.index[burst['start']],
                    'end_time': self.data.index[burst['end']],
                    'duration': burst['end'] - burst['start']
                })
        
        # DOWN burst-ök detektálása
        if len(down_indices) > 3:
            print("Detecting DOWN bursts using simplified approach...")
            down_bursts_raw = group_into_bursts(down_indices, min_gap=self.kleinberg_gamma * 10) 
            print(f"Detected {len(down_bursts_raw)} DOWN bursts")
            
            # Konvertálás a teljes stratégiával kompatibilis dict formátumba
            for burst in down_bursts_raw:
                down_bursts.append({
                    'start': burst['start'],
                    'end': burst['end'],
                    'intensity': burst['intensity'],
                    'direction': 'down',
                    'start_time': self.data.index[burst['start']],
                    'end_time': self.data.index[burst['end']],
                    'duration': burst['end'] - burst['start']
                })
                
        # Debug: burst-ök megjelenítése
        if up_bursts or down_bursts:
            print("\nBurst detection results:")
            print(f"UP bursts: {len(up_bursts)}, DOWN bursts: {len(down_bursts)}")
            if up_bursts:
                print(f"First UP burst: Start={up_bursts[0]['start']}, End={up_bursts[0]['end']}, Duration={up_bursts[0]['duration']}")
            if down_bursts:
                print(f"First DOWN burst: Start={down_bursts[0]['start']}, End={down_bursts[0]['end']}, Duration={down_bursts[0]['duration']}")
                
        return up_bursts, down_bursts
            
        return up_bursts, down_bursts
    
    def generate_trades(self, up_bursts: List[dict], down_bursts: List[dict]) -> List[dict]:
        """
        Generate trades based on burst detection
        
        Args:
            up_bursts: List of up burst dictionaries
            down_bursts: List of down burst dictionaries
            
        Returns:
            List of trade dictionaries
        """
        print("Generating trades from bursts...")
        
        # Combine and sort all bursts by start time
        all_bursts = sorted(up_bursts + down_bursts, key=lambda x: x['start'])
        
        trades = []
        active_trade = None
        
        for burst in all_bursts:
            # If we have an active trade
            if active_trade is not None:
                # Skip this burst if it starts before current active trade is closed
                if burst['start'] <= active_trade['exit_index']:
                    continue
            
            # New trade setup
            entry_index = burst['start']
            entry_price = self.data['price'].iloc[entry_index]
            
            # Add slippage
            if burst['direction'] == 'up':
                side = 'long'
                entry_price += self.slippage_ticks * self.data['atr'].iloc[entry_index] / 10  # Slippage
            else:
                side = 'short'
                entry_price -= self.slippage_ticks * self.data['atr'].iloc[entry_index] / 10  # Slippage
            
            # Calculate exit based on burst end or max holding period
            exit_index = min(burst['end'], entry_index + self.max_holding_ticks)
            exit_price = self.data['price'].iloc[exit_index]
            
            # Add slippage to exit as well
            if side == 'long':
                exit_price -= self.slippage_ticks * self.data['atr'].iloc[exit_index] / 10
            else:
                exit_price += self.slippage_ticks * self.data['atr'].iloc[exit_index] / 10
                
            # Calculate stop loss and take profit levels
            atr_value = self.data['atr'].iloc[entry_index]
            
            if side == 'long':
                stop_loss = entry_price - self.sl_atr_factor * atr_value
                take_profit = entry_price + self.tp_atr_factor * atr_value
            else:
                stop_loss = entry_price + self.sl_atr_factor * atr_value
                take_profit = entry_price - self.tp_atr_factor * atr_value
                
            # Check if stop loss or take profit is hit before the exit
            for i in range(entry_index + 1, exit_index + 1):
                price = self.data['price'].iloc[i]
                
                if side == 'long':
                    if price <= stop_loss:
                        exit_index = i
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                        break
                    elif price >= take_profit:
                        exit_index = i
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                        break
                else:  # short
                    if price >= stop_loss:
                        exit_index = i
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                        break
                    elif price <= take_profit:
                        exit_index = i
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                        break
            else:
                # If no SL/TP hit
                exit_reason = 'burst_end' if exit_index == burst['end'] else 'max_holding'
                
            # Calculate profit/loss
            if side == 'long':
                profit_pct = (exit_price / entry_price) - 1 - self.commission_per_trade * 2
            else:
                profit_pct = 1 - (exit_price / entry_price) - self.commission_per_trade * 2
                
            profit_ticks = (exit_price - entry_price) if side == 'long' else (entry_price - exit_price)
                
            # Create the trade
            trade = {
                'entry_index': entry_index,
                'entry_time': self.data.index[entry_index],
                'entry_price': entry_price,
                'exit_index': exit_index,
                'exit_time': self.data.index[exit_index],
                'exit_price': exit_price,
                'direction': side,
                'duration': exit_index - entry_index,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'profit_pct': profit_pct,
                'profit_ticks': profit_ticks,
                'exit_reason': exit_reason,
                'burst_intensity': burst['intensity'],
            }
            
            trades.append(trade)
            active_trade = trade
            
        self.trades = trades
        print(f"Generated {len(trades)} trades")
        
        return trades
    
    def analyze_performance(self) -> Dict:
        """
        Analyze trading performance
        """
        if not self.trades:
            return {}
            
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_pct'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = trades_df['profit_pct'].sum()
        avg_return = trades_df['profit_pct'].mean() 
        
        cumulative_returns = (1 + trades_df['profit_pct']).cumprod() - 1
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / (1 + peak)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']
        
        long_win_rate = len(long_trades[long_trades['profit_pct'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
        short_win_rate = len(short_trades[short_trades['profit_pct'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
        
        exit_stats = trades_df.groupby('exit_reason')['profit_pct'].agg(['count', 'mean', 'sum'])
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'max_drawdown': max_drawdown,
            'long_win_rate': long_win_rate, 
            'short_win_rate': short_win_rate,
            'exit_stats': exit_stats.to_dict() if not exit_stats.empty else {}
        }
        
        self.results = results
        return results