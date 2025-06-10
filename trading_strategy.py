#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Trading Strategy based on DeepLOB predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class SimpleTradingStrategy:
    """
    Simple trading strategy based on DeepLOB model predictions.
    Buys on +1, sells on -1, holds on 0, closes positions at end of day.
    Uses raw_price for trading calculations.
    """
    
    def __init__(self):
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
        
        # Print basic information
        print(f"Loaded data shape: {df.shape}")
        
        # Set received_time as index
        df = df.set_index('received_time')
        
        # Print time range
        print(f"Time range: {df.index.min()} - {df.index.max()}")
        
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
        
        # Convert predictions to direction: 0->-1, 1->0, 2->1
        df['direction'] = df['prediction'].replace({0: -1, 1: 0, 2: 1})
        
        # Store the processed data
        self.data = df
        
        print("Data preprocessing complete.")
        
        return df
    
    def generate_trades(self, signal_threshold: int = 1) -> List[dict]:
        """
        Generate trades based on predictions
        - Buy on +1
        - Sell on -1
        - Hold on 0
        - Close all positions at end of day
        - Only open/close positions when signal_threshold consecutive signals are detected
        
        Args:
            signal_threshold: Number of consecutive signals required to open/close a position
                             Default is 1 (original strategy)
        
        Returns:
            List of trade dictionaries
        """
        print(f"Generating trades with signal threshold: {signal_threshold}...")
        
        # Validate signal threshold
        if not isinstance(signal_threshold, int) or signal_threshold < 1:
            print("Warning: signal_threshold must be a positive integer. Using default value of 1.")
            signal_threshold = 1
        
        trades = []
        current_position = None
        entry_price = None
        entry_time = None
        
        # Group data by day
        daily_groups = self.data.groupby(self.data.index.date)
        
        # Process each day's data
        for date, day_data in daily_groups:
            # Reset position at start of each day
            if current_position is not None:
                # Close position at end of previous day
                exit_price = day_data['raw_price'].iloc[0]
                exit_time = day_data.index[0]
                
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'direction': current_position,
                    'profit_pct': (exit_price / entry_price - 1) if current_position == 'long' else (1 - exit_price / entry_price),
                    'exit_reason': 'end_of_day'
                })
                
                current_position = None
                entry_price = None
                entry_time = None
            
            # Counters for consecutive signals
            consecutive_up = 0
            consecutive_down = 0
            
            # Process each tick in the day
            for idx, row in day_data.iterrows():
                # Update consecutive signal counters
                if row['direction'] == 1:  # Up signal
                    consecutive_up += 1
                    consecutive_down = 0
                elif row['direction'] == -1:  # Down signal
                    consecutive_down += 1
                    consecutive_up = 0
                else:  # Stable (0) resets both counters
                    consecutive_up = 0
                    consecutive_down = 0
                
                if current_position is None:
                    # No position, look for entry when threshold is reached
                    if consecutive_up >= signal_threshold:  # Buy signal threshold met
                        current_position = 'long'
                        entry_price = row['raw_price']
                        entry_time = idx
                        # Reset counter after action
                        consecutive_up = 0
                    elif consecutive_down >= signal_threshold:  # Sell signal threshold met
                        current_position = 'short'
                        entry_price = row['raw_price']
                        entry_time = idx
                        # Reset counter after action
                        consecutive_down = 0
                else:
                    # Have position, look for exit when threshold is reached
                    exit_condition = (current_position == 'long' and consecutive_down >= signal_threshold) or \
                                    (current_position == 'short' and consecutive_up >= signal_threshold)
                    
                    if exit_condition:
                        # Close position
                        trades.append({
                            'entry_time': entry_time,
                            'entry_price': entry_price,
                            'exit_time': idx,
                            'exit_price': row['raw_price'],
                            'direction': current_position,
                            'profit_pct': (row['raw_price'] / entry_price - 1) if current_position == 'long' else (1 - row['raw_price'] / entry_price),
                            'exit_reason': 'signal_reversal'
                        })
                        current_position = None
                        entry_price = None
                        entry_time = None
                        # Reset counters after action
                        consecutive_up = 0
                        consecutive_down = 0
            
            # Close any remaining position at end of day
            if current_position is not None:
                exit_price = day_data['raw_price'].iloc[-1]
                exit_time = day_data.index[-1]
                
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'direction': current_position,
                    'profit_pct': (exit_price / entry_price - 1) if current_position == 'long' else (1 - exit_price / entry_price),
                    'exit_reason': 'end_of_day'
                })
                
                current_position = None
                entry_price = None
                entry_time = None
        
        self.trades = trades
        print(f"Generated {len(trades)} trades")
        
        return trades
    
    def calculate_buy_and_hold(self) -> float:
        """
        Calculate buy-and-hold return
        Buy at first price, sell at last price
        
        Returns:
            Total return as percentage
        """
        if self.data is None or len(self.data) < 2:
            return 0.0
            
        first_price = self.data['raw_price'].iloc[0]
        last_price = self.data['raw_price'].iloc[-1]
        
        return (last_price / first_price) - 1
    
    def analyze_performance(self) -> Dict:
        """
        Analyze trading performance and compare with buy-and-hold
        """
        if not self.trades:
            return {}
            
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate strategy metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_pct'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = trades_df['profit_pct'].sum()
        avg_return = trades_df['profit_pct'].mean()
        
        # Calculate Sharpe ratio (assuming daily returns)
        daily_returns = trades_df.groupby(trades_df['entry_time'].dt.date)['profit_pct'].sum()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
        
        # Calculate buy-and-hold return
        buy_and_hold_return = self.calculate_buy_and_hold()
        
        # Calculáljuk a kereskedések átlagos és medián hosszát másodpercekben
        if len(trades_df) > 0:
            # Számítsuk ki a másodpercek tört részét is
            trades_df['duration_seconds'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds()
            avg_trade_duration = trades_df['duration_seconds'].mean()
            median_trade_duration = trades_df['duration_seconds'].median()
        else:
            avg_trade_duration = 0.0
            median_trade_duration = 0.0
        
        # Calculate cumulative returns
        cumulative_returns = (1 + trades_df['profit_pct']).cumprod() - 1
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / (1 + peak)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return': avg_return,
            'avg_trade_duration': avg_trade_duration,
            'median_trade_duration': median_trade_duration,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_and_hold_return': buy_and_hold_return,
            'outperformance': total_return - buy_and_hold_return
        }
        
        self.results = results
        return results