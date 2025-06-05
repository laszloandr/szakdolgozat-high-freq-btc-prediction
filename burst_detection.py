#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementation of Kleinberg's burst detection algorithm.

Based on the paper:
Kleinberg, J. (2003). Bursty and hierarchical structure in streams.
Data Mining and Knowledge Discovery, 7(4), 373-397.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd


def burst_detection(
    events: np.ndarray,
    time_points: Optional[np.ndarray] = None,
    gamma: float = 1.5,
    s: float = 2,
    n_states: int = 3,
    min_burst_length: int = 200
) -> np.ndarray:
    """
    Detect bursts in a time series using Kleinberg's burst detection algorithm.
    
    Args:
        events: Binary array where 1 indicates event occurrence
        time_points: Optional array of time points corresponding to events
        gamma: Difficulty of moving up a level (cost parameter), higher = fewer bursts
        s: Rate multiplier, how many times faster is the burst state (typically 2 or 3)
        n_states: Number of burst states to consider (including no-burst state 0)
        min_burst_length: Minimum length of a burst to be considered valid
        
    Returns:
        Array with burst state for each time point (0 = no burst)
    """
    if time_points is None:
        time_points = np.arange(len(events))
    
    N = len(events)
    
    # Check if we have any events at all
    if np.sum(events) == 0:
        return np.zeros(N, dtype=int)
    
    # Calculate empirical event rate
    total_events = np.sum(events)
    baseline_rate = total_events / N
    
    # Avoid division by zero or very small rates
    if baseline_rate < 1e-6:
        baseline_rate = 1e-6
    
    # Calculate rates for different states
    rates = np.zeros(n_states)
    rates[0] = baseline_rate
    for j in range(1, n_states):
        rates[j] = baseline_rate * (s ** j)
    
    # Initialize state costs and transitions
    state_costs = np.zeros(n_states)
    for j in range(1, n_states):
        state_costs[j] = state_costs[j-1] + gamma * np.log(N)
    
    # Viterbi algorithm
    # Initialize cost and backpointer matrices
    viterbi_cost = np.zeros((N, n_states))
    backpointer = np.zeros((N, n_states), dtype=int)
    
    # Base case: t=0
    for j in range(n_states):
        if events[0] == 1:
            viterbi_cost[0, j] = -np.log(rates[j]) + state_costs[j]
        else:
            viterbi_cost[0, j] = -np.log(1 - rates[j]) + state_costs[j]
    
    # Dynamic programming to fill viterbi_cost and backpointer
    for t in range(1, N):
        for j in range(n_states):
            # Calculate cost for staying in the same state
            if events[t] == 1:
                emission_cost = -np.log(rates[j])
            else:
                emission_cost = -np.log(1 - rates[j])
            
            cost_same_state = viterbi_cost[t-1, j] + emission_cost
            
            # Initialize with cost of staying in the same state
            min_cost = cost_same_state
            min_state = j
            
            # Check if transitioning from a lower state is better
            if j > 0:
                cost_from_lower = viterbi_cost[t-1, j-1] + emission_cost + gamma
                if cost_from_lower < min_cost:
                    min_cost = cost_from_lower
                    min_state = j-1
            
            # Check if transitioning from a higher state is better
            if j < n_states - 1:
                cost_from_higher = viterbi_cost[t-1, j+1] + emission_cost + gamma
                if cost_from_higher < min_cost:
                    min_cost = cost_from_higher
                    min_state = j+1
                    
            viterbi_cost[t, j] = min_cost
            backpointer[t, j] = min_state
    
    # Find the best ending state
    last_state = np.argmin(viterbi_cost[N-1, :])
    
    # Backtrack to find the best state sequence
    states = np.zeros(N, dtype=int)
    states[N-1] = last_state
    
    for t in range(N-2, -1, -1):
        states[t] = backpointer[t+1, states[t+1]]
    
    # Filter out short bursts
    burst_runs = enumerate_bursts(states)
    for burst in burst_runs:
        if burst['level'] > 0 and burst['end'] - burst['start'] < min_burst_length:
            states[burst['start']:burst['end']+1] = 0
            
    return states


def enumerate_bursts(states: np.ndarray) -> List[Dict]:
    """
    Enumerate bursts from state sequence
    
    Args:
        states: Array of burst states from burst_detection
        
    Returns:
        List of dictionaries with burst information
    """
    bursts = []
    
    current_state = states[0]
    current_start = 0
    
    for i in range(1, len(states)):
        if states[i] != current_state:
            # Record the completed burst
            if current_state > 0:  # Only record if it was a burst state
                bursts.append({
                    'start': current_start,
                    'end': i - 1,
                    'level': int(current_state),
                    'duration': i - current_start
                })
            
            # Start a new burst
            current_state = states[i]
            current_start = i
    
    # Handle the last burst
    if current_state > 0:  # Only record if it was a burst state
        bursts.append({
            'start': current_start,
            'end': len(states) - 1,
            'level': int(current_state),
            'duration': len(states) - current_start
        })
    
    return bursts
