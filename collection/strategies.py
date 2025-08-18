# src/strategies.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class TradingStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
    
    @abstractmethod
    def generate_signal(self, data: Dict, indicators: Dict) -> str:
        """
        Generate trading signal based on data and indicators
        Returns: 'BUY', 'SELL', or 'HOLD'
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, portfolio_value: float, risk_per_trade: float) -> int:
        """Calculate number of shares to trade"""
        pass

class MovingAverageCrossover(TradingStrategy):
    def __init__(self, short_window: int = 10, long_window: int = 20):
        super().__init__("Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
        self.parameters = {
            'short_window': short_window,
            'long_window': long_window
        }
    
    def generate_signal(self, data: Dict, indicators: Dict) -> str:
        """Generate signal based on moving average crossover"""
        current_price = data.get('price')
        sma_short = indicators.get(f'sma_{self.short_window}')
        sma_long = indicators.get(f'sma_{self.long_window}')
        
        if not all([current_price, sma_short, sma_long]):
            return 'HOLD'
        
        # Buy signal: short MA crosses above long MA and price is above short MA
        if sma_short > sma_long and current_price > sma_short:
            return 'BUY'
        
        # Sell signal: short MA crosses below long MA or price drops below short MA
        elif sma_short < sma_long or current_price < sma_short:
            return 'SELL'
        
        return 'HOLD'
    
    def calculate_position_size(self, portfolio_value: float, risk_per_trade: float, price: float) -> int:
        """Calculate position size based on fixed percentage of portfolio"""
        position_value = portfolio_value * risk_per_trade
        shares = int(position_value / price)
        return max(1, shares)  # At least 1 share

class MeanReversion(TradingStrategy):
    def __init__(self, lookback_period: int = 20, z_threshold: float = 2.0):
        super().__init__("Mean Reversion")
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.parameters = {
            'lookback_period': lookback_period,
            'z_threshold': z_threshold
        }
    
    def generate_signal(self, data: Dict, indicators: Dict) -> str:
        """Generate signal based on mean reversion"""
        current_price = data.get('price')
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        sma = indicators.get('sma')
        
        if not all([current_price, bb_upper, bb_lower, sma]):
            return 'HOLD'
        
        # Calculate z-score
        bb_width = bb_upper - bb_lower
        if bb_width == 0:
            return 'HOLD'
        
        z_score = (current_price - sma) / (bb_width / 4)  # Approximate z-score
        
        # Buy signal: price is oversold (below lower Bollinger Band)
        if z_score < -self.z_threshold:
            return 'BUY'
        
        # Sell signal: price is overbought (above upper Bollinger Band)
        elif z_score > self.z_threshold:
            return 'SELL'
        
        return 'HOLD'
    
    def calculate_position_size(self, portfolio_value: float, risk_per_trade: float, price: float) -> int:
        """Calculate position size for mean reversion strategy"""
        position_value = portfolio_value * risk_per_trade * 0.5  # More conservative
        shares = int(position_value / price)
        return max(1, shares)

class RSIMomentum(TradingStrategy):
    def __init__(self, rsi_oversold: float = 30, rsi_overbought: float = 70):
        super().__init__("RSI Momentum")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.parameters = {
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought
        }
    
    def generate_signal(self, data: Dict, indicators: Dict) -> str:
        """Generate signal based on RSI momentum"""
        rsi = indicators.get('rsi')
        
        if rsi is None:
            return 'HOLD'
        
        # Buy signal: RSI indicates oversold condition
        if rsi < self.rsi_oversold:
            return 'BUY'
        
        # Sell signal: RSI indicates overbought condition
        elif rsi > self.rsi_overbought:
            return 'SELL'
        
        return 'HOLD'
    
    def calculate_position_size(self, portfolio_value: float, risk_per_trade: float, price: float) -> int:
        """Calculate position size based on RSI strength"""
        position_value = portfolio_value * risk_per_trade
        shares = int(position_value / price)
        return max(1, shares)