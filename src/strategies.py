import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yaml

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = "BaseStrategy"
    
    def generate_signals(self, symbol: str, historical_data: pd.DataFrame, 
                        technical_indicators: Dict) -> Dict:
        """Generate trading signals - to be implemented by subclasses"""
        raise NotImplementedError

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy with RSI and Volume confirmation"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "MovingAverageCrossover"
        self.short_window = config.get('short_window', 10)
        self.long_window = config.get('long_window', 20)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.min_confidence = config.get('min_confidence', 60)
    
    def generate_signals(self, symbol: str, historical_data: pd.DataFrame, 
                        technical_indicators: Dict) -> Dict:
        """Generate buy/sell signals based on MA crossover strategy"""
        
        if len(historical_data) < self.long_window:
            return {'action': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        # Calculate moving averages
        short_ma = historical_data['Close'].rolling(window=self.short_window).mean()
        long_ma = historical_data['Close'].rolling(window=self.long_window).mean()
        
        # Current values
        current_price = historical_data['Close'].iloc[-1]
        short_ma_current = short_ma.iloc[-1]
        long_ma_current = long_ma.iloc[-1]
        short_ma_prev = short_ma.iloc[-2] if len(short_ma) > 1 else short_ma_current
        long_ma_prev = long_ma.iloc[-2] if len(long_ma) > 1 else long_ma_current
        
        # Technical indicators
        rsi = technical_indicators.get('rsi', 50)
        
        # Volume analysis
        avg_volume = historical_data['Volume'].rolling(window=10).mean().iloc[-1]
        current_volume = historical_data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Generate signal
        signal = self._evaluate_signals(
            short_ma_current, long_ma_current, short_ma_prev, long_ma_prev,
            current_price, rsi, volume_ratio
        )
        
        return signal
    
    def _evaluate_signals(self, short_ma_current, long_ma_current, 
                         short_ma_prev, long_ma_prev, current_price, 
                         rsi, volume_ratio) -> Dict:
        """Evaluate all signals and return trading decision"""
        
        confidence = 0
        reasons = []
        
        # Moving Average Crossover signals
        bullish_cross = (short_ma_current > long_ma_current and 
                        short_ma_prev <= long_ma_prev)
        bearish_cross = (short_ma_current < long_ma_current and 
                        short_ma_prev >= long_ma_prev)
        
        # RSI signals
        rsi_oversold = rsi < self.rsi_oversold
        rsi_overbought = rsi > self.rsi_overbought
        
        # Volume confirmation
        volume_confirmed = volume_ratio >= self.volume_threshold
        
        # Buy signals
        if bullish_cross:
            confidence += 40
            reasons.append("Bullish MA crossover")
        
        if rsi_oversold:
            confidence += 30
            reasons.append("RSI oversold")
        
        if volume_confirmed and (bullish_cross or rsi_oversold):
            confidence += 20
            reasons.append("Volume confirmation")
        
        if current_price > short_ma_current > long_ma_current:
            confidence += 10
            reasons.append("Uptrend confirmed")
        
        # Sell signals
        sell_confidence = 0
        sell_reasons = []
        
        if bearish_cross:
            sell_confidence += 40
            sell_reasons.append("Bearish MA crossover")
        
        if rsi_overbought:
            sell_confidence += 30
            sell_reasons.append("RSI overbought")
        
        if volume_confirmed and (bearish_cross or rsi_overbought):
            sell_confidence += 20
            sell_reasons.append("Volume confirmation")
        
        # Decision logic
        if confidence >= self.min_confidence:
            return {
                'action': 'BUY',
                'confidence': confidence,
                'reason': '; '.join(reasons),
                'indicators': {
                    'short_ma': short_ma_current,
                    'long_ma': long_ma_current,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio
                }
            }
        elif sell_confidence >= self.min_confidence:
            return {
                'action': 'SELL',
                'confidence': sell_confidence,
                'reason': '; '.join(sell_reasons),
                'indicators': {
                    'short_ma': short_ma_current,
                    'long_ma': long_ma_current,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio
                }
            }
        else:
            return {
                'action': 'HOLD',
                'confidence': max(confidence, sell_confidence),
                'reason': 'No strong signals',
                'indicators': {
                    'short_ma': short_ma_current,
                    'long_ma': long_ma_current,
                    'rsi': rsi,
                    'volume_ratio': volume_ratio
                }
            }

class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy using Bollinger Bands"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = "MeanReversion"
        self.bb_window = config.get('bb_window', 20)
        self.bb_std = config.get('bb_std', 2)
        self.rsi_threshold = config.get('rsi_threshold', 30)
    
    def generate_signals(self, symbol: str, historical_data: pd.DataFrame, 
                        technical_indicators: Dict) -> Dict:
        """Generate signals based on mean reversion"""
        
        if len(historical_data) < self.bb_window:
            return {'action': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
        
        current_price = historical_data['Close'].iloc[-1]
        bb_upper = technical_indicators.get('bb_upper')
        bb_lower = technical_indicators.get('bb_lower')
        rsi = technical_indicators.get('rsi', 50)
        
        if bb_upper is None or bb_lower is None:
            return {'action': 'HOLD', 'confidence': 0, 'reason': 'Missing BB data'}
        
        confidence = 0
        reasons = []
        
        # Buy signal: price near lower band + oversold RSI
        if current_price <= bb_lower and rsi < self.rsi_threshold:
            confidence = 70
            reasons.append("Price at lower BB + RSI oversold")
            action = 'BUY'
        
        # Sell signal: price near upper band + overbought RSI
        elif current_price >= bb_upper and rsi > (100 - self.rsi_threshold):
            confidence = 70
            reasons.append("Price at upper BB + RSI overbought")
            action = 'SELL'
        
        else:
            action = 'HOLD'
            reasons.append("Price within BB range")
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': '; '.join(reasons),
            'indicators': {
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'current_price': current_price,
                'rsi': rsi
            }
        }

def create_strategy(strategy_name: str, config: Dict) -> BaseStrategy:
    """Factory function to create strategy instances"""
    strategies = {
        'MovingAverageCrossover': MovingAverageCrossoverStrategy,
        'MeanReversion': MeanReversionStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategies[strategy_name](config)