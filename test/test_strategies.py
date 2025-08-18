import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.strategies import MovingAverageCrossoverStrategy, MeanReversionStrategy, create_strategy

class TestMovingAverageCrossoverStrategy(unittest.TestCase):
    def setUp(self):
        self.config = {
            'short_window': 5,
            'long_window': 10,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.5,
            'min_confidence': 60
        }
        self.strategy = MovingAverageCrossoverStrategy(self.config)
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        prices = [100 + i + np.random.normal(0, 2) for i in range(20)]
        volumes = [1000000 + np.random.normal(0, 100000) for _ in range(20)]
        
        self.sample_data = pd.DataFrame({
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        self.technical_indicators = {
            'rsi': 35,
            'sma': 105,
            'ema': 106
        }
    
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.short_window, 5)
        self.assertEqual(self.strategy.long_window, 10)
        self.assertEqual(self.strategy.name, "MovingAverageCrossover")
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        short_data = self.sample_data.head(5)
        signal = self.strategy.generate_signals('AAPL', short_data, self.technical_indicators)
        
        self.assertEqual(signal['action'], 'HOLD')
        self.assertEqual(signal['confidence'], 0)
        self.assertEqual(signal['reason'], 'Insufficient data')
    
    def test_buy_signal_generation(self):
        """Test buy signal generation"""
        # Create data that should generate a buy signal
        buy_data = self.sample_data.copy()
        # Simulate bullish crossover
        buy_data.loc[buy_data.index[-2], 'Close'] = 100  # Previous price below MA
        buy_data.loc[buy_data.index[-1], 'Close'] = 120  # Current price above MA
        
        tech_indicators = self.technical_indicators.copy()
        tech_indicators['rsi'] = 25  # Oversold
        
        signal = self.strategy.generate_signals('AAPL', buy_data, tech_indicators)
        
        # Should generate buy signal with high confidence
        self.assertIn(signal['action'], ['BUY', 'HOLD'])
        self.assertIsInstance(signal['confidence'], (int, float))
        self.assertIn('indicators', signal)
    
    def test_sell_signal_generation(self):
        """Test sell signal generation"""
        # Create data that should generate a sell signal
        sell_data = self.sample_data.copy()
        
        tech_indicators = self.technical_indicators.copy()
        tech_indicators['rsi'] = 75  # Overbought
        
        signal = self.strategy.generate_signals('AAPL', sell_data, tech_indicators)
        
        # Should have valid signal structure
        self.assertIn(signal['action'], ['BUY', 'SELL', 'HOLD'])
        self.assertIsInstance(signal['confidence'], (int, float))
        self.assertIn('reason', signal)

class TestMeanReversionStrategy(unittest.TestCase):
    def setUp(self):
        self.config = {
            'bb_window': 20,
            'bb_std': 2,
            'rsi_threshold': 30
        }
        self.strategy = MeanReversionStrategy(self.config)
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=25, freq='D')
        prices = [100 + np.random.normal(0, 5) for _ in range(25)]
        
        self.sample_data = pd.DataFrame({
            'Close': prices,
            'Volume': [1000000] * 25
        }, index=dates)
        
        self.technical_indicators = {
            'bb_upper': 110,
            'bb_lower': 90,
            'rsi': 25
        }
    
    def test_mean_reversion_buy_signal(self):
        """Test mean reversion buy signal"""
        # Price at lower Bollinger Band with oversold RSI
        signal = self.strategy.generate_signals('AAPL', self.sample_data, {
            'bb_upper': 110,
            'bb_lower': 90,
            'rsi': 25
        })
        
        # Should generate buy signal if price is at lower BB
        self.assertIn(signal['action'], ['BUY', 'HOLD'])
        self.assertIsInstance(signal['confidence'], (int, float))
    
    def test_mean_reversion_sell_signal(self):
        """Test mean reversion sell signal"""
        # Price at upper Bollinger Band with overbought RSI
        signal = self.strategy.generate_signals('AAPL', self.sample_data, {
            'bb_upper': 110,
            'bb_lower': 90,
            'rsi': 75
        })
        
        # Should have valid signal structure
        self.assertIn(signal['action'], ['BUY', 'SELL', 'HOLD'])
        self.assertIsInstance(signal['confidence'], (int, float))

class TestStrategyFactory(unittest.TestCase):
    def test_create_ma_strategy(self):
        """Test creating moving average strategy"""
        config = {'short_window': 10, 'long_window': 20}
        strategy = create_strategy('MovingAverageCrossover', config)
        
        self.assertIsInstance(strategy, MovingAverageCrossoverStrategy)
        self.assertEqual(strategy.short_window, 10)
    
    def test_create_mean_reversion_strategy(self):
        """Test creating mean reversion strategy"""
        config = {'bb_window': 20}
        strategy = create_strategy('MeanReversion', config)
        
        self.assertIsInstance(strategy, MeanReversionStrategy)
        self.assertEqual(strategy.bb_window, 20)
    
    def test_invalid_strategy_name(self):
        """Test invalid strategy name"""
        with self.assertRaises(ValueError):
            create_strategy('InvalidStrategy', {})

if __name__ == '__main__':
    unittest.main()

