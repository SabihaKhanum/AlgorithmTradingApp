# src/data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional

class DataFetcher:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_cache = {}
        self.last_update = {}
    
    def get_historical_data(self, symbol: str, period: str = "6mo") -> pd.DataFrame:
        """Fetch historical data for backtesting"""
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """Get current market data with caching"""
        current_time = datetime.now()
        

        if (symbol in self.last_update and 
            (current_time - self.last_update[symbol]).seconds < 60):
            return self.data_cache[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                volume = hist['Volume'].iloc[-1]
                
                data = {
                    'symbol': symbol,
                    'price': current_price,
                    'volume': volume,
                    'timestamp': current_time,
                    'bid': info.get('bid', current_price),
                    'ask': info.get('ask', current_price),
                    'market_cap': info.get('marketCap', 0)
                }
                
                self.data_cache[symbol] = data
                self.last_update[symbol] = current_time
                return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_technical_indicators(self, symbol: str, window: int = 20) -> Dict:
        """Calculate technical indicators"""
        hist_data = self.get_historical_data(symbol, period="3mo")
        
        # Simple Moving Average
        sma = hist_data['Close'].rolling(window=window).mean()
        
        # Exponential Moving Average
        ema = hist_data['Close'].ewm(span=window).mean()
        
        # Bollinger Bands
        bb_std = hist_data['Close'].rolling(window=window).std()
        bb_upper = sma + (bb_std * 2)
        bb_lower = sma - (bb_std * 2)
        
        # RSI
        delta = hist_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'sma': sma.iloc[-1] if not sma.empty else None,
            'ema': ema.iloc[-1] if not ema.empty else None,
            'bb_upper': bb_upper.iloc[-1] if not bb_upper.empty else None,
            'bb_lower': bb_lower.iloc[-1] if not bb_lower.empty else None,
            'rsi': rsi.iloc[-1] if not rsi.empty else None,
            'current_price': hist_data['Close'].iloc[-1]
        }
