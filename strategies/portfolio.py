# src/portfolio.py
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

class Portfolio:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'shares': int, 'avg_price': float}}
        self.trade_history = []
        self.performance_history = []
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        stock_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                stock_value += position['shares'] * current_prices[symbol]
        
        return self.cash + stock_value
    
    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get value of specific position"""
        if symbol in self.positions:
            return self.positions[symbol]['shares'] * current_price
        return 0
    
    def can_buy(self, symbol: str, shares: int, price: float) -> bool:
        """Check if we have enough cash to buy"""
        total_cost = shares * price
        return self.cash >= total_cost
    
    def can_sell(self, symbol: str, shares: int) -> bool:
        """Check if we have enough shares to sell"""
        if symbol not in self.positions:
            return False
        return self.positions[symbol]['shares'] >= shares
    
    def execute_buy(self, symbol: str, shares: int, price: float, timestamp: datetime = None) -> bool:
        """Execute buy order"""
        if not self.can_buy(symbol, shares, price):
            return False
        
        total_cost = shares * price
        self.cash -= total_cost
        
        if symbol in self.positions:
            # Update average price
            existing_shares = self.positions[symbol]['shares']
            existing_value = existing_shares * self.positions[symbol]['avg_price']
            new_avg_price = (existing_value + total_cost) / (existing_shares + shares)
            
            self.positions[symbol]['shares'] += shares
            self.positions[symbol]['avg_price'] = new_avg_price
        else:
            self.positions[symbol] = {'shares': shares, 'avg_price': price}
        
        # Record trade
        trade = {
            'timestamp': timestamp or datetime.now(),
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'total': total_cost
        }
        self.trade_history.append(trade)
        return True
    
    def execute_sell(self, symbol: str, shares: int, price: float, timestamp: datetime = None) -> bool:
        """Execute sell order"""
        if not self.can_sell(symbol, shares):
            return False
        
        total_value = shares * price
        self.cash += total_value
        
        self.positions[symbol]['shares'] -= shares
        
        # Remove position if no shares left
        if self.positions[symbol]['shares'] == 0:
            del self.positions[symbol]
        
        # Record trade
        trade = {
            'timestamp': timestamp or datetime.now(),
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'total': total_value
        }
        self.trade_history.append(trade)
        return True
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate performance metrics"""
        current_value = self.get_portfolio_value(current_prices)
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns for Sharpe ratio
        df = pd.DataFrame(self.performance_history)
        if len(df) > 1:
            df['daily_return'] = df['portfolio_value'].pct_change()
            volatility = df['daily_return'].std() * (252 ** 0.5)  # Annualized
            sharpe_ratio = (total_return * 252) / volatility if volatility > 0 else 0
        else:
            sharpe_ratio = 0
            volatility = 0
        
        return {
            'total_value': current_value,
            'cash': self.cash,
            'stock_value': current_value - self.cash,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'num_trades': len(self.trade_history),
            'positions': len(self.positions)
        }
    
    def save_performance_snapshot(self, current_prices: Dict[str, float]):
        """Save current portfolio state for performance tracking"""
        snapshot = {
            'timestamp': datetime.now(),
            'portfolio_value': self.get_portfolio_value(current_prices),
            'cash': self.cash,
            'positions': dict(self.positions)
        }
        self.performance_history.append(snapshot)