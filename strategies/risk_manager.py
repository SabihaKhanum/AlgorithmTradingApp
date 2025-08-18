# src/risk_manager.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

class RiskManager:
    def __init__(self, 
                 max_position_size: float = 0.1,      # 10% of portfolio per position
                 max_portfolio_risk: float = 0.02,     # 2% portfolio risk per trade
                 stop_loss_pct: float = 0.05,          # 5% stop loss
                 take_profit_pct: float = 0.10,        # 10% take profit
                 max_daily_loss: float = 0.05,         # 5% max daily loss
                 max_drawdown: float = 0.15):          # 15% max drawdown
        
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        
        self.daily_start_value = None
        self.peak_portfolio_value = 0
        self.stop_losses = {}  # {symbol: price}
        self.take_profits = {}  # {symbol: price}
    
    def check_position_size_limit(self, symbol: str, shares: int, price: float, 
                                portfolio_value: float) -> Tuple[bool, int]:
        """Check and adjust position size based on limits"""
        position_value = shares * price
        max_position_value = portfolio_value * self.max_position_size
        
        if position_value > max_position_value:
            adjusted_shares = int(max_position_value / price)
            return False, adjusted_shares
        
        return True, shares
    
    def check_daily_loss_limit(self, current_portfolio_value: float) -> bool:
        """Check if daily loss limit is exceeded"""
        if self.daily_start_value is None:
            self.daily_start_value = current_portfolio_value
            return True
        
        daily_loss = (self.daily_start_value - current_portfolio_value) / self.daily_start_value
        return daily_loss < self.max_daily_loss
    
    def check_drawdown_limit(self, current_portfolio_value: float) -> bool:
        """Check if drawdown limit is exceeded"""
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        
        if self.peak_portfolio_value == 0:
            return True
        
        drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        return drawdown < self.max_drawdown
    
    def set_stop_loss(self, symbol: str, entry_price: float, is_long: bool = True):
        """Set stop loss for a position"""
        if is_long:
            stop_price = entry_price * (1 - self.stop_loss_pct)
        else:
            stop_price = entry_price * (1 + self.stop_loss_pct)
        
        self.stop_losses[symbol] = stop_price
    
    def set_take_profit(self, symbol: str, entry_price: float, is_long: bool = True):
        """Set take profit for a position"""
        if is_long:
            take_profit_price = entry_price * (1 + self.take_profit_pct)
        else:
            take_profit_price = entry_price * (1 - self.take_profit_pct)
        
        self.take_profits[symbol] = take_profit_price
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if symbol not in self.stop_losses:
            return False
        
        stop_price = self.stop_losses[symbol]
        return current_price <= stop_price
    
    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        if symbol not in self.take_profits:
            return False
        
        take_profit_price = self.take_profits[symbol]
        return current_price >= take_profit_price
    
    def should_allow_trade(self, symbol: str, action: str, shares: int, price: float,
                          portfolio_value: float, current_positions: Dict) -> Tuple[bool, str, int]:
        """
        Main risk check function
        Returns: (allowed, reason, adjusted_shares)
        """
        
        # Check daily loss limit
        if not self.check_daily_loss_limit(portfolio_value):
            return False, "Daily loss limit exceeded", 0
        
        # Check drawdown limit
        if not self.check_drawdown_limit(portfolio_value):
            return False, "Maximum drawdown limit exceeded", 0
        
        if action == 'BUY':
            # Check position size limit
            allowed, adjusted_shares = self.check_position_size_limit(
                symbol, shares, price, portfolio_value)
            
            if not allowed:
                return True, f"Position size adjusted from {shares} to {adjusted_shares}", adjusted_shares
            
            return True, "Trade approved", shares
        
        elif action == 'SELL':
            # Check if we actually have the position
            if symbol not in current_positions:
                return False, "No position to sell", 0
            
            available_shares = current_positions[symbol]['shares']
            if shares > available_shares:
                return True, f"Sell quantity adjusted to {available_shares}", available_shares
            
            return True, "Trade approved", shares
        
        return False, "Unknown action", 0
    
    def update_daily_start(self, portfolio_value: float):
        """Update daily start value (call at market open)"""
        self.daily_start_value = portfolio_value
    
    def get_risk_metrics(self, portfolio_value: float) -> Dict:
        """Get current risk metrics"""
        daily_pnl = 0
        if self.daily_start_value:
            daily_pnl = (portfolio_value - self.daily_start_value) / self.daily_start_value
        
        drawdown = 0
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        
        return {
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl * 100,
            'current_drawdown': drawdown,
            'current_drawdown_pct': drawdown * 100,
            'peak_value': self.peak_portfolio_value,
            'daily_start_value': self.daily_start_value,
            'stop_losses_active': len(self.stop_losses),
            'take_profits_active': len(self.take_profits)
        }