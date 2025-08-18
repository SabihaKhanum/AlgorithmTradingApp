import os
import time
import logging
import yaml
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import csv

from .portfolio import Portfolio
from .risk_manager import RiskManager
from .data_fetcher import DataFetcher
from .strategies import create_strategy

class TradingBot:
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize components
        self.symbols = self.config['trading']['symbols']
        self.portfolio = Portfolio(self.config['trading']['initial_capital'])
        self.risk_manager = RiskManager(**self.config['risk_management'])
        self.data_fetcher = DataFetcher(self.symbols)
        self.strategy = create_strategy(
            self.config['trading']['strategy'], 
            self.config['strategy']
        )
        
        # Bot state
        self.is_running = False
        self.last_trade_time = {}
        self.min_trade_interval = timedelta(
            seconds=self.config['data']['min_trade_interval']
        )
        
        # Setup logging
        self._setup_logging()
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        self.trade_log_file = 'data/trades.csv'
        self._init_trade_log()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config['logging']
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_trade_log(self):
        """Initialize trade log CSV file"""
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'symbol', 'action', 'shares', 'price', 
                    'total', 'reason', 'portfolio_value'
                ])
    
    def _log_trade(self, trade: Dict, reason: str, portfolio_value: float):
        """Log trade to CSV file"""
        with open(self.trade_log_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                trade['timestamp'],
                trade['symbol'],
                trade['action'],
                trade['shares'],
                trade['price'],
                trade['total'],
                reason,
                portfolio_value
            ])
    
    def start(self, mode: str = "backtest", period: str = "3mo"):
        """Start the trading bot"""
        self.is_running = True
        self.logger.info(f"Trading bot started in {mode} mode")
        
        if mode == "backtest":
            self.run_backtest(period)
        elif mode == "paper":
            self.run_paper_trading()
        elif mode == "live":
            self.logger.warning("Live trading mode - USE AT YOUR OWN RISK!")
            self.run_live_trading()
        else:
            raise ValueError("Mode must be 'backtest', 'paper', or 'live'")
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        self.logger.info("Trading bot stopped")
    
    def run_backtest(self, period: str = "3mo"):
        """Run backtesting on historical data"""
        self.logger.info(f"Starting backtest for period: {period}")
        
        # Get historical data for all symbols
        all_data = {}
        for symbol in self.symbols:
            data = self.data_fetcher.get_historical_data(symbol, period)
            if not data.empty:
                all_data[symbol] = data
        
        if not all_data:
            self.logger.error("No historical data available for backtesting")
            return
        
        # Get common date range
        start_date = max([data.index[0] for data in all_data.values()])
        end_date = min([data.index[-1] for data in all_data.values()])
        
        self.logger.info(f"Backtest period: {start_date} to {end_date}")
        
        # Run backtest day by day
        current_date = start_date
        while current_date <= end_date and self.is_running:
            for symbol in self.symbols:
                if symbol in all_data:
                    # Get data up to current date
                    historical_subset = all_data[symbol].loc[:current_date]
                    if len(historical_subset) >= self.config['strategy']['long_window']:
                        
                        current_price = historical_subset['Close'].iloc[-1]
                        technical_indicators = self._calculate_backtest_indicators(
                            historical_subset)
                        
                        # Generate and execute signals
                        signal = self.strategy.generate_signals(
                            symbol, historical_subset, technical_indicators)
                        self._execute_signal(symbol, signal, current_price, current_date)
            
            current_date += timedelta(days=1)
        
        # Generate final report
        self._generate_backtest_report(all_data)
    
    def run_paper_trading(self):
        """Run paper trading with real-time data"""
        self.logger.info("Starting paper trading...")
        
        while self.is_running:
            try:
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                
                # Update portfolio performance
                current_prices = self._get_current_prices()
                if current_prices:
                    self.portfolio.save_performance_snapshot(current_prices)
                    self.risk_manager.update_daily_start(
                        self.portfolio.get_portfolio_value(current_prices))
                
                # Log status every 10 minutes
                if datetime.now().minute % 10 == 0:
                    self._log_status()
                
                # Wait before next iteration
                time.sleep(self.config['data']['update_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("Received stop signal")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def run_live_trading(self):
        """Run live trading - REAL MONEY AT RISK"""
        self.logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK!")
        self.logger.warning("Make sure you have tested thoroughly in backtest and paper modes")
        
        # Add additional safety checks for live trading
        response = input("Are you sure you want to proceed with live trading? (yes/no): ")
        if response.lower() != 'yes':
            self.logger.info("Live trading cancelled by user")
            return
        
        # Run same logic as paper trading
        self.run_paper_trading()
    
    def _process_symbol(self, symbol: str):
        """Process trading logic for a single symbol"""
        
        # Check if enough time has passed since last trade
        if (symbol in self.last_trade_time and 
            datetime.now() - self.last_trade_time[symbol] < self.min_trade_interval):
            return
        
        # Get market data
        real_time_data = self.data_fetcher.get_real_time_data(symbol)
        if not real_time_data:
            return
        
        historical_data = self.data_fetcher.get_historical_data(symbol)
        technical_indicators = self.data_fetcher.get_technical_indicators(symbol)
        
        # Generate trading signals
        signal = self.strategy.generate_signals(symbol, historical_data, technical_indicators)
        
        # Execute trades based on signals
        self._execute_signal(symbol, signal, real_time_data['price'])
    
    def _execute_signal(self, symbol: str, signal: Dict, current_price: float, 
                       timestamp: datetime = None):
        """Execute trading signal"""
        
        action = signal['action']
        confidence = signal['confidence']
        
        if action == 'HOLD':
            return
        
        current_prices = {symbol: current_price}
        portfolio_value = self.portfolio.get_portfolio_value(current_prices)
        
        if action == 'BUY':
            # Calculate position size
            risk_amount = portfolio_value * self.config['risk_management']['max_portfolio_risk']
            stop_loss_pct = self.config['risk_management']['stop_loss_pct']
            shares = int(risk_amount / (current_price * stop_loss_pct))
            
            # Risk management check
            allowed, reason, adjusted_shares = self.risk_manager.should_allow_trade(
                symbol, 'BUY', shares, current_price, portfolio_value, 
                self.portfolio.positions
            )
            
            if allowed and adjusted_shares > 0:
                success = self.portfolio.execute_buy(
                    symbol, adjusted_shares, current_price, timestamp)
                if success:
                    # Set risk management levels
                    self.risk_manager.set_stop_loss(symbol, current_price)
                    self.risk_manager.set_take_profit(symbol, current_price)
                    
                    self.last_trade_time[symbol] = datetime.now()
                    
                    # Log trade
                    trade = self.portfolio.trade_history[-1]
                    self._log_trade(trade, signal['reason'], portfolio_value)
                    
                    self.logger.info(f"BUY {adjusted_shares} shares of {symbol} at ${current_price:.2f}")
                    self.logger.info(f"Reason: {signal['reason']}")
                    
                    if reason != "Trade approved":
                        self.logger.info(f"Risk adjustment: {reason}")
        
        elif action == 'SELL':
            if symbol in self.portfolio.positions:
                shares_to_sell = self.portfolio.positions[symbol]['shares']
                
                allowed, reason, adjusted_shares = self.risk_manager.should_allow_trade(
                    symbol, 'SELL', shares_to_sell, current_price, portfolio_value,
                    self.portfolio.positions
                )
                
                if allowed and adjusted_shares > 0:
                    success = self.portfolio.execute_sell(
                        symbol, adjusted_shares, current_price, timestamp)
                    if success:
                        # Remove risk management levels
                        if symbol in self.risk_manager.stop_losses:
                            del self.risk_manager.stop_losses[symbol]
                        if symbol in self.risk_manager.take_profits:
                            del self.risk_manager.take_profits[symbol]
                        
                        self.last_trade_time[symbol] = datetime.now()
                        
                        # Log trade
                        trade = self.portfolio.trade_history[-1]
                        self._log_trade(trade, signal['reason'], portfolio_value)
                        
                        self.logger.info(f"SELL {adjusted_shares} shares of {symbol} at ${current_price:.2f}")
                        self.logger.info(f"Reason: {signal['reason']}")
        
        # Check stop loss and take profit
        self._check_risk_exits(symbol, current_price, timestamp)
    
    def _check_risk_exits(self, symbol: str, current_price: float, 
                         timestamp: datetime = None):
        """Check and execute stop loss and take profit orders"""
        
        if symbol not in self.portfolio.positions:
            return
        
        portfolio_value = self.portfolio.get_portfolio_value({symbol: current_price})
        
        # Check stop loss
        if self.risk_manager.check_stop_loss(symbol, current_price):
            shares = self.portfolio.positions[symbol]['shares']
            success = self.portfolio.execute_sell(symbol, shares, current_price, timestamp)
            if success:
                trade = self.portfolio.trade_history[-1]
                self._log_trade(trade, "Stop Loss", portfolio_value)
                
                self.logger.info(f"STOP LOSS: Sold {shares} shares of {symbol} at ${current_price:.2f}")
                del self.risk_manager.stop_losses[symbol]
                if symbol in self.risk_manager.take_profits:
                    del self.risk_manager.take_profits[symbol]
        
        # Check take profit
        elif self.risk_manager.check_take_profit(symbol, current_price):
            shares = self.portfolio.positions[symbol]['shares']
            success = self.portfolio.execute_sell(symbol, shares, current_price, timestamp)
            if success:
                trade = self.portfolio.trade_history[-1]
                self._log_trade(trade, "Take Profit", portfolio_value)
                
                self.logger.info(f"TAKE PROFIT: Sold {shares} shares of {symbol} at ${current_price:.2f}")
                del self.risk_manager.take_profits[symbol]
                if symbol in self.risk_manager.stop_losses:
                    del self.risk_manager.stop_losses[symbol]
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        prices = {}
        for symbol in self.symbols:
            data = self.data_fetcher.get_real_time_data(symbol)
            if data:
                prices[symbol] = data['price']
        return prices
    
    def _log_status(self):
        """Log current bot status"""
        current_prices = self._get_current_prices()
        if not current_prices:
            return
        
        performance = self.portfolio.get_performance_metrics(current_prices)
        risk_metrics = self.risk_manager.get_risk_metrics(performance['total_value'])
        
        self.logger.info(f"Portfolio Value: ${performance['total_value']:,.2f}")
        self.logger.info(f"Total Return: {performance['total_return_pct']:.2f}%")
        self.logger.info(f"Daily P&L: {risk_metrics['daily_pnl_pct']:.2f}%")
        self.logger.info(f"Active Positions: {performance['positions']}")
    
    def _calculate_backtest_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for backtesting"""
        if len(data) < 20:
            return {'sma': None, 'ema': None, 'rsi': None}
        
        # Simple Moving Average
        sma = data['Close'].rolling(window=20).mean().iloc[-1]
        
        # Exponential Moving Average
        ema = data['Close'].ewm(span=20).mean().iloc[-1]
        
        # Bollinger Bands
        bb_std = data['Close'].rolling(window=20).std().iloc[-1]
        bb_upper = sma + (bb_std * 2)
        bb_lower = sma - (bb_std * 2)
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        return {
            'sma': sma,
            'ema': ema,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'rsi': rsi,
            'current_price': data['Close'].iloc[-1]
        }
    
    def _generate_backtest_report(self, all_data: Dict):
        """Generate comprehensive backtest report"""
        final_prices = {symbol: all_data[symbol]['Close'].iloc[-1] 
                       for symbol in all_data.keys()}
        final_performance = self.portfolio.get_performance_metrics(final_prices)
        
        self.logger.info("=" * 50)
        self.logger.info("BACKTEST RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"Strategy: {self.strategy.name}")
        self.logger.info(f"Initial Capital: ${self.portfolio.initial_capital:,.2f}")
        self.logger.info(f"Final Portfolio Value: ${final_performance['total_value']:,.2f}")
        self.logger.info(f"Total Return: {final_performance['total_return_pct']:.2f}%")
        self.logger.info(f"Sharpe Ratio: {final_performance['sharpe_ratio']:.2f}")
        self.logger.info(f"Volatility: {final_performance['volatility']:.2f}")
        self.logger.info(f"Total Trades: {final_performance['num_trades']}")
        self.logger.info(f"Final Cash: ${final_performance['cash']:,.2f}")
        self.logger.info(f"Stock Value: ${final_performance['stock_value']:,.2f}")
        self.logger.info("=" * 50)
        
        # Save detailed results
        results = {
            'strategy': self.strategy.name,
            'initial_capital': self.portfolio.initial_capital,
            'final_value': final_performance['total_value'],
            'total_return_pct': final_performance['total_return_pct'],
            'sharpe_ratio': final_performance['sharpe_ratio'],
            'volatility': final_performance['volatility'],
            'total_trades': final_performance['num_trades'],
            'trade_history': self.portfolio.trade_history,
            'performance_history': self.portfolio.performance_history
        }
        
        # Save to file
        import json
        with open(f'data/backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
