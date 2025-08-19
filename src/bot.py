# src/bot.py - Fixed version that creates directories automatically
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
        # Create directories if they don't exist
        self._create_directories()
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found")
            print("Creating default configuration...")
            self._create_default_config(config_path)
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
        
        self.trade_log_file = 'data/trades.csv'
        self._init_trade_log()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['logs', 'data', 'config', 'tests']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created/verified directory: {directory}")
    
    def _create_default_config(self, config_path: str):
        """Create default configuration file"""
        default_config = {
            'trading': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                'initial_capital': 100000,
                'strategy': 'MovingAverageCrossover'
            },
            'strategy': {
                'short_window': 10,
                'long_window': 20,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'volume_threshold': 1.5,
                'min_confidence': 60
            },
            'risk_management': {
                'max_position_size': 0.1,
                'max_portfolio_risk': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.15
            },
            'data': {
                'update_interval': 60,
                'min_trade_interval': 300,
                'cache_duration': 60
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/trading_bot.log',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, indent=2)
        
        print(f"✓ Created default configuration at: {config_path}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config['logging']
        
        # Ensure log directory exists
        log_file = log_config['file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Setup new logging
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True  # Force reconfiguration
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Trading bot logging initialized")
    
    def _init_trade_log(self):
        """Initialize trade log CSV file"""
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'symbol', 'action', 'shares', 'price', 
                    'total', 'reason', 'portfolio_value'
                ])
            print(f"✓ Created trade log file: {self.trade_log_file}")
    
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
        print(f"🚀 Starting trading bot in {mode} mode...")
        
        if mode == "backtest":
            self.run_backtest(period)
        elif mode == "paper":
            self.run_paper_trading()
        elif mode == "live":
            self.logger.warning("Live trading mode - USE AT YOUR OWN RISK!")
            print("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!")
            self.run_live_trading()
        else:
            raise ValueError("Mode must be 'backtest', 'paper', or 'live'")
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        self.logger.info("Trading bot stopped")
        print("🛑 Trading bot stopped")
    
    def run_backtest(self, period: str = "3mo"):
        """Run backtesting on historical data"""
        self.logger.info(f"Starting backtest for period: {period}")
        print(f"📊 Running backtest for {period}...")
        print(f"📈 Trading symbols: {', '.join(self.symbols)}")
        
        # Get historical data for all symbols
        all_data = {}
        print("📥 Fetching historical data...")
        
        for i, symbol in enumerate(self.symbols, 1):
            print(f"   {i}/{len(self.symbols)} - Fetching {symbol}...")
            try:
                data = self.data_fetcher.get_historical_data(symbol, period)
                if not data.empty:
                    all_data[symbol] = data
                    print(f"   ✓ {symbol}: {len(data)} days of data")
                else:
                    print(f"   ❌ {symbol}: No data available")
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {e}")
        
        if not all_data:
            self.logger.error("No historical data available for backtesting")
            print("❌ No historical data available. Check your internet connection.")
            return
        
        # Get common date range
        start_date = max([data.index[0] for data in all_data.values()])
        end_date = min([data.index[-1] for data in all_data.values()])
        total_days = (end_date - start_date).days
        
        self.logger.info(f"Backtest period: {start_date.date()} to {end_date.date()}")
        print(f"📅 Backtest period: {start_date.date()} to {end_date.date()} ({total_days} days)")
        
        # Run backtest day by day
        current_date = start_date
        processed_days = 0
        
        while current_date <= end_date and self.is_running:
            if processed_days % 30 == 0:  # Progress update every 30 days
                progress = (processed_days / total_days) * 100
                print(f"⏳ Progress: {progress:.1f}% ({processed_days}/{total_days} days)")
            
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
            processed_days += 1
        
        print("✅ Backtest completed!")
        
        # Generate final report
        self._generate_backtest_report(all_data)
    
    def run_paper_trading(self):
        """Run paper trading with real-time data"""
        self.logger.info("Starting paper trading...")
        print("📄 Starting paper trading mode...")
        print("💡 This uses real market data but no real money")
        print("🔄 Press Ctrl+C to stop")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                if iteration % 10 == 1:  # Log every 10 iterations
                    print(f"🔄 Running iteration {iteration}...")
                
                for symbol in self.symbols:
                    self._process_symbol(symbol)
                
                # Update portfolio performance
                current_prices = self._get_current_prices()
                if current_prices:
                    self.portfolio.save_performance_snapshot(current_prices)
                    self.risk_manager.update_daily_start(
                        self.portfolio.get_portfolio_value(current_prices))
                
                # Log status every 10 minutes
                if iteration % 10 == 0:
                    self._log_status()
                
                # Wait before next iteration
                time.sleep(self.config['data']['update_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("Received stop signal")
                print("🛑 Stopping paper trading...")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                print(f"❌ Error: {e}")
                time.sleep(60)
    
    def run_live_trading(self):
        """Run live trading - REAL MONEY AT RISK"""
        self.logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK!")
        print("⚠️  WARNING: LIVE TRADING MODE")
        print("💰 REAL MONEY WILL BE AT RISK!")
        print("🧪 Make sure you have tested thoroughly in backtest and paper modes")
        
        # Add additional safety checks for live trading
        response = input("Are you absolutely sure you want to proceed with live trading? (type 'YES' to continue): ")
        if response != 'YES':
            self.logger.info("Live trading cancelled by user")
            print("✅ Live trading cancelled - staying safe!")
            return
        
        # Run same logic as paper trading
        self.run_paper_trading()
    
    def _process_symbol(self, symbol: str):
        """Process trading logic for a single symbol"""
        
        # Check if enough time has passed since last trade
        if (symbol in self.last_trade_time and 
            datetime.now() - self.last_trade_time[symbol] < self.min_trade_interval):
            return
        
        try:
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
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
    
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
                    print(f"📈 BUY {adjusted_shares} {symbol} @ ${current_price:.2f} - {signal['reason']}")
                    
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
                        print(f"📉 SELL {adjusted_shares} {symbol} @ ${current_price:.2f} - {signal['reason']}")
        
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
                print(f"🛑 STOP LOSS: {shares} {symbol} @ ${current_price:.2f}")
                
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
                print(f"🎯 TAKE PROFIT: {shares} {symbol} @ ${current_price:.2f}")
                
                del self.risk_manager.take_profits[symbol]
                if symbol in self.risk_manager.stop_losses:
                    del self.risk_manager.stop_losses[symbol]
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols"""
        prices = {}
        for symbol in self.symbols:
            try:
                data = self.data_fetcher.get_real_time_data(symbol)
                if data:
                    prices[symbol] = data['price']
            except Exception as e:
                self.logger.warning(f"Failed to get price for {symbol}: {e}")
        return prices
    
    def _log_status(self):
        """Log current bot status"""
        current_prices = self._get_current_prices()
        if not current_prices:
            return
        
        performance = self.portfolio.get_performance_metrics(current_prices)
        risk_metrics = self.risk_manager.get_risk_metrics(performance['total_value'])
        
        status_msg = (f"Portfolio: ${performance['total_value']:,.2f} | "
                     f"Return: {performance['total_return_pct']:.2f}% | "
                     f"Daily P&L: {risk_metrics['daily_pnl_pct']:.2f}% | "
                     f"Positions: {performance['positions']}")
        
        self.logger.info(status_msg)
        print(f"💼 {status_msg}")
    
    def _calculate_backtest_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators for backtesting"""
        if len(data) < 20:
            return {'sma': None, 'ema': None, 'rsi': None}
        
        try:
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
        except Exception as e:
            self.logger.warning(f"Error calculating indicators: {e}")
            return {'sma': None, 'ema': None, 'rsi': None}
    
    def _generate_backtest_report(self, all_data: Dict):
        """Generate comprehensive backtest report"""
        final_prices = {symbol: all_data[symbol]['Close'].iloc[-1] 
                       for symbol in all_data.keys()}
        final_performance = self.portfolio.get_performance_metrics(final_prices)
        
        # Console output
        print("\n" + "="*60)
        print("🎯 BACKTEST RESULTS")
        print("="*60)
        print(f"📊 Strategy: {self.strategy.name}")
        print(f"💰 Initial Capital: ${self.portfolio.initial_capital:,.2f}")
        print(f"📈 Final Portfolio Value: ${final_performance['total_value']:,.2f}")
        print(f"📊 Total Return: {final_performance['total_return_pct']:.2f}%")
        print(f"📉 Sharpe Ratio: {final_performance['sharpe_ratio']:.2f}")
        print(f"📊 Volatility: {final_performance['volatility']:.2f}")
        print(f"🔄 Total Trades: {final_performance['num_trades']}")
        print(f"💵 Final Cash: ${final_performance['cash']:,.2f}")
        print(f"📈 Stock Value: ${final_performance['stock_value']:,.2f}")
        
        # Performance analysis
        if final_performance['total_return_pct'] > 0:
            print("✅ PROFITABLE STRATEGY")
        else:
            print("❌ UNPROFITABLE STRATEGY")
            
        if final_performance['sharpe_ratio'] > 1.0:
            print("✅ GOOD RISK-ADJUSTED RETURNS")
        else:
            print("⚠️  POOR RISK-ADJUSTED RETURNS")
        
        print("="*60)
        
        # Log to file
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
            'trade_history': [dict(trade) for trade in self.portfolio.trade_history],  # Convert to dict
            'performance_history': self.portfolio.performance_history
        }
        
        # Save to file
        import json
        results_file = f'data/backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Detailed results saved to: {results_file}")
        except Exception as e:
            self.logger.warning(f"Could not save results file: {e}")