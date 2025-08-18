
"""
Trading Bot Entry Point
"""
import argparse
import sys
import os
from src.bot import TradingBot

def main():
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], 
                       default='backtest', help='Trading mode')
    parser.add_argument('--period', default='3mo', 
                       help='Backtest period (1mo, 3mo, 6mo, 1y)')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch dashboard')
    
    args = parser.parse_args()
    
    if args.dashboard:
        # Launch dashboard
        import subprocess
        subprocess.run(['streamlit', 'run', 'src/dashboard.py'])
        return
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)
    
    # Create bot and start
    try:
        bot = TradingBot(args.config)
        
        print(f"Starting trading bot in {args.mode} mode...")
        if args.mode == 'backtest':
            print(f"Backtest period: {args.period}")
        
        bot.start(mode=args.mode, period=args.period)
        
    except KeyboardInterrupt:
        print("\nShutting down trading bot...")
        bot.stop()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
