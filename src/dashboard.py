import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os

class TradingDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="Trading Bot Dashboard",
            page_icon="📈",
            layout="wide"
        )
    
    def run(self):
        """Main dashboard function"""
        st.title("🤖 Trading Bot Dashboard")
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Trades", "Performance", "Strategy"])
        
        with tab1:
            self.render_portfolio_tab()
        
        with tab2:
            self.render_trades_tab()
        
        with tab3:
            self.render_performance_tab()
        
        with tab4:
            self.render_strategy_tab()
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("Controls")
        
        # Bot status
        if os.path.exists('data/bot_status.txt'):
            with open('data/bot_status.txt', 'r') as f:
                status = f.read().strip()
            st.sidebar.success(f"Bot Status: {status}")
        else:
            st.sidebar.error("Bot Status: Offline")
        
        # Refresh button
        if st.sidebar.button("Refresh Data"):
            st.experimental_rerun()
        
        # Time range selector
        st.sidebar.subheader("Time Range")
        self.time_range = st.sidebar.selectbox(
            "Select Range",
            ["1D", "1W", "1M", "3M", "6M", "1Y", "All"]
        )
    
    def render_portfolio_tab(self):
        """Render portfolio overview"""
        st.header("Portfolio Overview")
        
        # Load portfolio data
        portfolio_data = self.load_portfolio_data()
        if portfolio_data is None:
            st.warning("No portfolio data available")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Value",
                f"${portfolio_data['total_value']:,.2f}",
                f"{portfolio_data['daily_change']:+.2f}%"
            )
        
        with col2:
            st.metric(
                "Total Return",
                f"{portfolio_data['total_return_pct']:+.2f}%",
                f"{portfolio_data['return_change']:+.2f}%"
            )
        
        with col3:
            st.metric(
                "Cash Balance",
                f"${portfolio_data['cash']:,.2f}",
                f"{portfolio_data['cash_change']:+.2f}%"
            )
        
        with col4:
            st.metric(
                "Active Positions",
                portfolio_data['num_positions'],
                f"{portfolio_data['position_change']:+d}"
            )
        
        # Portfolio composition chart
        if portfolio_data['positions']:
            st.subheader("Portfolio Composition")
            
            # Create pie chart
            labels = list(portfolio_data['positions'].keys()) + ['Cash']
            values = list(portfolio_data['positions'].values()) + [portfolio_data['cash']]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig.update_layout(title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
        
        # Positions table
        st.subheader("Current Positions")
        if portfolio_data['positions_detail']:
            df = pd.DataFrame(portfolio_data['positions_detail'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No active positions")
    
    def render_trades_tab(self):
        """Render trades history"""
        st.header("Trade History")
        
        # Load trades data
        trades_df = self.load_trades_data()
        if trades_df is None or trades_df.empty:
            st.warning("No trades data available")
            return
        
        # Filter by time range
        filtered_trades = self.filter_by_time_range(trades_df)
        
        # Trade metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = len(filtered_trades)
        winning_trades = len(filtered_trades[filtered_trades['pnl'] > 0]) if 'pnl' in filtered_trades.columns else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = filtered_trades['pnl'].sum() if 'pnl' in filtered_trades.columns else 0
        
        with col1:
            st.metric("Total Trades", total_trades)
        
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            st.metric("Total P&L", f"${total_pnl:,.2f}")
        
        with col4:
            avg_trade = total_pnl / total_trades if total_trades > 0 else 0
            st.metric("Avg Trade P&L", f"${avg_trade:,.2f}")
        
        # Trades table
        st.subheader("Recent Trades")
        st.dataframe(filtered_trades.tail(20), use_container_width=True)
        
        # Trade distribution chart
        if 'pnl' in filtered_trades.columns:
            st.subheader("P&L Distribution")
            fig = px.histogram(filtered_trades, x='pnl', nbins=20, title="Trade P&L Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_tab(self):
        """Render performance analytics"""
        st.header("Performance Analytics")
        
        # Load performance data
        performance_data = self.load_performance_data()
        if performance_data is None:
            st.warning("No performance data available")
            return
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sharpe Ratio", f"{performance_data['sharpe_ratio']:.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{performance_data['max_drawdown']:.2f}%")
        
        with col3:
            st.metric("Volatility", f"{performance_data['volatility']:.2f}%")
        
        # Performance charts
        if 'equity_curve' in performance_data:
            st.subheader("Equity Curve")
            df_equity = pd.DataFrame(performance_data['equity_curve'])
            fig = px.line(df_equity, x='timestamp', y='portfolio_value', title="Portfolio Value Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly returns heatmap
        if 'monthly_returns' in performance_data:
            st.subheader("Monthly Returns Heatmap")
            monthly_df = pd.DataFrame(performance_data['monthly_returns'])
            fig = px.imshow(monthly_df, title="Monthly Returns (%)", color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_strategy_tab(self):
        """Render strategy information"""
        st.header("Strategy Configuration")
        
        # Load strategy config
        strategy_config = self.load_strategy_config()
        if strategy_config:
            st.json(strategy_config)
        
        # Strategy performance by symbol
        st.subheader("Performance by Symbol")
        symbol_performance = self.load_symbol_performance()
        if symbol_performance:
            df = pd.DataFrame(symbol_performance)
            fig = px.bar(df, x='symbol', y='return_pct', title="Return by Symbol")
            st.plotly_chart(fig, use_container_width=True)
    
    def load_portfolio_data(self):
        """Load portfolio data from files"""
        try:
            # Mock data for demonstration
            return {
                'total_value': 105000,
                'daily_change': 2.5,
                'total_return_pct': 5.0,
                'return_change': 0.5,
                'cash': 25000,
                'cash_change': -1.2,
                'num_positions': 4,
                'position_change': 1,
                'positions': {'AAPL': 20000, 'MSFT': 25000, 'GOOGL': 20000, 'TSLA': 15000},
                'positions_detail': [
                    {'Symbol': 'AAPL', 'Shares': 100, 'Avg Price': 150.00, 'Current Price': 155.00, 'P&L': 500.00},
                    {'Symbol': 'MSFT', 'Shares': 80, 'Avg Price': 300.00, 'Current Price': 312.50, 'P&L': 1000.00},
                ]
            }
        except:
            return None
    
    def load_trades_data(self):
        """Load trades data from CSV"""
        try:
            if os.path.exists('data/trades.csv'):
                df = pd.read_csv('data/trades.csv')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Calculate P&L if not present
                if 'pnl' not in df.columns:
                    df['pnl'] = df.apply(lambda row: 
                        row['total'] if row['action'] == 'SELL' else -row['total'], axis=1)
                return df
            return None
        except:
            return None
    
    def load_performance_data(self):
        """Load performance data"""
        try:
            # Mock data for demonstration
            return {
                'sharpe_ratio': 1.2,
                'max_drawdown': -8.5,
                'volatility': 15.2,
                'equity_curve': [
                    {'timestamp': '2024-01-01', 'portfolio_value': 100000},
                    {'timestamp': '2024-02-01', 'portfolio_value': 102000},
                    {'timestamp': '2024-03-01', 'portfolio_value': 105000},
                ]
            }
        except:
            return None
    
    def load_strategy_config(self):
        """Load strategy configuration"""
        try:
            with open('config/config.yaml', 'r') as f:
                import yaml
                return yaml.safe_load(f)
        except:
            return None
    
    def load_symbol_performance(self):
        """Load performance by symbol"""
        try:
            return [
                {'symbol': 'AAPL', 'return_pct': 3.5},
                {'symbol': 'MSFT', 'return_pct': 4.2},
                {'symbol': 'GOOGL', 'return_pct': 2.8},
                {'symbol': 'TSLA', 'return_pct': 6.1},
            ]
        except:
            return None
    
    def filter_by_time_range(self, df):
        """Filter dataframe by selected time range"""
        if df is None or df.empty:
            return df
        
        now = datetime.now()
        if self.time_range == "1D":
            start_date = now - timedelta(days=1)
        elif self.time_range == "1W":
            start_date = now - timedelta(weeks=1)
        elif self.time_range == "1M":
            start_date = now - timedelta(days=30)
        elif self.time_range == "3M":
            start_date = now - timedelta(days=90)
        elif self.time_range == "6M":
            start_date = now - timedelta(days=180)
        elif self.time_range == "1Y":
            start_date = now - timedelta(days=365)
        else:  # All
            return df
        
        return df[df['timestamp'] >= start_date]

if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
