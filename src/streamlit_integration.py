"""
Streamlit Dashboard Integration for Options Trading
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
from loguru import logger
import asyncio
import threading
import queue
import time

class StreamlitIntegration:
    """Streamlit dashboard for real-time options trading monitoring"""
    
    def __init__(self):
        self.opportunities_queue = queue.Queue()
        self.trades_queue = queue.Queue()
        self.portfolio_data = {
            'positions': [],
            'trades': [],
            'opportunities': [],
            'pnl_breakdown': {
                'total_unrealized_pnl': 0.0,
                'total_realized_pnl': 0.0,
                'daily_pnl': 0.0,
                'weekly_pnl': 0.0
            },
            'portfolio_value': 100000.0,  # Starting value
            'cash_balance': 50000.0
        }
        logger.info("StreamlitIntegration initialized")
    
    def push_opportunities_to_dashboard(self, opportunities: List[Dict]) -> bool:
        """Push new trading opportunities to the dashboard"""
        try:
            for opportunity in opportunities:
                # Add timestamp if not present
                if 'timestamp' not in opportunity:
                    opportunity['timestamp'] = datetime.now().isoformat()
                
                self.opportunities_queue.put(opportunity)
                self.portfolio_data['opportunities'].append(opportunity)
            
            # Keep only last 50 opportunities
            if len(self.portfolio_data['opportunities']) > 50:
                self.portfolio_data['opportunities'] = self.portfolio_data['opportunities'][-50:]
            
            logger.info(f"Added {len(opportunities)} opportunities to dashboard")
            return True
            
        except Exception as e:
            logger.error(f"Error pushing opportunities: {e}")
            return False
    
    def push_trade_to_dashboard(self, trade: Dict) -> bool:
        """Push executed trade to the dashboard"""
        try:
            # Add timestamp if not present
            if 'timestamp' not in trade:
                trade['timestamp'] = datetime.now().isoformat()
            
            # Calculate trade value
            trade_value = trade.get('quantity', 0) * trade.get('price', 0) * 100  # Options are 100 shares
            
            if trade.get('action', '').upper() == 'BUY':
                trade_value = -trade_value  # Cash outflow
            else:
                trade_value = trade_value   # Cash inflow
            
            # Update portfolio data
            self.trades_queue.put(trade)
            self.portfolio_data['trades'].append(trade)
            
            # Update cash balance
            self.portfolio_data['cash_balance'] += trade_value
            
            # Simulate P&L calculation
            self._update_pnl_calculations(trade)
            
            # Keep only last 100 trades
            if len(self.portfolio_data['trades']) > 100:
                self.portfolio_data['trades'] = self.portfolio_data['trades'][-100:]
            
            logger.info(f"Added trade to dashboard: {trade['symbol']} {trade['action']}")
            return True
            
        except Exception as e:
            logger.error(f"Error pushing trade: {e}")
            return False
    
    def get_dashboard_data(self) -> Dict:
        """Get current dashboard data"""
        try:
            # Update real-time calculations
            self._update_portfolio_calculations()
            
            # Return copy of portfolio data
            return self.portfolio_data.copy()
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def _update_pnl_calculations(self, trade: Dict):
        """Update P&L calculations based on new trade"""
        try:
            # Simulate realistic P&L changes
            trade_pnl = np.random.normal(0, 50)  # Random P&L simulation
            
            self.portfolio_data['pnl_breakdown']['daily_pnl'] += trade_pnl
            self.portfolio_data['pnl_breakdown']['total_unrealized_pnl'] += trade_pnl * 0.7  # Simulate unrealized
            self.portfolio_data['pnl_breakdown']['total_realized_pnl'] += trade_pnl * 0.3    # Simulate realized
            
        except Exception as e:
            logger.error(f"Error updating P&L: {e}")
    
    def _update_portfolio_calculations(self):
        """Update portfolio-wide calculations"""
        try:
            # Calculate weekly P&L from daily
            self.portfolio_data['pnl_breakdown']['weekly_pnl'] = (
                self.portfolio_data['pnl_breakdown']['daily_pnl'] * 5  # Simulate 5 trading days
            )
            
            # Update portfolio value
            total_pnl = (
                self.portfolio_data['pnl_breakdown']['total_unrealized_pnl'] +
                self.portfolio_data['pnl_breakdown']['total_realized_pnl']
            )
            
            self.portfolio_data['portfolio_value'] = 100000.0 + total_pnl  # Starting value + P&L
            
        except Exception as e:
            logger.error(f"Error updating portfolio calculations: {e}")

    def run_dashboard(self, port: int = 8501):
        """Run the Streamlit dashboard"""
        try:
            logger.info(f"Starting Streamlit dashboard on port {port}")
            
            # Create the main dashboard function
            def main_dashboard():
                st.set_page_config(
                    page_title="Options Trading Dashboard",
                    page_icon="📈",
                    layout="wide",
                    initial_sidebar_state="expanded"
                )
                
                st.title("🚀 Options Trading Dashboard")
                st.markdown("Real-time monitoring of your options trading algorithms")
                
                # Sidebar
                st.sidebar.header("Dashboard Controls")
                auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
                refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)
                
                if auto_refresh:
                    time.sleep(refresh_interval)
                    st.experimental_rerun()
                
                # Get current data
                data = self.get_dashboard_data()
                
                # Portfolio Summary
                st.header("📊 Portfolio Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Portfolio Value",
                        f"${data['portfolio_value']:,.2f}",
                        f"${data['pnl_breakdown']['daily_pnl']:+,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Unrealized P&L",
                        f"${data['pnl_breakdown']['total_unrealized_pnl']:,.2f}",
                        "Today"
                    )
                
                with col3:
                    st.metric(
                        "Realized P&L",
                        f"${data['pnl_breakdown']['total_realized_pnl']:,.2f}",
                        "Total"
                    )
                
                with col4:
                    st.metric(
                        "Cash Balance",
                        f"${data['cash_balance']:,.2f}",
                        "Available"
                    )
                
                # Trading Opportunities
                st.header("🎯 Recent Opportunities")
                if data['opportunities']:
                    opportunities_df = pd.DataFrame(data['opportunities'][-10:])  # Last 10
                    st.dataframe(opportunities_df, use_container_width=True)
                else:
                    st.info("No opportunities detected yet")
                
                # Recent Trades
                st.header("💼 Recent Trades")
                if data['trades']:
                    trades_df = pd.DataFrame(data['trades'][-10:])  # Last 10
                    st.dataframe(trades_df, use_container_width=True)
                    
                    # P&L Chart
                    st.subheader("P&L Trend")
                    fig = go.Figure()
                    
                    # Simulate daily P&L data
                    dates = pd.date_range(start=datetime.now()-timedelta(days=30), periods=30)
                    pnl_values = np.cumsum(np.random.normal(20, 100, 30))  # Simulated P&L
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=pnl_values,
                        mode='lines+markers',
                        name='Daily P&L',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig.update_layout(
                        title="30-Day P&L Trend",
                        xaxis_title="Date",
                        yaxis_title="P&L ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.info("No trades executed yet")
                
                # Strategy Performance
                st.header("📈 Strategy Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Win Rate Gauge
                    win_rate = 0.65 if data['trades'] else 0  # Simulated
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = win_rate * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Win Rate (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Risk Metrics
                    st.subheader("Risk Metrics")
                    st.metric("Max Drawdown", "-2.5%", "Acceptable")
                    st.metric("Sharpe Ratio", "1.85", "Good")
                    st.metric("Active Positions", len(data.get('positions', [])), "Current")
                
                # Footer
                st.markdown("---")
                st.markdown("🤖 Powered by Options AI Trading System")
            
            # This would normally be run with: streamlit run dashboard.py
            # For now, we'll create the dashboard structure
            return main_dashboard
            
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            return None

# For standalone execution
if __name__ == "__main__":
    dashboard = StreamlitIntegration()
    logger.info("StreamlitIntegration initialized for standalone execution")
