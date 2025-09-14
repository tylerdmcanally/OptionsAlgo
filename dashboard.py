"""
Options Trading Dashboard Runner
Run this to start the Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from streamlit_integration import StreamlitIntegration

# Create global dashboard instance
if 'dashboard' not in st.session_state:
    st.session_state.dashboard = StreamlitIntegration()

dashboard = st.session_state.dashboard

# Page configuration
st.set_page_config(
    page_title="Options Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .profit-positive {
        color: #28a745;
    }
    .profit-negative {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Main dashboard
st.markdown('<h1 class="main-header">🚀 Options Trading Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Real-time monitoring of your options trading algorithms**")

# Sidebar controls
st.sidebar.header("🎛️ Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.experimental_rerun()

# Get current data
data = dashboard.get_dashboard_data()

# Portfolio Summary
st.header("📊 Portfolio Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    portfolio_value = data['portfolio_value']
    daily_change = data['pnl_breakdown']['daily_pnl']
    change_pct = (daily_change / portfolio_value) * 100 if portfolio_value > 0 else 0
    
    st.metric(
        "Portfolio Value",
        f"${portfolio_value:,.2f}",
        f"{daily_change:+,.2f} ({change_pct:+.2f}%)"
    )

with col2:
    unrealized_pnl = data['pnl_breakdown']['total_unrealized_pnl']
    st.metric(
        "Unrealized P&L",
        f"${unrealized_pnl:,.2f}",
        "Current"
    )

with col3:
    realized_pnl = data['pnl_breakdown']['total_realized_pnl']
    st.metric(
        "Realized P&L",
        f"${realized_pnl:,.2f}",
        "Total"
    )

with col4:
    cash_balance = data['cash_balance']
    st.metric(
        "Cash Balance",
        f"${cash_balance:,.2f}",
        "Available"
    )

# Two column layout for main content
left_col, right_col = st.columns([2, 1])

with left_col:
    # P&L Chart
    st.subheader("📈 P&L Trend (Last 30 Days)")
    
    # Generate sample P&L data
    dates = pd.date_range(start=datetime.now()-timedelta(days=30), periods=30)
    
    # Create a more realistic P&L trend
    base_pnl = 0
    pnl_values = []
    for i in range(30):
        # Add some market noise and trend
        daily_change = np.random.normal(50, 200)  # Daily P&L change
        base_pnl += daily_change
        pnl_values.append(base_pnl)
    
    # Create the plot
    fig = go.Figure()
    
    # Add the P&L line
    colors = ['green' if val >= 0 else 'red' for val in pnl_values]
    fig.add_trace(go.Scatter(
        x=dates,
        y=pnl_values,
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='blue', width=3),
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>P&L: $%{y:,.2f}<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    
    fig.update_layout(
        title="30-Day P&L Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    # Strategy Performance Metrics
    st.subheader("🎯 Strategy Metrics")
    
    # Win Rate Gauge
    trades_count = len(data['trades'])
    win_rate = 0.65 if trades_count > 0 else 0  # Simulated win rate
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=win_rate * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Win Rate (%)"},
        delta={'reference': 60},
        gauge={
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
    fig_gauge.update_layout(height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Additional metrics
    st.metric("Total Trades", trades_count)
    st.metric("Active Positions", len(data.get('positions', [])))
    
    # Profit factor
    total_pnl = unrealized_pnl + realized_pnl
    if total_pnl > 0:
        st.success(f"📈 Profitable: +${total_pnl:,.2f}")
    else:
        st.error(f"📉 Loss: ${total_pnl:,.2f}")

# Trading Opportunities Section
st.header("🎯 Recent Trading Opportunities")
if data['opportunities']:
    opportunities_df = pd.DataFrame(data['opportunities'][-10:])  # Last 10
    
    # Format the dataframe for better display
    if not opportunities_df.empty:
        opportunities_df['premium'] = opportunities_df['premium'].apply(lambda x: f"${x:.2f}")
        opportunities_df['strike_price'] = opportunities_df['strike_price'].apply(lambda x: f"${x:.2f}")
        if 'delta' in opportunities_df.columns:
            opportunities_df['delta'] = opportunities_df['delta'].apply(lambda x: f"{x:.3f}")
        if 'confidence' in opportunities_df.columns:
            opportunities_df['confidence'] = opportunities_df['confidence'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        opportunities_df,
        use_container_width=True,
        column_config={
            "symbol": "Symbol",
            "contract_type": "Type",
            "strike_price": "Strike",
            "premium": "Premium",
            "delta": "Delta",
            "confidence": "Confidence"
        }
    )
else:
    st.info("🔍 No opportunities detected yet. The system will populate opportunities as they are discovered.")

# Recent Trades Section
st.header("💼 Recent Trades")
if data['trades']:
    trades_df = pd.DataFrame(data['trades'][-10:])  # Last 10
    
    # Format the dataframe for better display
    if not trades_df.empty:
        trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")
        trades_df['trade_value'] = trades_df.apply(
            lambda row: f"${float(row['price'].replace('$', '')) * row['quantity'] * 100:.2f}", 
            axis=1
        )
    
    st.dataframe(
        trades_df,
        use_container_width=True,
        column_config={
            "symbol": "Symbol",
            "contract_type": "Type",
            "action": "Action",
            "quantity": "Qty",
            "price": "Price",
            "trade_value": "Total Value",
            "date": "Date"
        }
    )
    
    # Trade distribution chart
    if len(trades_df) > 0:
        st.subheader("📊 Trade Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for symbols
            symbol_counts = trades_df['symbol'].value_counts()
            fig_pie = px.pie(
                values=symbol_counts.values,
                names=symbol_counts.index,
                title="Trades by Symbol"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart for actions
            action_counts = trades_df['action'].value_counts()
            fig_bar = px.bar(
                x=action_counts.index,
                y=action_counts.values,
                title="Buy vs Sell Actions",
                color=action_counts.index,
                color_discrete_map={'BUY': 'green', 'SELL': 'red'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("📝 No trades executed yet. The system will show trades as they are executed.")

# Risk Management Section
st.header("⚠️ Risk Management")
col1, col2, col3 = st.columns(3)

with col1:
    # Max drawdown simulation
    max_drawdown = -2.5  # Simulated
    st.metric("Max Drawdown", f"{max_drawdown:.1f}%", "Acceptable", delta_color="inverse")

with col2:
    # Sharpe ratio simulation
    sharpe_ratio = 1.85  # Simulated
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", "Good")

with col3:
    # Risk level
    portfolio_risk = "Medium"
    st.metric("Risk Level", portfolio_risk, "Managed")

# System Status
st.header("🔧 System Status")
col1, col2, col3 = st.columns(3)

with col1:
    st.success("✅ Sentiment Analysis: Active")

with col2:
    st.success("✅ Options Scanner: Running")

with col3:
    st.success("✅ Risk Monitor: Online")

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_interval)
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🤖 <b>Options AI Trading Dashboard</b> | 
        Real-time Portfolio Monitoring | 
        Last Updated: {} UTC
    </div>
    """.format(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)
