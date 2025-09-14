"""
Options AI Scanner Backtesting Engine
Comprehensive backtesting framework for options strategies with FinBERT sentiment analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our existing components
from sentiment_analyzer import SentimentAnalyzer
from options_scanner import OptionsScanner
from polygon_manager import PolygonAPIManager
from scoring_engine import ScoringEngine


@dataclass
class OptionsPosition:
    """Represents an options position"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: datetime
    entry_date: datetime
    entry_price: float
    quantity: int
    entry_iv: Optional[float] = None
    entry_delta: Optional[float] = None
    entry_gamma: Optional[float] = None
    entry_theta: Optional[float] = None
    entry_vega: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    max_positions: int = 10
    position_size_pct: float = 0.05  # 5% of capital per position
    commission_per_contract: float = 1.0
    max_days_to_expiry: int = 45
    min_days_to_expiry: int = 7
    stop_loss_pct: float = 0.50  # 50% stop loss
    take_profit_pct: float = 1.00  # 100% take profit
    sentiment_threshold: float = 0.1  # Minimum absolute sentiment score
    scoring_threshold: float = 0.6   # Minimum composite score


@dataclass
class BacktestResults:
    """Backtesting results container"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    positions: List[OptionsPosition]
    equity_curve: pd.Series
    daily_returns: pd.Series
    metrics: Dict[str, Any]


class OptionsBacktester:
    """
    Comprehensive backtesting engine for options strategies
    """
    
    def __init__(self, config: BacktestConfig, strategy_manager=None):
        self.config = config
        self.strategy_manager = strategy_manager
        
        # Initialize components (will be set up by integrated backtester)
        self.sentiment_analyzer = None
        self.options_scanner = None
        self.scoring_engine = None
        self.polygon_manager = None
        
        # Portfolio tracking
        self.current_capital = config.initial_capital
        self.positions: List[OptionsPosition] = []
        self.closed_positions: List[OptionsPosition] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[Tuple[datetime, float]] = []
        
        logger.info(f"Backtest initialized: {config.start_date} to {config.end_date}")
    
    def set_components(self, sentiment_analyzer, options_scanner, scoring_engine, polygon_manager):
        """Set the components after initialization"""
        self.sentiment_analyzer = sentiment_analyzer
        self.options_scanner = options_scanner
        self.scoring_engine = scoring_engine
        self.polygon_manager = polygon_manager
    
    async def run_backtest(self, market_data_cache: Dict) -> BacktestResults:
        """
        Run complete backtesting simulation
        """
        # Store the market data cache for use in price lookups
        self.market_data_cache = market_data_cache
        
        symbols = list(market_data_cache.keys())
        logger.info(f"Starting backtest for {len(symbols)} symbols")
        
        # Generate trading days
        trading_days = pd.bdate_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='B'  # Business days
        )
        
        for current_date in trading_days:
            await self._process_trading_day(current_date, market_data_cache)
            
            # Record daily equity
            daily_equity = self._calculate_portfolio_value(current_date)
            self.equity_curve.append((current_date, daily_equity))
            
            # Calculate daily return
            if len(self.equity_curve) > 1:
                prev_equity = self.equity_curve[-2][1]
                daily_return = (daily_equity - prev_equity) / prev_equity
                self.daily_returns.append((current_date, daily_return))
        
        return self._generate_results()
    
    async def _process_trading_day(self, current_date: datetime, market_data_cache: Dict):
        """Process a single trading day"""
        logger.info(f"Processing {current_date.strftime('%Y-%m-%d')}")
        
        # 1. Check and close expired/stopped positions
        await self._manage_existing_positions(current_date)
        
        # 2. Scan for new opportunities if we have capacity
        if len(self.positions) < self.config.max_positions:
            symbols = list(market_data_cache.keys())
            await self._scan_new_opportunities(current_date, market_data_cache)
    
    async def _manage_existing_positions(self, current_date: datetime):
        """Manage existing positions - check exits, stops, etc."""
        positions_to_close = []
        
        for position in self.positions:
            # Check if expired
            if current_date >= position.expiry:
                position.exit_date = current_date
                position.exit_price = 0.01  # Assume worthless expiry
                position.exit_reason = "expired"
                positions_to_close.append(position)
                continue
            
            # Get current option price for P&L calculation
            try:
                current_price = await self._get_option_price(position, current_date)
                if current_price is None:
                    continue
                
                # Check stop loss
                unrealized_pnl = (current_price - position.entry_price) * position.quantity
                if unrealized_pnl < -abs(position.entry_price * position.quantity * self.config.stop_loss_pct):
                    position.exit_date = current_date
                    position.exit_price = current_price
                    position.exit_reason = "stop_loss"
                    positions_to_close.append(position)
                    continue
                
                # Check take profit
                if unrealized_pnl > position.entry_price * position.quantity * self.config.take_profit_pct:
                    position.exit_date = current_date
                    position.exit_price = current_price
                    position.exit_reason = "take_profit"
                    positions_to_close.append(position)
                    continue
                
                # Check days to expiry threshold for early exit
                days_to_expiry = (position.expiry - current_date).days
                if days_to_expiry <= 7:  # Close within 7 days of expiry
                    position.exit_date = current_date
                    position.exit_price = current_price
                    position.exit_reason = "dte_threshold"
                    positions_to_close.append(position)
                    
            except Exception as e:
                logger.warning(f"Error managing position {position.symbol}: {e}")
        
        # Close positions and update capital
        for position in positions_to_close:
            self._close_position(position)
    
    async def _scan_new_opportunities(self, current_date: datetime, market_data_cache: Dict):
        """Scan for new trading opportunities"""
        
        # Get symbols from market data cache
        symbols = list(market_data_cache.keys())
        
        # Limit concurrent analysis to avoid API rate limits
        semaphore = asyncio.Semaphore(5)
        
        async def analyze_symbol(symbol: str):
            async with semaphore:
                try:
                    return await self._analyze_symbol_opportunity(symbol, current_date, market_data_cache)
                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {e}")
                    return None
        
        # Analyze all symbols concurrently
        tasks = [analyze_symbol(symbol) for symbol in symbols]
        opportunities = await asyncio.gather(*tasks)
        
        # Filter valid opportunities
        valid_opportunities = [opp for opp in opportunities if opp is not None]
        
        # Sort by composite score
        valid_opportunities.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Open new positions
        for opportunity in valid_opportunities:
            if len(self.positions) >= self.config.max_positions:
                break
            
            if opportunity['composite_score'] >= self.config.scoring_threshold:
                await self._open_position(opportunity, current_date)
    
    async def _analyze_symbol_opportunity(self, symbol: str, current_date: datetime, market_data_cache: Dict) -> Optional[Dict]:
        """Analyze a single symbol for trading opportunities"""
        try:
            # Get market data from cache
            market_data = market_data_cache.get(symbol)
            if not market_data:
                return None
            
            # Use strategy manager if available
            if self.strategy_manager:
                # Get current sentiment score - use cached or generate
                current_sentiment = market_data.get('sentiment_scores', [0.0])
                if isinstance(current_sentiment, list):
                    sentiment_score = current_sentiment[-1] if current_sentiment else 0.0
                else:
                    sentiment_score = current_sentiment
                
                # Skip if sentiment not strong enough
                if abs(sentiment_score) < self.config.sentiment_threshold:
                    return None
                
                # Generate strategy signals
                signals = await self.strategy_manager.generate_combined_signals(
                    market_data, sentiment_score, self.positions
                )
                
                if not signals:
                    return None
                
                # Use the highest confidence signal
                best_signal = max(signals, key=lambda s: s.confidence)
                
                return {
                    'symbol': symbol,
                    'sentiment_score': sentiment_score,
                    'stock_price': market_data['stock_price'],
                    'option_type': best_signal.option_type,
                    'strike': best_signal.strike_target,
                    'expiry': best_signal.expiry_target,
                    'signal_reasoning': best_signal.reasoning,
                    'composite_score': best_signal.confidence,
                    'strategy': best_signal.metadata.get('strategy', 'unknown') if best_signal.metadata else 'unknown'
                }
            
            # Fallback: simple sentiment-based analysis
            else:
                sentiment_score = await self.sentiment_analyzer.get_sentiment(symbol) if self.sentiment_analyzer else 0.0
                
                if abs(sentiment_score) < self.config.sentiment_threshold:
                    return None
                
                # Simple strategy: calls for positive sentiment, puts for negative
                option_type = 'call' if sentiment_score > 0 else 'put'
                stock_price = market_data['stock_price']
                strike = stock_price * (1.05 if option_type == 'call' else 0.95)  # 5% OTM
                
                return {
                    'symbol': symbol,
                    'sentiment_score': sentiment_score,
                    'stock_price': stock_price,
                    'option_type': option_type,
                    'strike': strike,
                    'expiry': current_date + timedelta(days=30),
                    'signal_reasoning': f'Simple sentiment-based {option_type}',
                    'composite_score': abs(sentiment_score),
                    'strategy': 'simple_sentiment'
                }
                
        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
            return None
    
    async def _open_position(self, opportunity: Dict, current_date: datetime):
        """Open a new options position"""
        try:
            symbol = opportunity['symbol']
            
            # Calculate position size
            position_value = self.current_capital * self.config.position_size_pct
            
            # Estimate option price (simplified - in real implementation would use pricing model)
            stock_price = opportunity['stock_price']
            strike = opportunity['strike']
            option_type = opportunity['option_type']
            
            # Simple option price estimation
            if option_type == 'call':
                intrinsic = max(0, stock_price - strike)
                time_value = strike * 0.05  # Rough estimate
            else:
                intrinsic = max(0, strike - stock_price)
                time_value = strike * 0.05
            
            estimated_option_price = intrinsic + time_value
            estimated_option_price = max(0.10, estimated_option_price)  # Minimum $0.10
            
            contracts = max(1, int(position_value / (estimated_option_price * 100)))
            
            # Check if we have enough capital
            total_cost = contracts * estimated_option_price * 100 + self.config.commission_per_contract * contracts
            if total_cost > self.current_capital * 0.8:  # Don't use more than 80% of capital
                return
            
            # Create position
            position = OptionsPosition(
                symbol=symbol,
                option_type=option_type,
                strike=strike,
                expiry=opportunity['expiry'],
                entry_date=current_date,
                entry_price=estimated_option_price,
                quantity=contracts
            )
            
            # Update capital
            self.current_capital -= total_cost
            self.positions.append(position)
            
            logger.info(f"Opened {contracts} {symbol} {option_type} ${strike} contracts @ ${estimated_option_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
    
    def _close_position(self, position: OptionsPosition):
        """Close an options position"""
        try:
            # Calculate P&L
            entry_cost = position.entry_price * position.quantity * 100
            exit_value = position.exit_price * position.quantity * 100
            commission = self.config.commission_per_contract * position.quantity * 2  # Entry + exit
            
            net_pnl = exit_value - entry_cost - commission
            
            # Update capital
            self.current_capital += exit_value - self.config.commission_per_contract * position.quantity
            
            # Move to closed positions
            self.positions.remove(position)
            self.closed_positions.append(position)
            
            logger.info(f"Closed position: {position.symbol} {position.option_type} - P&L: ${net_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def _get_option_price(self, position: OptionsPosition, current_date: datetime) -> Optional[float]:
        """Get current option price (simplified - in reality would use options pricing model)"""
        try:
            # This is a simplified implementation
            # In a real backtest, you'd want to:
            # 1. Get current stock price
            # 2. Use Black-Scholes or other pricing model
            # 3. Account for implied volatility changes
            # 4. Use actual historical options data if available
            
            stock_price = await self._get_stock_price(position.symbol, current_date)
            if stock_price is None:
                return None
            
            # Simple intrinsic value calculation (placeholder)
            if position.option_type == 'call':
                intrinsic = max(0, stock_price - position.strike)
            else:
                intrinsic = max(0, position.strike - stock_price)
            
            # Add some time value based on days to expiry
            days_to_expiry = (position.expiry - current_date).days
            time_value = max(0.01, intrinsic * 0.1 * (days_to_expiry / 30))
            
            return intrinsic + time_value
            
        except Exception as e:
            logger.error(f"Error getting option price: {e}")
            return None
    
    async def _get_stock_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get stock price for a given date"""
        try:
            # First try to use cached market data
            if symbol in self.market_data_cache:
                symbol_data = self.market_data_cache[symbol]
                
                # Handle mock data format where DataFrame is in 'price_data' key
                if isinstance(symbol_data, dict) and 'price_data' in symbol_data:
                    price_df = symbol_data['price_data']
                    if hasattr(price_df, 'columns') and 'close' in price_df.columns:
                        # Find the row closest to the requested date
                        if 'timestamp' in price_df.columns:
                            price_df = price_df.copy()
                            price_df['date_diff'] = abs((price_df['timestamp'] - date).dt.days)
                            closest_idx = price_df['date_diff'].idxmin()
                            return price_df.loc[closest_idx, 'close']
                        else:
                            # If no timestamp column, use the last available price
                            return price_df['close'].iloc[-1]
                
                # Handle direct DataFrame format
                elif hasattr(symbol_data, 'columns') and 'close' in symbol_data.columns:
                    # Find the row closest to the requested date
                    if 'timestamp' in symbol_data.columns:
                        symbol_data = symbol_data.copy()
                        symbol_data['date_diff'] = abs((symbol_data['timestamp'] - date).dt.days)
                        closest_idx = symbol_data['date_diff'].idxmin()
                        return symbol_data.loc[closest_idx, 'close']
                    else:
                        # If no timestamp column, use the last available price
                        return symbol_data['close'].iloc[-1]
            
            # Fallback to Polygon if available
            if self.polygon_manager:
                bars = await self.polygon_manager.get_daily_bars(
                    symbol,
                    from_date=date.strftime('%Y-%m-%d'),
                    to_date=date.strftime('%Y-%m-%d')
                )
                
                if bars and len(bars) > 0:
                    return bars[0].get('close', bars[0].get('c'))
            
            logger.warning(f"No price data available for {symbol} on {date}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting stock price for {symbol}: {e}")
            return None
    
    def _calculate_portfolio_value(self, current_date: datetime) -> float:
        """Calculate total portfolio value including unrealized P&L"""
        total_value = self.current_capital
        
        # Add unrealized P&L from open positions (simplified calculation)
        for position in self.positions:
            try:
                # Get current stock price from cache
                if position.symbol in self.market_data_cache:
                    symbol_data = self.market_data_cache[position.symbol]
                    
                    # Handle mock data format where DataFrame is in 'price_data' key
                    if isinstance(symbol_data, dict) and 'price_data' in symbol_data:
                        price_df = symbol_data['price_data']
                        if hasattr(price_df, 'columns') and 'close' in price_df.columns and len(price_df) > 0:
                            current_stock_price = price_df['close'].iloc[-1]
                        else:
                            continue
                    # Handle direct DataFrame format
                    elif hasattr(symbol_data, 'columns') and 'close' in symbol_data.columns and len(symbol_data) > 0:
                        current_stock_price = symbol_data['close'].iloc[-1]
                    else:
                        continue
                        
                    # Simplified option value approximation
                    # In reality, would use Black-Scholes with current Greeks
                    intrinsic_value = max(0, current_stock_price - position.strike) if position.option_type == 'call' else max(0, position.strike - current_stock_price)
                    time_value = position.entry_price * 0.5  # Assume 50% time decay
                    current_option_price = intrinsic_value + time_value
                    
                    position_value = current_option_price * position.quantity * 100
                    total_value += position_value - (position.entry_price * position.quantity * 100)
                        
            except Exception as e:
                logger.debug(f"Error calculating position value for {position.symbol}: {e}")
                
        return total_value
    
    def _generate_results(self) -> BacktestResults:
        """Generate comprehensive backtest results"""
        
        # Convert to pandas for calculations
        equity_df = pd.DataFrame(self.equity_curve, columns=['date', 'equity'])
        equity_df.set_index('date', inplace=True)
        equity_series = equity_df['equity']
        
        returns_df = pd.DataFrame(self.daily_returns, columns=['date', 'return'])
        returns_df.set_index('date', inplace=True)
        returns_series = returns_df['return']
        
        # Calculate metrics
        total_return = (equity_series.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized return
        days = (self.config.end_date - self.config.start_date).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = returns_series - 0.02/252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Max drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        winning_trades = [p for p in self.closed_positions if self._calculate_pnl(p) > 0]
        losing_trades = [p for p in self.closed_positions if self._calculate_pnl(p) <= 0]
        
        win_rate = len(winning_trades) / len(self.closed_positions) if self.closed_positions else 0
        avg_win = np.mean([self._calculate_pnl(p) for p in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([self._calculate_pnl(p) for p in losing_trades]) if losing_trades else 0
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=len(self.closed_positions),
            positions=self.closed_positions,
            equity_curve=equity_series,
            daily_returns=returns_series,
            metrics={
                'initial_capital': self.config.initial_capital,
                'final_capital': equity_series.iloc[-1],
                'total_positions_opened': len(self.closed_positions),
                'avg_position_size': np.mean([p.quantity for p in self.closed_positions]) if self.closed_positions else 0
            }
        )
    
    def _calculate_pnl(self, position: OptionsPosition) -> float:
        """Calculate P&L for a closed position"""
        if position.exit_price is None:
            return 0
        
        entry_cost = position.entry_price * position.quantity * 100
        exit_value = position.exit_price * position.quantity * 100
        commission = self.config.commission_per_contract * position.quantity * 2
        
        return exit_value - entry_cost - commission


# Example usage
async def run_example_backtest():
    """Example of how to run a backtest"""
    
    # Configuration
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        initial_capital=100000.0,
        max_positions=5,
        position_size_pct=0.10
    )
    
    # Initialize backtester
    backtester = OptionsBacktester(config, "your_polygon_api_key")
    
    # Run backtest
    symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'GOOGL']
    results = await backtester.run_backtest(symbols)
    
    # Print results
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annual Return: {results.annual_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Total Trades: {results.total_trades}")


if __name__ == "__main__":
    asyncio.run(run_example_backtest())
