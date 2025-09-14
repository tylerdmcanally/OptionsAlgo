"""
Strategy Framework for Options AI Backtesting
Implements various options strategies with FinBERT sentiment integration
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from loguru import logger

from backtest_engine import OptionsPosition, BacktestConfig
from options_pricing import AdvancedOptionsPricer, OptionsGreeks


@dataclass
class StrategySignal:
    """Strategy signal for entering/exiting positions"""
    symbol: str
    signal_type: str  # 'enter', 'exit', 'adjust'
    option_type: str  # 'call', 'put', 'spread'
    confidence: float  # 0-1
    reasoning: str
    expiry_target: datetime
    strike_target: float
    quantity: int
    metadata: Dict = None


class BaseOptionsStrategy(ABC):
    """Base class for all options strategies"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.pricer = AdvancedOptionsPricer()
        
    @abstractmethod
    async def generate_signals(self, market_data: Dict, sentiment_score: float, 
                             current_positions: List[OptionsPosition]) -> List[StrategySignal]:
        """Generate trading signals based on market data and sentiment"""
        pass
    
    @abstractmethod
    def risk_check(self, signal: StrategySignal, portfolio_value: float) -> bool:
        """Check if signal passes risk management criteria"""
        pass


class SentimentMomentumStrategy(BaseOptionsStrategy):
    """
    Strategy that trades options based on sentiment momentum
    - Strong positive sentiment -> Long calls
    - Strong negative sentiment -> Long puts
    - Sentiment reversal -> Close positions
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            'sentiment_threshold': 0.3,
            'momentum_lookback': 5,
            'max_dte': 45,
            'min_dte': 7,
            'target_delta': 0.3,
            'stop_loss_pct': 0.5,
            'take_profit_pct': 1.0
        }
        if config:
            default_config.update(config)
        
        super().__init__("SentimentMomentum", default_config)
    
    async def generate_signals(self, market_data: Dict, sentiment_score: float, 
                             current_positions: List[OptionsPosition]) -> List[StrategySignal]:
        """Generate signals based on sentiment momentum"""
        signals = []
        symbol = market_data['symbol']
        stock_price = market_data['stock_price']
        sentiment_history = market_data.get('sentiment_history', [])
        
        # Calculate sentiment momentum
        if len(sentiment_history) >= self.config['momentum_lookback']:
            recent_sentiment = sentiment_history[-self.config['momentum_lookback']:]
            sentiment_trend = np.mean(np.diff(recent_sentiment))
        else:
            sentiment_trend = 0
        
        # Check for entry signals
        current_symbol_positions = [p for p in current_positions if p.symbol == symbol]
        
        if not current_symbol_positions:  # No existing positions
            if abs(sentiment_score) >= self.config['sentiment_threshold']:
                
                # Balanced entry criteria - strong but not too restrictive
                if abs(sentiment_score) >= 0.35 and abs(sentiment_trend) >= 0.2:  # Relaxed slightly for testing
                    # Determine option type based on sentiment
                    if sentiment_score > 0:
                        option_type = 'call'
                        target_strike = self._calculate_target_strike(stock_price, 'call')
                    else:
                        option_type = 'put'
                        target_strike = self._calculate_target_strike(stock_price, 'put')
                    
                    # Shorter expiry to reduce time decay risk (max 2 weeks)
                    target_expiry = datetime.now() + timedelta(days=min(self.config['max_dte'] // 3, 14))
                    
                    # Much more conservative confidence calculation
                    base_confidence = min(abs(sentiment_score) * 1.2, 0.85)  # Reduced multiplier
                    trend_boost = min(abs(sentiment_trend) * 0.3, 0.15)  # Small trend boost
                    final_confidence = min(base_confidence + trend_boost, 1.0)
                    
                    signal = StrategySignal(
                        symbol=symbol,
                        signal_type='enter',
                        option_type=option_type,
                        confidence=final_confidence,
                        reasoning=f"Strong {'bullish' if sentiment_score > 0 else 'bearish'} sentiment ({sentiment_score:.3f}) with trend ({sentiment_trend:.3f})",
                        expiry_target=target_expiry,
                        strike_target=target_strike,
                        quantity=1,  # Always start with 1 contract
                        metadata={
                            'sentiment_score': sentiment_score,
                            'sentiment_trend': sentiment_trend,
                            'target_delta': self.config['target_delta'],
                            'entry_criteria': 'strict_sentiment_and_trend'
                        }
                    )
                    signals.append(signal)
        
        else:
            # Check exit conditions for existing positions
            for position in current_symbol_positions:
                exit_signal = self._check_exit_conditions(position, sentiment_score, sentiment_trend)
                if exit_signal:
                    signals.append(exit_signal)
        
        return signals
    
    def _calculate_target_strike(self, stock_price: float, option_type: str) -> float:
        """Calculate target strike much closer to ATM for better probability"""
        target_delta = self.config['target_delta']
        
        if option_type == 'call':
            # For calls, stay closer to ATM - max 5% OTM
            strike_multiplier = 1 + min((0.3 - target_delta) * 0.3, 0.05)  # Much closer to ATM
            return stock_price * strike_multiplier
        else:
            # For puts, stay closer to ATM - max 5% OTM  
            strike_multiplier = 1 - min((0.3 - abs(target_delta)) * 0.3, 0.05)
            return stock_price * strike_multiplier
    
    def _check_exit_conditions(self, position: OptionsPosition, current_sentiment: float, 
                              sentiment_trend: float) -> Optional[StrategySignal]:
        """Enhanced exit conditions with profit targets and stop losses"""
        
        # Time-based exit - close if less than 7 days to expiry to avoid time decay
        days_to_expiry = (position.expiry - datetime.now()).days
        if days_to_expiry <= 7:
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                option_type=position.option_type,
                confidence=0.95,
                reasoning=f"Time decay risk - {days_to_expiry} days to expiry",
                expiry_target=position.expiry,
                strike_target=position.strike,
                quantity=position.quantity
            )
        
        # Sentiment reversal check - much more sensitive
        sentiment_threshold = 0.2  # Lower threshold for quicker exits
        if position.option_type == 'call' and current_sentiment < -sentiment_threshold:
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                option_type=position.option_type,
                confidence=0.9,
                reasoning=f"Sentiment reversed to bearish ({current_sentiment:.3f})",
                expiry_target=position.expiry,
                strike_target=position.strike,
                quantity=position.quantity
            )
        
        elif position.option_type == 'put' and current_sentiment > sentiment_threshold:
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                option_type=position.option_type,
                confidence=0.9,
                reasoning=f"Sentiment reversed to bullish ({current_sentiment:.3f})",
                expiry_target=position.expiry,
                strike_target=position.strike,
                quantity=position.quantity
            )
        
        # Exit on trend reversal
        original_sentiment = position.metadata.get('sentiment_score', 0) if hasattr(position, 'metadata') else 0
        if abs(current_sentiment - original_sentiment) > 0.4:  # Significant sentiment change
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                option_type=position.option_type,
                confidence=0.85,
                reasoning=f"Major sentiment change: {original_sentiment:.3f} -> {current_sentiment:.3f}",
                expiry_target=position.expiry,
                strike_target=position.strike,
                quantity=position.quantity
            )
        
        elif position.option_type == 'put' and current_sentiment > self.config['sentiment_threshold']:
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                option_type=position.option_type,
                confidence=0.8,
                reasoning="Sentiment reversed to bullish",
                expiry_target=position.expiry,
                strike_target=position.strike,
                quantity=position.quantity
            )
        
        # Time decay check (exit if close to expiry)
        days_to_expiry = (position.expiry - datetime.now()).days
        if days_to_expiry <= self.config['min_dte']:
            return StrategySignal(
                symbol=position.symbol,
                signal_type='exit',
                option_type=position.option_type,
                confidence=0.9,
                reasoning=f"Close to expiry ({days_to_expiry} days)",
                expiry_target=position.expiry,
                strike_target=position.strike,
                quantity=position.quantity
            )
        
        return None
    
    def risk_check(self, signal: StrategySignal, portfolio_value: float) -> bool:
        """Enhanced risk management checks"""
        # Much smaller position sizes to preserve capital (2% max per position)
        max_position_value = portfolio_value * 0.02
        
        # More realistic option value estimate 
        estimated_option_value = signal.strike_target * 0.02  # Very conservative estimate
        position_value = estimated_option_value * signal.quantity * 100
        
        if position_value > max_position_value:
            logger.warning(f"Position size too large: ${position_value:.2f} > ${max_position_value:.2f}")
            return False
        
        # Balanced confidence threshold for selective but active trading
        if signal.confidence < 0.45:  # Reduced from 0.65 to 0.45 for more opportunities
            logger.warning(f"Signal confidence too low: {signal.confidence:.2f}")
            return False
        
        # Strong sentiment requirement but not too restrictive
        if abs(signal.metadata.get('sentiment_score', 0)) < 0.3:
            logger.warning(f"Sentiment signal too weak: {signal.metadata.get('sentiment_score', 0):.3f}")
            return False
        
        return True


class VolatilityCaptureStrategy(BaseOptionsStrategy):
    """
    Strategy that captures volatility expansions and contractions
    - High IV with negative sentiment -> Sell premium (credit spreads)
    - Low IV with strong sentiment -> Buy premium (long options)
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            'iv_percentile_threshold': 70,  # High IV threshold
            'iv_low_threshold': 30,         # Low IV threshold
            'sentiment_threshold': 0.2,
            'vega_target': 100,            # Target vega exposure
            'max_short_delta': 0.2,        # Max delta for short options
        }
        if config:
            default_config.update(config)
        
        super().__init__("VolatilityCapture", default_config)
    
    async def generate_signals(self, market_data: Dict, sentiment_score: float, 
                             current_positions: List[OptionsPosition]) -> List[StrategySignal]:
        """Generate volatility-based signals"""
        signals = []
        symbol = market_data['symbol']
        stock_price = market_data['stock_price']
        implied_vol = market_data.get('implied_vol', 0.25)
        iv_percentile = market_data.get('iv_percentile', 50)
        
        current_symbol_positions = [p for p in current_positions if p.symbol == symbol]
        
        if not current_symbol_positions:
            
            # High IV + Strong sentiment -> Sell premium
            if (iv_percentile >= self.config['iv_percentile_threshold'] and 
                abs(sentiment_score) >= self.config['sentiment_threshold']):
                
                # Sell OTM options in direction opposite to sentiment
                if sentiment_score > 0:
                    # Bullish sentiment -> sell puts
                    option_type = 'put'
                    target_strike = stock_price * 0.95  # 5% OTM put
                else:
                    # Bearish sentiment -> sell calls
                    option_type = 'call'
                    target_strike = stock_price * 1.05  # 5% OTM call
                
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type='enter',
                    option_type=f'short_{option_type}',
                    confidence=0.7,
                    reasoning=f"High IV ({iv_percentile}%ile) with {option_type} opportunity",
                    expiry_target=datetime.now() + timedelta(days=30),
                    strike_target=target_strike,
                    quantity=1,
                    metadata={
                        'strategy_type': 'sell_premium',
                        'iv_percentile': iv_percentile,
                        'sentiment_score': sentiment_score
                    }
                )
                signals.append(signal)
            
            # Low IV + Strong sentiment -> Buy premium
            elif (iv_percentile <= self.config['iv_low_threshold'] and 
                  abs(sentiment_score) >= self.config['sentiment_threshold']):
                
                option_type = 'call' if sentiment_score > 0 else 'put'
                target_strike = self._calculate_atm_strike(stock_price, option_type)
                
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type='enter',
                    option_type=option_type,
                    confidence=0.8,
                    reasoning=f"Low IV ({iv_percentile}%ile) with strong sentiment",
                    expiry_target=datetime.now() + timedelta(days=45),
                    strike_target=target_strike,
                    quantity=1,
                    metadata={
                        'strategy_type': 'buy_premium',
                        'iv_percentile': iv_percentile,
                        'sentiment_score': sentiment_score
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _calculate_atm_strike(self, stock_price: float, option_type: str) -> float:
        """Calculate near-the-money strike"""
        # Round to nearest $5 for liquid strikes
        return round(stock_price / 5) * 5
    
    def risk_check(self, signal: StrategySignal, portfolio_value: float) -> bool:
        """Risk checks for volatility strategy"""
        # Check for short option risk
        if 'short_' in signal.option_type:
            # Limit short option exposure
            max_short_exposure = portfolio_value * 0.1  # 10% max short exposure
            estimated_margin = signal.strike_target * signal.quantity * 100 * 0.2  # 20% margin req
            
            if estimated_margin > max_short_exposure:
                return False
        
        return super().risk_check(signal, portfolio_value)


class MeanReversionStrategy(BaseOptionsStrategy):
    """
    Mean reversion strategy using options
    - Oversold with positive sentiment -> Long calls
    - Overbought with negative sentiment -> Long puts
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'sentiment_confirmation': 0.15,
            'bollinger_std': 2.0,
            'mean_reversion_window': 20
        }
        if config:
            default_config.update(config)
        
        super().__init__("MeanReversion", default_config)
    
    async def generate_signals(self, market_data: Dict, sentiment_score: float, 
                             current_positions: List[OptionsPosition]) -> List[StrategySignal]:
        """Generate mean reversion signals"""
        signals = []
        symbol = market_data['symbol']
        stock_price = market_data['stock_price']
        rsi = market_data.get('rsi', 50)
        price_history = market_data.get('price_history', [])
        
        if len(price_history) < self.config['mean_reversion_window']:
            return signals
        
        # Calculate Bollinger Bands
        prices = pd.Series(price_history)
        rolling_mean = prices.rolling(self.config['mean_reversion_window']).mean().iloc[-1]
        rolling_std = prices.rolling(self.config['mean_reversion_window']).std().iloc[-1]
        
        upper_band = rolling_mean + (self.config['bollinger_std'] * rolling_std)
        lower_band = rolling_mean - (self.config['bollinger_std'] * rolling_std)
        
        current_symbol_positions = [p for p in current_positions if p.symbol == symbol]
        
        if not current_symbol_positions:
            
            # Oversold + positive sentiment
            if (rsi <= self.config['rsi_oversold'] and 
                stock_price <= lower_band and
                sentiment_score >= self.config['sentiment_confirmation']):
                
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type='enter',
                    option_type='call',
                    confidence=0.7,
                    reasoning=f"Oversold reversal: RSI={rsi:.1f}, price below lower BB",
                    expiry_target=datetime.now() + timedelta(days=30),
                    strike_target=stock_price * 1.02,  # Slightly OTM
                    quantity=1,
                    metadata={
                        'rsi': rsi,
                        'bb_position': 'below_lower',
                        'sentiment_score': sentiment_score
                    }
                )
                signals.append(signal)
            
            # Overbought + negative sentiment
            elif (rsi >= self.config['rsi_overbought'] and 
                  stock_price >= upper_band and
                  sentiment_score <= -self.config['sentiment_confirmation']):
                
                signal = StrategySignal(
                    symbol=symbol,
                    signal_type='enter',
                    option_type='put',
                    confidence=0.7,
                    reasoning=f"Overbought reversal: RSI={rsi:.1f}, price above upper BB",
                    expiry_target=datetime.now() + timedelta(days=30),
                    strike_target=stock_price * 0.98,  # Slightly OTM
                    quantity=1,
                    metadata={
                        'rsi': rsi,
                        'bb_position': 'above_upper',
                        'sentiment_score': sentiment_score
                    }
                )
                signals.append(signal)
        
        return signals
    
    def risk_check(self, signal: StrategySignal, portfolio_value: float) -> bool:
        """Risk management for mean reversion strategy"""
        return super().risk_check(signal, portfolio_value)


class StrategyManager:
    """
    Manages multiple strategies and combines their signals
    """
    
    def __init__(self):
        self.strategies: List[BaseOptionsStrategy] = []
        self.strategy_weights: Dict[str, float] = {}
    
    def add_strategy(self, strategy: BaseOptionsStrategy, weight: float = 1.0):
        """Add a strategy with specified weight"""
        self.strategies.append(strategy)
        self.strategy_weights[strategy.name] = weight
    
    async def generate_combined_signals(self, market_data: Dict, sentiment_score: float,
                                      current_positions: List[OptionsPosition]) -> List[StrategySignal]:
        """Generate signals from all strategies and combine them"""
        all_signals = []
        
        for strategy in self.strategies:
            try:
                strategy_signals = await strategy.generate_signals(market_data, sentiment_score, current_positions)
                
                # Apply strategy weight to confidence
                weight = self.strategy_weights[strategy.name]
                for signal in strategy_signals:
                    # Apply weight more gently to preserve confidence scores
                    signal.confidence = min(signal.confidence * (0.7 + weight * 0.6), 1.0)
                    signal.metadata = signal.metadata or {}
                    signal.metadata['strategy'] = strategy.name
                
                all_signals.extend(strategy_signals)
                
            except Exception as e:
                logger.error(f"Error generating signals for {strategy.name}: {e}")
        
        # Remove duplicate signals and combine confidence
        combined_signals = self._combine_duplicate_signals(all_signals)
        
        # Apply risk checks
        portfolio_value = sum([pos.entry_price * pos.quantity * 100 for pos in current_positions])
        portfolio_value = max(portfolio_value, 100000)  # Minimum portfolio value
        
        risk_checked_signals = []
        for signal in combined_signals:
            strategy = next(s for s in self.strategies if s.name == signal.metadata.get('strategy'))
            if strategy.risk_check(signal, portfolio_value):
                risk_checked_signals.append(signal)
        
        return risk_checked_signals
    
    def _combine_duplicate_signals(self, signals: List[StrategySignal]) -> List[StrategySignal]:
        """Combine duplicate signals from different strategies"""
        signal_groups = {}
        
        for signal in signals:
            key = (signal.symbol, signal.option_type, signal.signal_type, 
                  signal.strike_target, signal.expiry_target.date())
            
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        combined_signals = []
        for group in signal_groups.values():
            if len(group) == 1:
                combined_signals.append(group[0])
            else:
                # Combine signals
                base_signal = group[0]
                total_confidence = sum(s.confidence for s in group)
                strategies = [s.metadata.get('strategy', 'unknown') for s in group]
                
                base_signal.confidence = min(total_confidence, 1.0)
                base_signal.reasoning += f" (Combined from: {', '.join(strategies)})"
                base_signal.metadata['combined_strategies'] = strategies
                
                combined_signals.append(base_signal)
        
        return combined_signals


# Example usage
def create_default_strategy_manager() -> StrategyManager:
    """Create default strategy manager with common strategies"""
    manager = StrategyManager()
    
    # Add strategies with weights
    manager.add_strategy(SentimentMomentumStrategy(), weight=0.4)
    manager.add_strategy(VolatilityCaptureStrategy(), weight=0.3)
    manager.add_strategy(MeanReversionStrategy(), weight=0.3)
    
    return manager


if __name__ == "__main__":
    # Test strategy creation
    manager = create_default_strategy_manager()
    print(f"Created strategy manager with {len(manager.strategies)} strategies")
    for strategy in manager.strategies:
        print(f"  - {strategy.name} (weight: {manager.strategy_weights[strategy.name]})")
