"""
Advanced Options Pricing Module for Backtesting
Implements Black-Scholes, Greeks calculations, and implied volatility modeling
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from scipy.stats import norm
from scipy.optimize import brentq
from loguru import logger
import yfinance as yf


class OptionsGreeks:
    """Container for options Greeks"""
    def __init__(self, delta: float, gamma: float, theta: float, vega: float, rho: float):
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.rho = rho


class BlackScholesModel:
    """
    Black-Scholes options pricing model with Greeks calculation
    """
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter"""
        return BlackScholesModel._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European call option price
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        """
        if T <= 0:
            return max(0, S - K)
        
        d1 = BlackScholesModel._d1(S, K, T, r, sigma)
        d2 = BlackScholesModel._d2(S, K, T, r, sigma)
        
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(0, call)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price"""
        if T <= 0:
            return max(0, K - S)
        
        d1 = BlackScholesModel._d1(S, K, T, r, sigma)
        d2 = BlackScholesModel._d2(S, K, T, r, sigma)
        
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(0, put)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> OptionsGreeks:
        """Calculate all Greeks for an option"""
        if T <= 0:
            # At expiration
            if option_type.lower() == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            return OptionsGreeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        
        d1 = BlackScholesModel._d1(S, K, T, r, sigma)
        d2 = BlackScholesModel._d2(S, K, T, r, sigma)
        
        # Common calculations
        phi_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        
        if option_type.lower() == 'call':
            # Call Greeks
            delta = cdf_d1
            gamma = phi_d1 / (S * sigma * np.sqrt(T))
            theta = (-S * phi_d1 * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * cdf_d2) / 365
            vega = S * phi_d1 * np.sqrt(T) / 100
            rho = K * T * np.exp(-r * T) * cdf_d2 / 100
        else:
            # Put Greeks
            delta = cdf_d1 - 1
            gamma = phi_d1 / (S * sigma * np.sqrt(T))
            theta = (-S * phi_d1 * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            vega = S * phi_d1 * np.sqrt(T) / 100
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return OptionsGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str) -> Optional[float]:
        """Calculate implied volatility using Brent's method"""
        if T <= 0:
            return None
        
        def objective(sigma):
            if option_type.lower() == 'call':
                model_price = BlackScholesModel.call_price(S, K, T, r, sigma)
            else:
                model_price = BlackScholesModel.put_price(S, K, T, r, sigma)
            return model_price - market_price
        
        try:
            # Search for IV between 0.01% and 500%
            iv = brentq(objective, 0.0001, 5.0, xtol=1e-6)
            return iv
        except (ValueError, RuntimeError):
            return None


class VolatilitySurface:
    """
    Implied volatility surface modeling
    """
    
    def __init__(self):
        self.vol_data: Dict[Tuple[float, float], float] = {}  # (strike, dte) -> iv
        self.atm_vol_term_structure: Dict[float, float] = {}  # dte -> atm_iv
    
    def add_iv_point(self, strike: float, days_to_expiry: float, iv: float):
        """Add an IV data point"""
        self.vol_data[(strike, days_to_expiry)] = iv
    
    def get_interpolated_iv(self, strike: float, days_to_expiry: float, 
                           spot_price: float) -> Optional[float]:
        """Get interpolated implied volatility"""
        if not self.vol_data:
            return 0.25  # Default 25% volatility
        
        # Simple interpolation - in practice, you'd use more sophisticated methods
        moneyness = strike / spot_price
        
        # Find closest matches
        closest_matches = []
        for (k, dte), iv in self.vol_data.items():
            k_moneyness = k / spot_price
            distance = abs(k_moneyness - moneyness) + abs(dte - days_to_expiry) / 100
            closest_matches.append((distance, iv))
        
        if closest_matches:
            closest_matches.sort()
            # Weight by inverse distance
            total_weight = 0
            weighted_iv = 0
            for i, (distance, iv) in enumerate(closest_matches[:3]):  # Use top 3 matches
                weight = 1 / (distance + 0.01)  # Add small epsilon
                weighted_iv += weight * iv
                total_weight += weight
            
            return weighted_iv / total_weight if total_weight > 0 else 0.25
        
        return 0.25


class OptionsHedgeCalculator:
    """
    Calculate hedge ratios and portfolio Greeks
    """
    
    @staticmethod
    def calculate_delta_hedge_ratio(positions: list, underlying_price: float) -> float:
        """Calculate delta hedge ratio for a portfolio"""
        total_delta = 0
        for position in positions:
            greeks = position.get('greeks')
            if greeks:
                quantity = position.get('quantity', 0)
                total_delta += greeks.delta * quantity * 100  # 100 shares per contract
        
        return -total_delta  # Negative to hedge
    
    @staticmethod
    def calculate_portfolio_greeks(positions: list) -> OptionsGreeks:
        """Calculate aggregate Greeks for a portfolio"""
        total_delta = total_gamma = total_theta = total_vega = total_rho = 0
        
        for position in positions:
            greeks = position.get('greeks')
            if greeks:
                quantity = position.get('quantity', 0)
                multiplier = quantity * 100  # 100 shares per contract
                
                total_delta += greeks.delta * multiplier
                total_gamma += greeks.gamma * multiplier
                total_theta += greeks.theta * multiplier
                total_vega += greeks.vega * multiplier
                total_rho += greeks.rho * multiplier
        
        return OptionsGreeks(total_delta, total_gamma, total_theta, total_vega, total_rho)


class AdvancedOptionsPricer:
    """
    Advanced options pricing with volatility surface and risk management
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.vol_surface = VolatilitySurface()
        self.bs_model = BlackScholesModel()
        
    def update_volatility_surface(self, vol_data: Dict):
        """Update the volatility surface with new data"""
        for (strike, dte), iv in vol_data.items():
            self.vol_surface.add_iv_point(strike, dte, iv)
    
    def price_option(self, spot_price: float, strike: float, days_to_expiry: int,
                    option_type: str, market_iv: Optional[float] = None) -> Dict:
        """
        Price an option with full Greeks calculation
        """
        try:
            T = days_to_expiry / 365.0  # Convert to years
            
            # Use market IV if provided, otherwise get from vol surface
            if market_iv is not None:
                sigma = market_iv
            else:
                sigma = self.vol_surface.get_interpolated_iv(strike, days_to_expiry, spot_price)
                if sigma is None:
                    sigma = 0.25  # Default volatility
            
            # Calculate price
            if option_type.lower() == 'call':
                price = self.bs_model.call_price(spot_price, strike, T, self.risk_free_rate, sigma)
            else:
                price = self.bs_model.put_price(spot_price, strike, T, self.risk_free_rate, sigma)
            
            # Calculate Greeks
            greeks = self.bs_model.calculate_greeks(spot_price, strike, T, self.risk_free_rate, sigma, option_type)
            
            return {
                'price': price,
                'iv': sigma,
                'greeks': greeks,
                'time_to_expiry': T,
                'moneyness': strike / spot_price
            }
            
        except Exception as e:
            logger.error(f"Error pricing option: {e}")
            return {
                'price': 0.01,
                'iv': 0.25,
                'greeks': OptionsGreeks(0, 0, 0, 0, 0),
                'time_to_expiry': days_to_expiry / 365.0,
                'moneyness': strike / spot_price
            }
    
    def calculate_portfolio_risk(self, positions: list, spot_price: float, 
                               price_scenarios: list = None) -> Dict:
        """
        Calculate portfolio risk metrics
        """
        if price_scenarios is None:
            # Default scenarios: -20% to +20% in 5% increments
            price_scenarios = [spot_price * (1 + i/100) for i in range(-20, 25, 5)]
        
        portfolio_greeks = OptionsHedgeCalculator.calculate_portfolio_greeks(positions)
        
        # Calculate P&L for different scenarios
        scenario_pnl = []
        for scenario_price in price_scenarios:
            pnl = 0
            for position in positions:
                # Simplified P&L calculation
                current_greeks = position.get('greeks')
                if current_greeks:
                    price_change = scenario_price - spot_price
                    delta_pnl = current_greeks.delta * price_change * position.get('quantity', 0) * 100
                    gamma_pnl = 0.5 * current_greeks.gamma * (price_change ** 2) * position.get('quantity', 0) * 100
                    pnl += delta_pnl + gamma_pnl
            
            scenario_pnl.append(pnl)
        
        return {
            'portfolio_greeks': portfolio_greeks,
            'price_scenarios': price_scenarios,
            'scenario_pnl': scenario_pnl,
            'max_loss': min(scenario_pnl),
            'max_gain': max(scenario_pnl),
            'delta_hedge_ratio': OptionsHedgeCalculator.calculate_delta_hedge_ratio(positions, spot_price)
        }


class HistoricalVolatilityCalculator:
    """
    Calculate historical volatility metrics
    """
    
    @staticmethod
    def calculate_realized_vol(prices: pd.Series, window: int = 30) -> float:
        """Calculate realized volatility"""
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
    
    @staticmethod
    def calculate_garch_vol(prices: pd.Series) -> float:
        """Calculate GARCH volatility (simplified)"""
        returns = np.log(prices / prices.shift(1)).dropna()
        # Simplified GARCH(1,1) - in practice, use arch library
        alpha = 0.1
        beta = 0.85
        
        variance = returns.var()
        for ret in returns[-30:]:  # Use last 30 observations
            variance = alpha * ret**2 + beta * variance
        
        return np.sqrt(variance * 252)
    
    @staticmethod
    def calculate_vol_cone(prices: pd.Series, windows: list = None) -> Dict[int, float]:
        """Calculate volatility cone"""
        if windows is None:
            windows = [10, 20, 30, 60, 90, 120]
        
        returns = np.log(prices / prices.shift(1)).dropna()
        vol_cone = {}
        
        for window in windows:
            if len(returns) >= window:
                vol = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
                vol_cone[window] = vol
        
        return vol_cone


# Example usage
async def test_options_pricing():
    """Test the options pricing module"""
    
    # Initialize pricer
    pricer = AdvancedOptionsPricer(risk_free_rate=0.03)
    
    # Example: Price AAPL options
    spot_price = 150.0
    strike = 155.0
    days_to_expiry = 30
    
    # Price call option
    call_result = pricer.price_option(spot_price, strike, days_to_expiry, 'call')
    print(f"Call Price: ${call_result['price']:.2f}")
    print(f"Call Delta: {call_result['greeks'].delta:.3f}")
    print(f"Call Gamma: {call_result['greeks'].gamma:.3f}")
    print(f"Call Theta: {call_result['greeks'].theta:.3f}")
    
    # Price put option
    put_result = pricer.price_option(spot_price, strike, days_to_expiry, 'put')
    print(f"Put Price: ${put_result['price']:.2f}")
    print(f"Put Delta: {put_result['greeks'].delta:.3f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_options_pricing())
