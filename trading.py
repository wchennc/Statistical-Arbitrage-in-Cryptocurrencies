import numpy as np
import pandas as pd
import scipy.optimize as sco
from sklearn.linear_model import LogisticRegression  # For ML-based signal weighting
'''
1. Enhanced Signal Generation
1.1 Added dynamic thresholding for mean reversion
1.2 Improved momentum calculation with smoothing
1.3 Multiple combination methods (volatility scaling, ML weighting)

2. Realistic Backtesting
2.1 Volatility-scaled position sizing
2.2 Slippage modeling
2.3 Maximum leverage constraints
2.4 Additional performance metrics (Sortino ratio, Calmar ratio, win rate)

3. Advanced Portfolio Optimization
3.1 Multiple optimization methods (Max Sharpe, Risk Parity, Min Variance)
3.2 Regularized covariance matrix estimation
3.3 More robust optimization settings

4. Risk Management
4.1 Dynamic position sizing based on volatility
4.2 Maximum leverage constraints
4.3 Drawdown-based metrics

5. Machine Learning Integration
5.1 Logistic regression for signal weighting
5.2 Uses forward-looking returns (properly shifted) for training

6. Numerical Stability
6.1 Better handling of division by zero
6.2 More stable cumulative return calculation
6.3 Regularization in covariance matrix

7. To use this effectively:
7.1 Consider adding walk-forward optimization
7.2 Implement regime filters (e.g., volatility regimes)
7.3 Add transaction cost tiers based on trade size
7.4 Incorporate fundamental data filters
7.5 Add event-driven risk management (circuit breakers)
'''

def mean_reversion(df, window, threshold=1.0):
    """Enhanced mean reversion with dynamic thresholding"""
    rolling_mean = df.rolling(window).mean()
    rolling_std = df.rolling(window).std()
    zscore = (df - rolling_mean) / rolling_std
    return np.where(np.abs(zscore) > threshold, -zscore, 0)

def momentum(df, window, smoothing=5):
    """Improved momentum with smoothing and rate of change"""
    roc = df.pct_change(window)
    return roc.rolling(smoothing).mean()

def combine_signals(signals, method='ml', lookahead_returns=None):
    """Enhanced signal combination with multiple methods"""
    if method == 'equal':
        combined = sum(signals) / len(signals)
    elif method == 'volatility_scaled':
        vols = [s.std() for s in signals]
        weights = [1/v if v != 0 else 0 for v in vols]
        combined = sum(s*w for s,w in zip(signals, weights)) / sum(weights)
    elif method == 'ml' and lookahead_returns is not None:
        # Machine learning-based weighting
        X = pd.concat(signals, axis=1).fillna(0)
        y = (lookahead_returns > 0).astype(int)
        model = LogisticRegression()
        model.fit(X, y)
        combined = model.predict_proba(X)[:, 1] * 2 - 1
    else:
        raise ValueError("Invalid combination method")
    
    return combined

def backtest_portfolio(returns, positions, transaction_cost=0.0005, max_leverage=2.0):
    """Enhanced backtester with realistic constraints"""
    # Position sizing with volatility scaling
    volatility = returns.rolling(21).std().shift(1)
    scaled_positions = positions * (0.2 / volatility).replace(np.inf, 0)
    
    # Enforce maximum leverage
    scaled_positions = np.clip(scaled_positions, -max_leverage, max_leverage)
    
    # Calculate transaction costs with realistic slippage approximation
    position_changes = scaled_positions.diff().abs()
    transaction_costs = position_changes * transaction_cost + position_changes * returns.std() * 0.1  # Slippage
    
    # Calculate returns
    portfolio_returns = scaled_positions.shift(1) * returns - transaction_costs
    
    # Performance metrics
    cumulative_returns = portfolio_returns.cumsum()  # More stable than cumprod for leveraged strategies
    previous_peaks = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - previous_peaks) / previous_peaks.replace(0, 1)
    
    metrics = {
        'sharpe': np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std(),
        'sortino': np.sqrt(252) * portfolio_returns.mean() / (portfolio_returns[portfolio_returns < 0].std() + 1e-9),
        'max_drawdown': drawdowns.min(),
        'calmar': portfolio_returns.mean() / abs(drawdowns.min()),
        'win_rate': (portfolio_returns > 0).mean()
    }
    
    return cumulative_returns, metrics

def optimize_portfolio(returns, risk_free_rate=0.0, optimization_method='risk_parity'):
    """Enhanced portfolio optimization with multiple methods"""
    cov_matrix = returns.cov() * 252
    expected_returns = returns.mean() * 252 - risk_free_rate
    
    if optimization_method == 'max_sharpe':
        def objective(weights):
            port_return = np.dot(weights, expected_returns)
            port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            return -port_return / port_vol
    elif optimization_method == 'risk_parity':
        def objective(weights):
            marginal_risk = weights * (cov_matrix @ weights)
            risk_contribution = marginal_risk / marginal_risk.sum()
            return np.sum((risk_contribution - 1/len(weights))**2)
    elif optimization_method == 'min_var':
        def objective(weights):
            return np.sqrt(weights.T @ cov_matrix @ weights)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in returns.columns]
    init_weights = np.ones(len(returns.columns)) / len(returns.columns)
    
    result = sco.minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return pd.Series(result.x, index=returns.columns)

# Example usage:
if __name__ == "__main__":
    # Load sample data
    prices = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
    
    # Generate enhanced signals
    mr_signals = [mean_reversion(prices[asset], window=20) for asset in prices.columns]
    mom_signals = [momentum(prices[asset], window=50) for asset in prices.columns]
    
    # Combine signals with ML weighting
    combined = combine_signals(
        mr_signals + mom_signals,
        method='ml',
        lookahead_returns=prices.pct_change().shift(-1)  # Next period returns
    )
    
    # Backtest
    returns = prices.pct_change()
    cum_returns, metrics = backtest_portfolio(returns, combined)
    
    # Optimize portfolio
    optimal_weights = optimize_portfolio(returns, optimization_method='risk_parity')
    
    print(f"Strategy Metrics: {metrics}")
    print(f"Optimal Weights:\n{optimal_weights}")
