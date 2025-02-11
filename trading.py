import numpy as np
import pandas as pd
import scipy.optimize as sco

def mean_reversion(df, window):
    """Calculate the mean reversion signal for a given window."""
    rolling_mean = df.rolling(window).mean()
    rolling_std = df.rolling(window).std()
    zscore = (df - rolling_mean) / (rolling_std + 1e-8)  # Adding small epsilon to avoid zero division
    return -zscore

def momentum(df, window):
    """Calculate the volatility-adjusted momentum signal for a given window."""
    returns = df.pct_change()
    rolling_volatility = returns.rolling(window).std()
    momentum_signal = returns.rolling(window).sum() / (rolling_volatility + 1e-8)  # Prevent division by zero
    return momentum_signal

def combine_signals(signals, weights=None):
    """Combine multiple trading signals with optional weights."""
    if weights is None:
        weights = np.ones(len(signals)) / len(signals)
    combined_signal = sum(weight * signal for weight, signal in zip(weights, signals))
    return np.sign(combined_signal)

def backtest_portfolio(returns, positions, transaction_cost, slippage=0.001):
    """Backtest a portfolio of positions with slippage and more realistic transaction costs."""
    # Calculate slippage-adjusted returns
    adjusted_returns = returns - slippage * np.abs(positions - positions.shift(1))

    # Calculate the portfolio returns with transaction cost
    portfolio_returns = (positions.shift(1) * returns) - (np.abs(positions - positions.shift(1)) * transaction_cost)
    
    # Calculate the cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Calculate the drawdowns
    previous_peaks = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - previous_peaks) / previous_peaks

    # Calculate the statistics
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    max_drawdown = drawdowns.min()

    return cumulative_returns, sharpe_ratio, max_drawdown

def optimize_portfolio(returns, max_weight=0.2):
    """Optimize the portfolio weights to maximize the Sharpe ratio with constraints on asset weights."""
    def objective_function(weights, returns):
        portfolio_returns = np.dot(returns, weights)
        sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        return -sharpe_ratio

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, max_weight) for _ in range(len(returns.columns))]

    # Initialize weights
    weights = np.ones(len(returns.columns)) / len(returns.columns)

    # Optimize weights
    optimized_weights = sco.minimize(objective_function, weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)

    return optimized_weights.x

def rebalance_portfolio(signals, frequency=20):
    """Rebalance the portfolio based on the signals at a specified frequency."""
    rebalance_positions = signals.copy()
    rebalance_positions[::frequency] = signals[::frequency]  # Rebalance at the specified frequency
    return rebalance_positions
