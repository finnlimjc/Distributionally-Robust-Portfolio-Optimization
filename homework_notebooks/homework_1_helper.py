import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt

class PerformanceMetrics:
    def __init__(self, returns_series:pd.Series, periods:int=252, is_log_returns:bool=True, is_pct_dd:bool=True):
        self.returns_series = returns_series if is_log_returns else np.log1p(returns_series)
        self.periods = periods
        self.is_pct_dd = is_pct_dd
    
    def annualized_mean(self) -> float:
        mu = np.expm1(self.returns_series.mean()*self.periods)
        return mu
    
    def annualized_vol(self) -> float:
        return self.returns_series.std(ddof=1)* np.sqrt(self.periods)
    
    def sharpe_ratio(self, mu:float, sigma:float, rf_rate:float=0):
        if sigma == 0:
            return np.nan
        return (mu-rf_rate)/sigma
    
    def drawdown_series(self) -> pd.Series:
        cum_return = self.returns_series.cumsum()
        hwm = cum_return.cummax().clip(lower=0)
        self.dd_series = (cum_return - hwm)
        
        if self.is_pct_dd:
            self.dd_series = np.expm1(self.dd_series)
        
        return self.dd_series
    
    def mdd(self) -> float:
        if not hasattr(self, 'dd_series'):
            self.drawdown_series()
        return self.dd_series.min()
    
    def pipeline(self, rf_rate:float=0) -> dict[str:float]:
        mu = self.annualized_mean()
        vol = self.annualized_vol()
        sr = self.sharpe_ratio(mu, vol, rf_rate)
        mdd = self.mdd()
        
        metrics = {
            'cagr': mu,
            'annualized_vol': vol,
            'sharpe_ratio': sr,
            'max_draw_down': mdd
        }
        return metrics

class Gmvp:
    def __init__(self, returns_data, risk_free_rate=0.0, window=252, rebalance_freq=21):
        self.returns = returns_data
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)
        self.rf = risk_free_rate
        self.window = window
        self.rebalance_freq = rebalance_freq
        self.weights_history = []
        self.portfolio_returns = []
    
    def portfolio_return(self, weights):
        return np.dot(weights, self.mean_returns.values)
    
    def optimise_min_var(self, cov_matrix):
        def var_objective(weights, cov_matrix):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1.0 / self.n_assets] * self.n_assets)
        
        result = optimize.minimize(
            var_objective,
            initial_guess,
            args=(cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'ftol': 1e-9, 'maxiter': 1000}
        )
        return result.x if result.success else initial_guess
    
    def run_rolling_optimisation(self):
        returns = self.returns
        n = len(returns)
        dates = returns.index
        weights_list = []
        port_ret_list = []
        for start in range(0, n - self.window, self.rebalance_freq):
            end = start + self.window
            train = returns.iloc[start:end]
            test = returns.iloc[end:end+self.rebalance_freq]
            cov_mat = train.cov().values
            weights = self.optimise_min_var(cov_mat)
            weights_list.append(weights)
            if not test.empty:
                port_ret = test.values @ weights
                port_ret_list.extend(port_ret)
        self.weights_history = weights_list
        self.portfolio_returns = pd.Series(port_ret_list, index=dates[self.window:self.window+len(port_ret_list)])

    def report_performance(self):
        pm = PerformanceMetrics(pd.Series(self.portfolio_returns), periods=252, is_log_returns=False)
        metrics = pm.pipeline()
        print("Max Sharpe Portfolio Out-of-Sample Performance:")
        print(f"CAGR:            {metrics['cagr']:.4f}")
        print(f"Annual Volatility:{metrics['annualized_vol']:.4f}")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown:    {metrics['max_draw_down']:.4f}")
        return metrics


class MaxSharpePortfolio:
    def __init__(self, returns_data, risk_free_rate=0.0, window=252, rebalance_freq=21):
        """
        returns_data: pd.DataFrame of asset returns (daily)
        risk_free_rate: daily risk-free rate (default 0)
        window: rolling window size (default 252 trading days)
        rebalance_freq: rebalance every N days (default monthly)
        """
        self.returns = returns_data
        self.rf = risk_free_rate
        self.window = window
        self.rebalance_freq = rebalance_freq
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)
        self.weights_history = []
        self.portfolio_returns = []

    def _sharpe_objective(self, weights, mean_returns, cov_matrix):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        # Negative Sharpe for minimization
        return -(port_return - self.rf) / port_vol if port_vol > 0 else 1e6

    def _optimize_weights(self, mean_returns, cov_matrix):
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1.0 / self.n_assets] * self.n_assets)
        result = optimize.minimize(
            self._sharpe_objective,
            initial_guess,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False, 'ftol': 1e-9, 'maxiter': 1000}
        )
        return result.x if result.success else initial_guess

    def run_rolling_optimization(self):
        returns = self.returns
        n = len(returns)
        dates = returns.index
        weights_list = []
        port_ret_list = []
        for start in range(0, n - self.window, self.rebalance_freq):
            end = start + self.window
            train = returns.iloc[start:end]
            test = returns.iloc[end:end+self.rebalance_freq]
            mean_ret = train.mean().values
            cov_mat = train.cov().values
            weights = self._optimize_weights(mean_ret, cov_mat)
            weights_list.append(weights)
            # Out-of-sample returns
            if not test.empty:
                port_ret = test.values @ weights
                port_ret_list.extend(port_ret)
        self.weights_history = weights_list
        self.portfolio_returns = pd.Series(port_ret_list, index=dates[self.window:self.window+len(port_ret_list)])
    
    def report_performance(self):
        pm = PerformanceMetrics(pd.Series(self.portfolio_returns), periods=252, is_log_returns=False)
        metrics = pm.pipeline()
        print("Max Sharpe Portfolio Out-of-Sample Performance:")
        print(f"CAGR:            {metrics['cagr']:.4f}")
        print(f"Annual Volatility:{metrics['annualized_vol']:.4f}")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown:    {metrics['max_draw_down']:.4f}")
        return metrics

class MaximumReturn:
    def __init__(self, returns_data, risk_free_rate=0.0, window=252, rebalance_freq=21):
        self.returns = returns_data
        self.rf = risk_free_rate
        self.window = window
        self.rebalance_freq = rebalance_freq
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)
        self.weights_history = []
        self.portfolio_returns = []
    
    def _assign_weights(self, return_df:pd.DataFrame) -> np.ndarray:
        mu_ret = return_df.mean().to_numpy()
        weights = np.zeros(self.n_assets)
        idx = np.argmax(mu_ret)
        weights[idx] = 1
        return weights
    
    def run_rolling_optimization(self):
        returns = self.returns
        n = len(returns)
        dates = returns.index
        weights_list = []
        port_ret_list = []
        for start in range(0, n - self.window, self.rebalance_freq):
            end = start + self.window
            train = returns.iloc[start:end]
            test = returns.iloc[end:end+self.rebalance_freq]
            weights = self._assign_weights(train)
            weights_list.append(weights)
            # Out-of-sample returns
            if not test.empty:
                port_ret = test.values @ weights
                port_ret_list.extend(port_ret)
        self.weights_history = weights_list
        self.portfolio_returns = pd.Series(port_ret_list, index=dates[self.window:self.window+len(port_ret_list)])
    
    def report_performance(self) -> dict[str:float]:
        pm = PerformanceMetrics(pd.Series(self.portfolio_returns), periods=252, is_log_returns=True)
        metrics = pm.pipeline()
        print("Max Sharpe Portfolio Out-of-Sample Performance:")
        print(f"CAGR:            {metrics['cagr']:.4f}")
        print(f"Annual Volatility:{metrics['annualized_vol']:.4f}")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown:    {metrics['max_draw_down']:.4f}")
        return metrics