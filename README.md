# Distributionally Robust Portfolio Optimization
In this project, the goal was to apply what we learned in class by replicating and extending on the following paper: "Distributionally robust deep Q-learning". This repository is a different version from what was submitted. If you wish to explore the original repository, you may find it here: https://github.com/keithgzx/RL-for-QF

# Instructions
1. Create a virtual environment:
```sh
# Open a terminal and navigate to your project folder
cd myproject

# Create the .venv folder
python -m venv .venv
```

2. Activate the virtual environment:
```sh
# Windows command prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS and Linux
source .venv/bin/activate
```

3. Install packages in the environment:
```sh
python -m pip install -r requirements.txt
```

4. Run the dashboard:
```sh
streamlit run main.py
```

5. Alternatively, if you have already completed the virtual environment and packages installation, run the appropriate commands as the following example:
```sh
cd myproject
.venv\Scripts\activate
python -m streamlit run main.py
```

# Key Extension
In the paper, the authors trained a generative Long Short-Term Memory (LSTM) model on the S&P 500 data using the Maximum Mean Discrepancy (MMD) with a signature kernel, while employing a Moving Average (MA) model to reproduce the volatility clustering property. However, we argue that a simpler model can be employed to achieve comparable results, reducing the complexity of the algorithm and enabling easier transferability to other assets rather than retraining and retuning hyperparameters. Moreover, the goal of being distributionally robust was to assume the worst-case transition from a ball around a reference probability measure, quantified by the Sinkhorn distance. This addresses model uncertainty as we are optimizing against the worst-case distribution.

## Stationary Block Bootstrapping
We need a method that preserves empirical volatility structures when simulating alternative market paths. Moreover, as the original intention was to capture the underlying returns distribution, the chosen method must also converge in distribution. Of which, we proposed using the Stationary Block Bookstrapping (SBB) method. The SBB algorithm has its limitations. There must be weak dependence, i.e. the dependence between observations decreases sufficiently fast as the lag increases, which implies that it fails to capture any long-term dependence structure. Moreover, since SBB samples from the underlying process, the observed data must approximate the true distribution of the process. Lastly, the results will vary with the average block size (L), and it is assumed that L will grow with the length of the observed returns, but at a slower rate to ensure that resampled paths are not dominated by a single set of blocks.

### Convergence in Distribution
<img width="423" height="273" alt="image" src="https://github.com/user-attachments/assets/93506c37-0055-4445-b0fc-f6e37a38c1d8" />

To assess whether SBB successfully reproduces the distribution of the underlying data, weapply the two-sample Kolmogorov–Smirnov (KS) test across bootstrap replications of varying lengths. Using time horizons increasing in increments of 256 days (i.e., 256, 512,…, up to the full dataset length), we ran 1,000 simulations at each horizon. Then, we ran the two-sample KS test between the bootstrap-simulated sample and the underlying data, recording the number of simulations where we reject the null at the α = 0. 05 level. 

### Preservation of Volatility Clustering Property
<img width="583" height="352" alt="image" src="https://github.com/user-attachments/assets/39e31e99-c1cc-44ed-85f9-4a8f4701af87" />

The next problem is to determine an appropriate block length. The autocorrelation structure of squared returns was analyzed up to a maximum lag of 50 𝐿 ∈ {50, 100, 200 , 400}. From all simulated Auto-Correlation Functions (ACF), we constructed pointwise (1-α)100% confidence interval bands and the bootstrap mean ACF. Then, through visual diagnostics, we will select the best average block size to preserve the volatility clustering property of the underlying. Suppose the underlying curve shows a positive, slowly decaying correlation at short lags and largely falls within the confidence interval. In that case, it indicates that the bootstrap simulations preserve the clustering strength observed in the data.

For more discussion on this matter, one can refer to ```./notebooks/bootstrap_env_sim.ipynb```.

# Out of Sample Performance

## Drawdown
<img width="503" height="284" alt="image" src="https://github.com/user-attachments/assets/0561768d-5c9d-48e3-bfd4-4bdd3da80cf2" />

The agent outperformed the benchmark buy-and-hold strategy on the S&P 500 Index (^SPX) from 1 Jan 1995 to 31 Oct 2025 (assumed transaction cost of 0.05% and a risk-free rate of 2.4%). In particular, the agent achieved higher cumulative returns while maintaining lower overall volatility and a controlled drawdown profile. These factors are advantageous from an emotional perspective as they mitigate investor anxiety and reduce the risk of mass withdrawals.

## Returns Distribution
<img width="322" height="261" alt="image" src="https://github.com/user-attachments/assets/68b43ef1-38b6-4454-980d-510ad564c392" />

Despite a relatively modest daily win rate of 41.7%, the return distribution reveals that the agent’s profitability stems from occasional high-impact trades. The elevated kurtosis of 29.6, compared to 10.2 for the buy-and-hold strategy, indicates that the agent experiences more extreme return events. From a strategic standpoint, the agent exhibits traits consistent with a momentum-oriented trading approach.

## Cumulative Wealth and Positions
<img width="619" height="409" alt="image" src="https://github.com/user-attachments/assets/dcb85ee5-60b0-4609-9310-8b957c861654" />

Based on the positions over time, we observed a noticeable bias towards long positions. When the agent is bullish, it invests all capital into the risky asset. However, when the agent is bearish, it takes a conservative position and hovers around a 25% short position. Of note, there is an exception during the aftermath of 2008. The agent took aggressive short positions, leading to the first point of divergence between the two strategies. Meanwhile, the second divergence occurred during the COVID-19 period. The agent avoided major drawdowns during the crash and capitalized on the momentum-driven rally that followed, leading to a steep rise in cumulative returns. Overall, the agent successfully fulfilled the objective of assuming the worst-case scenario and minimizing drawdowns while tracking the underlying.

## Limitation
<img width="900" height="400" alt="image" src="https://github.com/user-attachments/assets/ab25683b-bddc-44aa-9bc5-2b7f63dbe246" />

After using the same agent in four other tickers (GDX, QQQ, BTC-USD, and KXI), we note that outperformance is not universally guaranteed, and the drawdown-minimization property is imperfect. For GDX, we observed that during COVID-19, the agent’s drawdown-minimization property led to an unfavourable position, causing a sharp drop in cumulative wealth and missing out on the subsequent recovery. Nonetheless, the agent still outperformed the buy-and-hold strategy.

In contrast, the agent performed poorly in situations like BTC-USD, characterized by prolonged periods of stagnation followed by explosive rallies. Other examples include TSLA and AAPL. Given the agent’s momentum-based strategy, we infer that the agent likely took numerous small loss-making trades while anticipating momentum breakouts. When the momentum finally arrives after a few years, the agent has insufficient capital to participate in the rally. This behaviour is an inherent weakness in momentum strategies.

# Future Work
Despite the promising results, there is much potential for future research to develop the idea. Bayesian optimisation could be applied to configure key hyperparameters, such as ε and δ values. However, careful implementation is essential to ensure that the agent generalizes beyond the training regime and does not overfit. Additionally, further analysis on the model’s investable universe would expose the limitations of the model and uncover the characteristics of environments where it performs optimally. This would involve examining how the model reacts across different asset classes and under varying market regimes.

Lastly, we could test different training data by changing the underlying distribution to understand how the optimal policy changes. The testing and incorporation of new features could also prove useful to improve the agent’s decision-making capabilities. We believe that augmenting the environment and conducting further experiments on its adaptability across diverse market conditions will further refine the model’s learning process and improve its consistency in generating sustainable, risk-adjusted returns over time.

# References
Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and practice (2nd ed.). OTexts. https://otexts.com/fpp2/autocorrelation.html

Lu, C. I., & Sester, J. (2024). Generative model for financial time series trained with MMD using a signature kernel. arXiv. https://arxiv.org/abs/2407.19848

Lu, C. I., Sester, J., & Zhang, A. (2025). Distributionally robust deep Q-learning. arXiv. https://arxiv.org/abs/2505.19058

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. University of Wisconsin – Social Science Computing Cooperative. https://www.ssc.wisc.edu/~bhansen/718/Politis%20Romano.pdf

Zaiontz, C. (n.d.). Two-sample Kolmogorov-Smirnov test. Real Statistics. https://real-statistics.com/non-parametric-tests/goodness-of-fit-tests/two-sample-kolm ogorov-smirnov-test/
