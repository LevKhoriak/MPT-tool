import pandas as pd
import numpy as np
import scipy.optimize
import seaborn as sns
import matplotlib.pyplot as plt

def mpt_optimizer(assets_files, year, month, day, required_return):

    
    assets_n = len(assets_files)
    tickers = []

    assets = pd.DataFrame()

    # Read close price data, add to database
    for asset_df in assets_files:
        assets = pd.concat([assets, asset_df])

    # Compute daily returns
    assets['Returns'] = assets.groupby('Ticker')['Close'].transform(lambda x: x / x.shift(1) - 1)
    assets.dropna(axis=0, inplace=True)

    window_size = 10

    # Pivot for later use of covariance and expected return
    returns_pivot = assets.pivot(index='Date', columns='Ticker', values='Returns')

    # Compute rolling covariance matrix and expected returns
    returns_pivot = returns_pivot.sort_index()
    covm = returns_pivot.rolling(window=window_size).cov()
    expected_returns = returns_pivot.rolling(window=window_size).mean()
    covm.dropna(axis=0, inplace=True)
    expected_returns.dropna(axis=0, inplace=True)

    Date = year + '-' + month + '-' + day

    # Find data for the required date
    Rmatrix = expected_returns.loc[Date].values
    Covmatrix = covm.loc[Date].values

    # Weighted average return using matrix multiplication
    def gen_return(W, R=Rmatrix):
        return np.matmul(W, R)

    # Total portfolio standard deviation = sqrt(W * Cov * W^T)
    def gen_std(W, Cov=Covmatrix):
        return np.sqrt(np.matmul(np.matmul(W, Cov), W.transpose()))

    # Initial weights (0.2 chosen randomly)
    W0 = np.array([0.2] * assets_n)
    bounds = [(0, 1) for _ in range(assets_n)]

    # Sum of weights should be equal to one
    def sum_of_weights(W):
        h = np.sum(W) - 1

        return h

    constraints = ({'type': 'eq', 'fun': sum_of_weights})

    # Minimum risk portfolio without return constraint
    global_minimum = scipy.optimize.minimize(fun=gen_std, x0=W0, bounds=bounds, method='SLSQP', constraints=constraints)


    # Random sampling from Dirichlet distribution to get array of random weights 
    W = np.random.dirichlet(np.ones(assets_n), 2000)

    period = 360 / window_size

    # Annualized return should be no less than required return
    def return_constraint(W):
        g = (1 + gen_return(W))**period - 1 - required_return
        return g

    constraints = ({'type': 'eq', 'fun': sum_of_weights}, {'type': 'ineq', 'fun': return_constraint})

    # Minimum risk portfolio with return contraint
    constrained_minimum = scipy.optimize.minimize(fun=gen_std, x0=W0, bounds=bounds, method='SLSQP', constraints=constraints)

    # Generate annualized total portfolio risk for random weights
    V = np.multiply(list(map(gen_std, W)), np.sqrt(period))

    # Generate returns from random weights
    R = list(map(gen_return, W))

    # Annualize returns
    R = np.power(np.add(R, 1), period) - 1

    # Annualize risks and returns of unconstrained portfolio
    global_minimum_std = gen_std(global_minimum.x) * np.sqrt(period)
    global_minimum_ret = (1 + gen_return(global_minimum.x))**period - 1

    # Annualize risks and returns of constrained portfolio
    constrained_minimum_std = gen_std(constrained_minimum.x) * np.sqrt(period)
    constrained_minimum_ret = (1 + gen_return(constrained_minimum.x))**period - 1

    fig, ax = plt.subplots()
    # Plot efficient frontier
    sns.scatterplot(x = V, y = R, ax=ax)
    if global_minimum.success:
        ax.plot(global_minimum_std, global_minimum_ret, 'ro')
    if constrained_minimum.success:
        ax.plot(constrained_minimum_std, constrained_minimum_ret, 'go')
    elif not constrained_minimum.success:
        print('Failed to find optimum portfolio with this required return')
    plt.legend(['possible portfolio', 'least risky', f'least risky with return $\\geq$ {required_return}'])
    plt.title(f'Efficient frontier at {Date}')
    plt.xlabel('Risk (annualized)')
    plt.ylabel('Return (annualized)')
    return fig
