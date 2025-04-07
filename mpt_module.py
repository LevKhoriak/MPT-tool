import pandas as pd
import numpy as np
import scipy.optimize
import seaborn as sns
import matplotlib.pyplot as plt

# Weighted average return using matrix multiplication
def gen_return_formula(R):
    def gen_return(W):
        return np.matmul(W, R)
    return gen_return

# Total portfolio standard deviation = sqrt(W * Cov * W^T)
def gen_std_formula(Cov):
    def gen_std(W):
        return np.sqrt(np.matmul(np.matmul(W, Cov), W.transpose()))
    return gen_std

# Sum of weights should be equal to one
def sum_of_weights(W):
    return np.sum(W) - 1

# Annualized return should be no less than required return
def return_constraint_formula(gen_return, period, required_return):
    def return_constraint(W):
        return (1 + gen_return(W))**period - 1 - required_return
    return return_constraint


def mpt_optimizer(assets_files, date, required_return, window_size):

    asset_n = len(assets_files)

    # Read close price data, add to database
    assets = pd.concat(assets_files)

    # Compute daily returns
    assets['Returns'] = assets.groupby('Ticker')['Close'].transform(lambda x: x / x.shift(1) - 1)
    assets.dropna(axis=0, inplace=True)

    # Pivot for later use of covariance and expected return
    returns_pivot = assets.pivot(index='Date', columns='Ticker', values='Returns')
    returns_pivot = returns_pivot.sort_index()

    # Compute rolling covariance matrix and expected returns
    covm = returns_pivot.rolling(window=window_size).cov()
    covm.dropna(axis=0, inplace=True)

    expected_returns = returns_pivot.rolling(window=window_size).mean()
    expected_returns.dropna(axis=0, inplace=True)

    # Rmatrix is the matrix of returns and Covmatrix is the covariance matrices
    Rmatrix = expected_returns.loc[date].values
    Covmatrix = covm.loc[date].values    

    # Initial weights (0.2 chosen randomly)
    W0 = np.array([0.2] * asset_n)
    bounds = [(0, 1) for _ in range(asset_n)]

    # Random sampling from Dirichlet distribution to get array of random weights 
    W = np.random.dirichlet(np.ones(asset_n), 2000)

    gen_return = gen_return_formula(Rmatrix)
    gen_std = gen_std_formula(Covmatrix)

    period = 360 / window_size
    return_constraint = return_constraint_formula(gen_return, period, required_return)

    def find_minvar_portfolio(constraints):
        return scipy.optimize.minimize(fun=gen_std, x0=W0, bounds=bounds, method='SLSQP', constraints=constraints)

    # Minimum risk portfolio without return constraint
    global_minimum = find_minvar_portfolio(
        ({'type': 'eq', 'fun': sum_of_weights})
    )

    # Minimum risk portfolio with return contraint
    constrained_minimum = find_minvar_portfolio(
        ({'type': 'eq', 'fun': sum_of_weights}, {'type': 'ineq', 'fun': return_constraint})
    )

    # Generate annualized total portfolio risk for random weights
    V = np.multiply(list(map(gen_std, W)), np.sqrt(period))

    # Generate returns from random weights
    R = list(map(gen_return, W))

    # Annualize returns
    R = np.power(np.add(R, 1), period) - 1

    def annualize_risk_ret(W):
        min_std_pa = gen_std(W) * np.sqrt(period)
        min_ret_pa = (1 + gen_return(W))**period - 1
        return (min_std_pa, min_ret_pa)

    # Annualize risks and returns of unconstrained portfolio
    (global_minimum_std, global_minimum_ret) = annualize_risk_ret(global_minimum.x)

    # Annualize risks and returns of constrained portfolio
    (constrained_minimum_std, constrained_minimum_ret) = annualize_risk_ret(constrained_minimum.x)

    fig, ax = plt.subplots()

    # Plot efficient frontier
    sns.scatterplot(x = V, y = R, ax=ax)
    if global_minimum.success:
        ax.plot(global_minimum_std, global_minimum_ret, 'ro')
    if constrained_minimum.success:
        ax.plot(constrained_minimum_std, constrained_minimum_ret, 'go')
    else:
        print('Failed to find optimum portfolio with this required return')

    plt.legend(['possible portfolio', 'least risky', f'least risky with return $\\geq$ {required_return:.2f}'])
    plt.title(f'Efficient frontier at {date}')
    plt.xlabel('Risk (annualized)')
    plt.ylabel('Return (annualized)')
    return fig
