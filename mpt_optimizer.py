import pandas as pd
import numpy as np
import scipy.optimize
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.filedialog as fd

# Input quotes time series and clean the data
root = tk.Tk()
root.withdraw()
assets_filenames = fd.askopenfilenames(parent=root, title='Choose a file')

assets_n = len(assets_filenames)
tickers = []

assets = pd.DataFrame()

begindate, enddate = (pd.Timestamp('1999-01-01'), pd.Timestamp('2025-01-01'))

for afname in assets_filenames:
    adf = pd.read_csv(filepath_or_buffer=afname, delimiter=';', usecols=[0, 2, 4], parse_dates=[1])
    adf.columns = ['Ticker', 'Date', 'Close']
    tickers.append(adf['Ticker'].iloc[0])
    begindate = max(begindate, adf['Date'].min())
    enddate = min(enddate, adf['Date'].max())

quote_range = pd.date_range(begindate, enddate)

for afname in assets_filenames:
    adf = pd.read_csv(filepath_or_buffer=afname, delimiter=';', usecols=[0, 2, 4], parse_dates=[1])
    adf.columns = ['Ticker', 'Date', 'Close']
    adf = adf.loc[(adf['Date'] >= begindate) & (adf['Date'] <= enddate)]
    adf.set_index('Date', inplace=True)
    adf = adf.reindex(quote_range, method='ffill')
    adf.reset_index(names=['Date'], inplace=True)
    assets = pd.concat([assets, adf])


# Compute daily returns
assets['Returns'] = assets.groupby('Ticker')['Close'].transform(lambda x: x / x.shift(1) - 1)
assets.dropna(axis=0, inplace=True)

window_size = 10

returns_pivot = assets.pivot(index='Date', columns='Ticker', values='Returns')

# Compute rolling covariance matrix and expected returns
returns_pivot = returns_pivot.sort_index()
covm = returns_pivot.rolling(window=window_size).cov()
expected_returns = returns_pivot.rolling(window=window_size).mean()

covm.dropna(axis=0, inplace=True)
expected_returns.dropna(axis=0, inplace=True)

print('What is the date of portfolio construction?')
year = input('Enter the year ')
month = input('Enter the month ')
day = input('Enter the day ')
Date = year + '-' + month + '-' + day

Rmatrix = expected_returns.loc[Date].values
Covmatrix = covm.loc[Date].values
Wmatrix = np.array([0.3] * assets_n)

def gen_return(W, R=Rmatrix):
    return np.matmul(W, R)

def gen_std(W, Cov=Covmatrix):
    return np.sqrt(np.matmul(np.matmul(W, Cov), W.transpose()))

def portfolio_performance(W, R, Cov):
    p_ret = gen_return(W, R)
    p_var = gen_std(W, Cov)
    return (p_ret, p_var)


W0 = np.array([0.2] * assets_n)
bounds = [(0, 1) for _ in range(assets_n)]

def sum_of_weights(W):
    h = np.sum(W) - 1

    return h

constraints = ({'type': 'eq', 'fun': sum_of_weights})

global_minimum = scipy.optimize.minimize(fun=gen_std, x0=W0, bounds=bounds, method='SLSQP', constraints=constraints)

W = np.random.dirichlet(np.ones(assets_n), 2000)

required_return = float(input('What is the required return (as a decimal)? '))

period = 360 / window_size

def return_constraint(W):
    g = (1 + gen_return(W))**period - 1 - required_return
    return g

constraints = ({'type': 'eq', 'fun': sum_of_weights}, {'type': 'ineq', 'fun': return_constraint})

constrained_minimum = scipy.optimize.minimize(fun=gen_std, x0=W0, bounds=bounds, method='SLSQP', constraints=constraints)

V = np.multiply(list(map(gen_std, W)), np.sqrt(period))
R = list(map(gen_return, W))
R = np.power(np.add(R, 1), period) - 1

global_minimum_std = gen_std(global_minimum.x) * np.sqrt(period)
global_minimum_ret = (1 + gen_return(global_minimum.x))**period - 1

constrained_minimum_std = gen_std(constrained_minimum.x) * np.sqrt(period)
constrained_minimum_ret = (1 + gen_return(constrained_minimum.x))**period - 1

plt.figure(figsize=(7, 5), dpi=200)
sns.scatterplot(x = V, y = R)
if global_minimum.success:
    plt.plot(global_minimum_std, global_minimum_ret, 'ro')
    print(global_minimum_ret, global_minimum_std)
else:
    print('Global minimum could not be found')
if constrained_minimum.success:
    plt.plot(constrained_minimum_std, constrained_minimum_ret, 'go')
    print(constrained_minimum_ret, constrained_minimum_std)
else:
    print('Constrained minimum could not be found')
plt.legend(['possible portfolio', 'least risky', f'least risky with return $\\geq$ {required_return}'])
plt.title(f'Efficient frontier at {Date}')
plt.xlabel('Risk (annualized)')
plt.ylabel('Return (annualized)')
plt.show()
