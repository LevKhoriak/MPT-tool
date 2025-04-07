# This constructor will only consider two risky asset case

from time_series_cleaner import clean_series

# To read the prices
import pandas as pd

# To calc mean, variance and covariance
from numpy import cov, mean

# To plot stuff
import seaborn as sns

import matplotlib.pyplot as plt

# Read prices of the first and the second asset
(a1, a2) = clean_series('./lkoh.csv', './rus47.csv')

a1["Return"] = a1["Close"].div(a1["Close"].shift(1)) - 1
a2["Return"] = a2["Close"].div(a2["Close"].shift(1)) - 1

a1 = a1.dropna()
a2 = a2.dropna()

er_1 = mean(a1["Return"]) # Mean return

er_2 = mean(a2["Return"]) # Mean return

(var_1, var_2, asset_covariance) = (cov(a1["Return"], a2["Return"])[0][0],
                                    cov(a1["Return"], a2["Return"])[1][1],
                                    cov(a1["Return"], a2["Return"])[0][1])

print(er_1, var_1)
print(er_2, var_2)

def var_p(w1):
    return w1**2 * var_1 + (1 - w1)**2 * var_2 + 2 * w1 * (1 - w1) * asset_covariance

# print('lkoh:', er_1, var_1)
# print('rus47:', er_2, var_2)

def ret_p(w1):
    return w1 * er_1 + (1 - w1) * er_2

# print(corrcoef(lkoh["Close"], lsrg["Close"]))

returns = []
risks = []
w_1 = []

for w in range(0, 101):
    w_1.append(w/100)
    returns.append(ret_p(w/100))
    risks.append(var_p(w/100)**0.5)

EF = pd.DataFrame(data={'w_1': w_1, 'returns': returns, 'risks': risks})
print(EF)

sns.scatterplot(data=EF, x='risks', y='returns')
plt.show()