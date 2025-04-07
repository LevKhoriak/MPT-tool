import pandas as pd

def read_mfd(filepath):
    df = pd.read_csv(filepath, delimiter=';', index_col=[2], parse_dates=True)
    df = df.drop(['<TICKER>', '<PER>', '<TIME>', '<VOL>'], axis=1).rename(columns={'<CLOSE>': 'Close'})
    df.index.names = ['Date']
    df.sort_index(inplace=True)
    return df

def clean_series(filepath1, filepath2):
    asset1 = read_mfd(filepath1)
    asset2 = read_mfd(filepath2)

    start_date = max(asset1.index[0], asset2.index[0])
    end_date = min(asset1.index[-1], asset2.index[-1])

    clean_asset1 = asset1.loc[start_date : end_date]
    clean_asset2 = asset2.loc[start_date : end_date]

    return (clean_asset1, clean_asset2)

(ca1, ca2) = clean_series('./lkoh.csv', './rus47.csv')

# print(ca1, ca2)