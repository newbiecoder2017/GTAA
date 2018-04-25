#This model trades based on the 10 Month SMA rule rotating through different asset classes.
#universe consittuents
#DBC - DB Liquid Commoties Index etf
#GLD - Gold
#SPY  - U.S. Stocks (Fama French top 30% by market capitalization)
#IEV- European Stocks (Stoxx 350 Index)
#EWJ - Japanese Stocks (MSCI Japan)
#EEM - Emerging Market Stocks (MSCI EM)
#IYR - U.S. REITs (Dow Jones U.S. Real Estate Index)
#RWX - International REITs (Dow Jones Int’l Real Estate Index)
#IEF - Intermediate Treasuries (Barclays 7-10 Year Treasury Index)
#TLT - Long Treasuries (Barclays 20+ Year Treasury Index)
#BIL - 1-3 Month T Bill
#SHY - Barclays Capital U.S. 1-3 Year Treasury Bond Index

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
yf.pdr_override() # <== that's all it takes :-)



#Function to pull data from yahoo
def pull_data(s):
    return pdr.get_data_yahoo(s, start="2000-12-31", end="2018-04-23")['Adj Close']

def read_price_file(frq = 'BM'):
    df_price = pd.read_csv("C:/Python27/Git/SMA_GTAA/adj_close.csv", index_col='Date', parse_dates=True)
    df_price = df_price.resample(frq, closed='right').last()
    return df_price

def ma_signal(data, window):

    trading_universe = ['DBC', 'GLD', 'SPY', 'IEV', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT']
    RollMa = data.rolling(window).mean()
    RollMa = RollMa[trading_universe]['5/30/2007':]
    data_trading = data[trading_universe]['5/30/2007':]
    returns_df = data_trading.pct_change().shift(1)
    #buy rule: if px(t) > 10M SMA(t)
    signal_df = data_trading > RollMa
    return signal_df[:-1]

def equal_weight_portfolio(px_data, signal):

    trading_universe = signal.columns.tolist()
    data_trading = px_data[trading_universe]['5/30/2007':]
    returns_df = data_trading.pct_change().shift(1)
    cash_ret = px_data['BIL']['5/30/2007':].pct_change().shift(1)[:-1]
    holdings_returns = returns_df[signal][:-1]
    pos_wt = (len(trading_universe) - holdings_returns.isnull().sum(axis=1)) / len(trading_universe)
    cash_wt = 1 - pos_wt
    total_return =  (pos_wt * holdings_returns.mean(axis=1)) + (cash_wt * cash_ret) - 0.001
    return total_return


def risk_weight_portfolio(px_data, signal):

    trading_universe = signal.columns.tolist()
    data_trading = px_data[trading_universe]['5/30/2007':]
    returns_df = data_trading.pct_change()
    std_df = 1.0 / returns_df.rolling(10).std()

    returns_df = data_trading.pct_change().shift(1)

    cash_ret = px_data['BIL']['5/30/2007':].pct_change().shift(1)[:-1]
    holdings_returns = returns_df[signal][:-1]
    holdings_std = std_df[signal][:-1]
    std_sum = holdings_std.sum(axis = 1)
    holdings_std = holdings_std.divide(std_sum, axis=0)

    pos_wt = (len(trading_universe) - holdings_returns.isnull().sum(axis=1)) / len(trading_universe)
    cash_wt = 1 - pos_wt
    total_return = (pos_wt * (holdings_std*holdings_returns).sum(axis=1)) + (cash_wt * cash_ret) - 0.001
    return total_return


if __name__ == "__main__":

    # universe list for the model
    universe_list = ['DBC', 'GLD', 'SPY', 'IEV', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'BIL', 'SHY']
    #   Universe Adj.Close dataframe
    #   df = pd.DataFrame({s:pull_data(s) for s in universe_list})
    #   df.to_csv("C:/Python27/Git/SMA_GTAA/adj_close.csv")
    # read_price_file('BM')

    adjusted_price = read_price_file('BM')
    #generate signal dataframe
    df_signal = ma_signal(adjusted_price, 10)

    #equal weight portfolio

    eq_wt_portfolio = equal_weight_portfolio(adjusted_price, df_signal).ffill()
    risk_wt_portfolio = risk_weight_portfolio(adjusted_price, df_signal).ffill()
    bm_ret = adjusted_price['5/30/2007':].pct_change()
    bm_ret =bm_ret[:-1]
    portfolio_returns = pd.DataFrame({'eq_wt' : eq_wt_portfolio, 'risk_wt' : risk_wt_portfolio, 'BM' : bm_ret['SPY'] }, index = risk_wt_portfolio.index)
    print(portfolio_returns)
    portfolio_returns.groupby(portfolio_returns.index.year).mean().plot(kind='bar')
    # portfolio_returns.cumsum().plot()
    plt.legend()
    plt.show()


