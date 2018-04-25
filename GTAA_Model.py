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

def drawdown(s):

    # Get SPY data for past several years
    SPY_Dat = s

    # We are going to use a trailing 252 trading day window

    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    Roll_Max = SPY_Dat.rolling(center=False, min_periods=1, window=12).max()

    Daily_Drawdown = SPY_Dat / Roll_Max - 1.0

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    #     Max_Daily_Drawdown = pd.rolling_min(Daily_Drawdown, window, min_periods=1)

    Max_Daily_Drawdown = Daily_Drawdown.rolling(center=False, min_periods=1, window=12).min()

    return Daily_Drawdown.mean(), Max_Daily_Drawdown.min()

    # Plot the results
    # Daily_Drawdown.plot()
    # Max_Daily_Drawdown.plot()
    # plt.legend()
    # plt.grid()
    # plt.title("DrawDown and MaxDD for %s" %(s.name))
    # plt.show()

def backtest_metrics(returnsframe):

    cummulative_return = (1 + returnsframe).cumprod()
    cpr = cummulative_return[-1:]
    N = len(returnsframe) / 12
    AnnReturns = (cpr.pow(1 / N) - 1)
    AnnRisk = (np.sqrt(12) * returnsframe.std())
    AnnSharpe = (AnnReturns-0.025) / AnnRisk
    dd = [drawdown(cummulative_return[c])[0] for c in cummulative_return.columns]
    mdd = [drawdown(cummulative_return[c])[1] for c in cummulative_return.columns]
    up = portfolio_returns[returnsframe > 0].count() / returnsframe.count()
    down = portfolio_returns[returnsframe < 0].count() / returnsframe.count()
    average_up = portfolio_returns[returnsframe > 0].mean()
    average_down = portfolio_returns[returnsframe < 0].mean()
    gain_to_loss = (average_up) / (-1 * average_down)

    metric_df = pd.DataFrame(AnnReturns.values.tolist(), index = ['AnnRet(%)','AnnRisk(%)','AnnSharpe(2.5%)','Avg_DD(%)','MaxDD(%)','WinRate(%)','Gain_to_Loss'],columns = ['Avg_Universe', 'BM', 'eq_wt', 'risk_wt'])
    metric_df.loc['AnnRet(%)'] = round(metric_df.loc['AnnRet(%)'], 3)*100
    metric_df.loc['AnnRisk(%)'] = 100 * AnnRisk
    metric_df.loc['AnnSharpe(2.5%)'] = AnnSharpe.values.tolist()[0]
    metric_df.loc['Avg_DD(%)'] =  [round(abs(i), 3) * 100 for i in dd]
    metric_df.loc['MaxDD(%)'] = [round(abs(i), 3) * 100 for i in mdd]
    metric_df.loc['WinRate(%)'] = round(up * 100, 3)
    metric_df.loc['Gain_to_Loss'] = round(gain_to_loss, 3)
    return metric_df


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
    portfolio_returns = pd.DataFrame({'eq_wt' : eq_wt_portfolio, 'risk_wt' : risk_wt_portfolio, 'BM' : bm_ret['SPY'],"Avg_Universe" : bm_ret.mean(axis=1)}, index = risk_wt_portfolio.index)
    print(backtest_metrics(portfolio_returns))





