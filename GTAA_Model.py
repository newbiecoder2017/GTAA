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
#PGAIX - PIMCO Global Multi-Asset Fund Institutional Class
#GYLD - Arrow Dow Jones Global Yield ETF
#ACWI - iShares MSCI ACWI ETF
#AGG - iShares Core U.S. Aggregate Bond ETF  AGG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
yf.pdr_override() # <== that's all it takes :-)
pd.set_option('precision',4)
pd.options.display.float_format = '{:.3f}'.format



#Function to pull data from yahoo
def pull_data(s):
    return pdr.get_data_yahoo(s, start="2000-12-31", end="2018-04-30")['Adj Close']

def read_price_file(frq = 'BM'):
    df_price = pd.read_csv("C:/Python27/Git/SMA_GTAA/adj_close.csv", index_col='Date', parse_dates=True)
    df_price = df_price.resample(frq, closed='right').last()
    return df_price

def ma_signal(data, trading_universe, window):

    # trading_universe = ['DBC', 'GLD', 'SPY', 'IEV', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT']

    #calculating the rolling sma
    RollMa = data.rolling(window).mean()
    RollMa = RollMa[trading_universe]['5/30/2007':]

    #aligning price with the trading universe
    data_trading = data[trading_universe]['5/30/2007':]

    # #calulating the monthly returns and shifting it one period for mapping
    # returns_df = data_trading.pct_change().shift(1)

    #buy rule: if px(t) > 10M SMA(t)
    signal_df = data_trading > RollMa

    return signal_df

def equal_weight_cash_portfolio(px_data, signal):

    trading_universe = signal.columns.tolist()
    data_trading = px_data[trading_universe]['5/30/2007':]
    returns_df = data_trading.pct_change()
    cash_ret = px_data['BIL']['5/30/2007':].pct_change()
    holdings_returns = returns_df[signal.shift(1).bfill()]
    pos_wt = (len(trading_universe) - holdings_returns.isnull().sum(axis=1)) / len(trading_universe)
    cash_wt = 1 - pos_wt
    total_return =  (pos_wt * holdings_returns.mean(axis=1).fillna(0)) + (cash_wt * cash_ret) - 0.001
    return total_return

def risk_weight_portfolio(px_data, signal, window):

    trading_universe = signal.columns.tolist()
    data_trading = px_data[trading_universe]['5/30/2007':]
    returns_df = data_trading.pct_change()
    std_df = 1.0 / returns_df.rolling(3).std()
    returns_df = data_trading.pct_change()
    cash_ret = px_data['BIL']['5/30/2007':].pct_change()
    holdings_returns = returns_df[signal.shift(1).bfill()]
    holdings_std = std_df[signal]
    std_sum = holdings_std.sum(axis = 1)
    holdings_std_wt = holdings_std.divide(std_sum, axis=0)
    buy_wts = holdings_std_wt[-1:]
    holdings_std_wt = holdings_std_wt.shift(1)
    pos_wt = (len(trading_universe) - holdings_returns.isnull().sum(axis=1)) / len(trading_universe)
    # cash_wt = 1 - pos_wt
    # total_return = (pos_wt * (holdings_std*holdings_returns).sum(axis=1).fillna(0)) + (cash_wt * cash_ret) - 0.001
    total_return = ((holdings_std_wt * holdings_returns).sum(axis=1).fillna(0)) - 0.001

    return total_return, buy_wts

def risk_weight_benchmark(px_data, signal):

    data_trading = px_data[signal.columns.tolist()]
    data_trading = data_trading['5/30/2007':]
    returns_df = data_trading.pct_change()
    std_df = 1.0 / returns_df.rolling(3).std()
    # returns_df = data_trading.pct_change()

    std_sum = std_df.sum(axis = 1)
    holdings_std = std_df.divide(std_sum, axis=0)
    # holdings_std = holdings_std.shift(1)
    pos_wt = (len(returns_df) - returns_df.isnull().sum(axis=1)) / len(returns_df)

    # total_return = (pos_wt * (holdings_std*holdings_returns).sum(axis=1).fillna(0)) + (cash_wt * cash_ret) - 0.001
    total_return = ((holdings_std * returns_df).sum(axis=1).fillna(0))

    return total_return

def drawdown(s):

    # Get SPY data for past several years
    SPY_Dat = s

    # We are going to use a trailing 252 trading day window

    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    Roll_Max = SPY_Dat.rolling(center=False, min_periods=1, window=26).max()

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

def regression_fit(returnsframe, port, bm, rfr):
    # risk free rate
    rfr = rfr['5/30/2007':].fillna(0)
    rfr = rfr
    # excess returns
    eY = (returnsframe[port] - rfr).fillna(0)
    eX = (returnsframe[bm] - rfr).fillna(0)

    # scipy.stats regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(eX, eY)
    intercept = (1+intercept)**12 - 1

    return slope, intercept, r_value, p_value, std_err


def backtest_metrics(returnsframe, rfr):

    cummulative_return = (1 + returnsframe).cumprod()
    cpr = cummulative_return[-1:]
    N = len(returnsframe) / 12

    #Annualized returns
    AnnReturns = (cpr.pow(1 / N) - 1)
    #Annualized Risk
    AnnRisk = (np.sqrt(12) * returnsframe.std())

    def returns_risk(retFrame, per):

        #1Yr Return
        returnsframe_N = retFrame[-per:]
        N = len(returnsframe_N) / 12
        cpr_N = (1 + returnsframe_N).cumprod()
        annRet_N = (cpr_N[-1:].pow(1/N) - 1)
        std_N = np.sqrt(12) * returnsframe_N.std()
        return annRet_N.values.tolist(), std_N

    ret_12m, std_12m = returns_risk(returnsframe, 12 )
    ret_36m, std_36m = returns_risk(returnsframe, 36)
    ret_60m, std_60m = returns_risk(returnsframe, 60)

    #Sharpe Ratio with 2.5% annualized RFR
    AnnSharpe = (AnnReturns - 0.025) / AnnRisk

    #The Sortino ratio takes the asset's return and subtracts the risk-free rate, and then divides that amount by the asset's downside deviation. MAR is 5%
    df_thres = returnsframe - 0.05
    df_thres[df_thres > 0] = 0
    downward_risk = (np.sqrt(12) * df_thres.std())
    sortino_ratio = (AnnReturns-0.05) / downward_risk
    # AnnSharpe = (AnnReturns-0.025) / AnnRisk

#   # Calulate Average Daily Drawdowns and Max DD
    dd = [drawdown(cummulative_return[c])[0] for c in cummulative_return.columns]
    mdd = [drawdown(cummulative_return[c])[1] for c in cummulative_return.columns]

    # Calulate the win ratio and Gain to Loss ratio
    up = portfolio_returns[returnsframe > 0].count() / returnsframe.count()
    down = portfolio_returns[returnsframe < 0].count() / returnsframe.count()
    average_up = portfolio_returns[returnsframe > 0].mean()
    average_down = portfolio_returns[returnsframe < 0].mean()
    gain_to_loss = (average_up) / (-1 * average_down)

    #MAR ratio, annualised return over MDD. Higher the better
    mar_ratio = AnnReturns / mdd

    # Annualisec return over average annual DD
    sterling_ratio = AnnReturns / ([i*12 for i in dd])

    metric_df = pd.DataFrame(AnnReturns.values.tolist(), index = ['AnnRet(%)','AnnRisk(%)','AnnSharpe(2.5%)','Avg_DD(%)','MaxDD(%)','WinRate(%)','Gain_to_Loss','RoMDD','Sortino(5%)',
                                                                  'Sterling_Ratio','beta','ann_alpha','R_squared','p_value', 'std_err',
                                                                  '1YrReturns', '1YrRisk','3YrReturns', '3YrRisk', '5YrReturns', '5YrRisk'],
                                                                    columns = ['Avg_Universe', 'S&P500', 'bm_6040','eq_wt', 'momo_6040', 'momo_index', 'q_momo', 'qo_momo', 'risk_wt', 'risk_wt_bm'])
    metric_df.loc['AnnRet(%)'] = round(metric_df.loc['AnnRet(%)'], 3)*100
    metric_df.loc['AnnRisk(%)'] = 100 * AnnRisk
    metric_df.loc['AnnSharpe(2.5%)'] = AnnSharpe.values.tolist()[0]
    metric_df.loc['Avg_DD(%)'] =  [round(abs(i), 3) * 100 for i in dd]
    metric_df.loc['MaxDD(%)'] = [round(abs(i), 3) * 100 for i in mdd]
    metric_df.loc['WinRate(%)'] = round(up * 100, 3)
    metric_df.loc['Gain_to_Loss'] = round(gain_to_loss, 3)
    metric_df.loc['RoMDD'] = [round(abs(i),3) for i in mar_ratio.values.tolist()[0]]
    metric_df.loc['Sortino(5%)'] = sortino_ratio.values.tolist()[0]
    metric_df.loc['Sterling_Ratio'] = [round(abs(i),3) for i in sterling_ratio.values.tolist()[0]]
    metric_df.loc['1YrReturns'] = [i*100.00 for i in ret_12m[0]]
    metric_df.loc['1YrRisk'] = 100 * std_12m
    metric_df.loc['3YrReturns'] = [i*100.00 for i in ret_36m[0]]
    metric_df.loc['3YrRisk'] = 100 * std_36m
    metric_df.loc['5YrReturns'] = [i*100.00 for i in ret_60m[0]]
    metric_df.loc['5YrRisk'] = 100 * std_60m
    return metric_df


if __name__ == "__main__":

    window = 8
    # universe list for the model
    universe_list = ['DBC', 'GLD', 'IVV', 'IEV', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'BIL', 'SHY','ACWI','AGG','GYLD']
    trading_universe = ['DBC', 'GLD', 'IVV', 'IEV', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT']
    #Universe Adj.Close dataframe
    # df = pd.DataFrame({s:pull_data(s) for s in universe_list})
    # df.to_csv("C:/Python27/Git/SMA_GTAA/adj_close.csv")
    # read_price_file('BM')

    adjusted_price = read_price_file('BM')
    #risk free rate


    rfr = adjusted_price.BIL.pct_change()
    #generate signal dataframe
    df_signal= ma_signal(adjusted_price, trading_universe, window)

    #equal weight portfolio
    eq_wt_portfolio = equal_weight_cash_portfolio(adjusted_price, df_signal)

    #risk weight portfolio
    risk_wt_portfolio,  buyWeights = risk_weight_portfolio(adjusted_price, df_signal, window)

    #risk weight benchmark
    risk_wt_benchmark = risk_weight_benchmark(adjusted_price, df_signal)

    #Broad Market Benchmark : IVV
    bm_ret = adjusted_price['5/30/2007':].pct_change()
    bm_ret =bm_ret

    #Custom Benchmark 60% ACWi and 40% AGG
    bm_6040 = adjusted_price[['ACWI','AGG']]['5/30/2007':].pct_change()
    bm_6040_index = ((bm_6040['ACWI'] * 0.6) + (bm_6040['AGG'] * 0.4)).fillna(0)

    #reading NASDAQ MoMo returns and QQQE
    momo_df = pd.read_csv("momo_returns.csv", index_col = 0, parse_dates = True)
    momo_df = momo_df.resample('BM', closed='right').last().ffill()
    momo_df =momo_df['5/30/2007':]


    # Custom portfolio  for Nasdaq model and Risk weight GTAA portfolio - 60/40
    momo_6040 = (0.7* momo_df['qo_rebal']) + (0.3 * risk_wt_portfolio)

    # Custom portfolio for Nasdaq Equal Weight index and risk weight portfolio
    momo_6040_index = ((momo_df['qqqe'] * 0.7) + (risk_wt_portfolio * 0.3)).fillna(0)

    #Generates the tuple with buy candidates and its weights
    buy_list = tuple(zip(df_signal[-1:].columns.tolist(), buyWeights.values.tolist()[0]))


    #DataFrame for all the portfolios and benchmarks
    portfolio_returns = pd.DataFrame({'eq_wt' : eq_wt_portfolio, 'risk_wt' : risk_wt_portfolio, 'S&P500' : bm_ret['IVV'], 'Avg_Universe' : bm_ret[trading_universe].mean(axis=1),
                                      'risk_wt_bm' :risk_wt_benchmark, 'bm_6040' : bm_6040_index, 'qo_momo' : momo_df['qo_rebal'],
                                      'q_momo': momo_df['q_rebal'], 'momo_6040' : momo_6040, 'momo_index' : momo_6040_index}, index = risk_wt_portfolio.index)

    # Remove the first row with NaN's
    portfolio_returns = portfolio_returns[1:]

    #BackTest Statistics for all teh portfolios and indexes
    stats_df = backtest_metrics(portfolio_returns, rfr)
    stats_df.loc['Best_Month', :] = 100 * portfolio_returns.max()
    stats_df.loc['Worst_Month', :] = 100 * portfolio_returns.min()
    stats_df.loc['Best_Year', :] = 100 * portfolio_returns.groupby(portfolio_returns.index.year).sum().max()
    stats_df.loc['Worst_Year', :] = 100 * portfolio_returns.groupby(portfolio_returns.index.year).sum().min()

    #Regression stats for all portfolios and indices
    for c in stats_df.columns:
        stats_df[c].loc[['beta','ann_alpha','R_squared','p_value','std_err']] = regression_fit(portfolio_returns, c, 'S&P500', rfr)

    # print("Trade Recommendation: ", buy_list)
    trade_reco = pd.DataFrame([v for i, v in buy_list], index=[i for i, v in buy_list], columns=['Weights'])
    print(trade_reco)

    #Portfolio Return Plot
    # portfolio_returns = portfolio_returns[['eq_wt', 'risk_wt', "Avg_Universe"]]
    # print(100 * portfolio_returns.groupby(portfolio_returns.index.year).sum())
    # portfolio_returns.cumsum().plot()
    # plt.legend()
    # plt.grid()
    # plt.show()

    #Returns grouped by year
    portfolio_returns.rename(columns = {'eq_wt':'EW_GTAA', 'risk_wt':'RiskWt_GTAA', 'Avg_Universe':'EW_GTAA_Universe', 'risk_wt_bm':'RiskWt_GTAA_Universe',
                                        'bm_6040':'60/40_ACWI/AGG', 'qo_momo':'MomoPortfoli_QO', 'q_momo':'MomoPortfolio_Q',
                                        'momo_6040':'70/30_QO_MP/RW_GTAA','momo_index':'70/30_QQQE/RW_GTAA_bm'}, inplace = True)

    portfolio_returns = portfolio_returns[['EW_GTAA', 'EW_GTAA_Universe', 'RiskWt_GTAA', 'RiskWt_GTAA_Universe',
                                           'MomoPortfoli_QO', 'MomoPortfolio_Q', '70/30_QO_MP/RW_GTAA',
                                           '70/30_QQQE/RW_GTAA_bm', '60/40_ACWI/AGG', 'S&P500']]
    # print(100 * portfolio_returns.groupby(portfolio_returns.index.year).sum())
    # print(100 * np.sqrt(12) * portfolio_returns.groupby(portfolio_returns.index.year).std())
    # portfolio_returns.cumsum().plot()
    # plt.legend()
    # plt.grid()
    # plt.show()

    #correaltion Plot
    # plt.matshow(portfolio_returns.corr())
    # plt.xticks(range(len(portfolio_returns.columns)), portfolio_returns.columns)
    # plt.yticks(range(len(portfolio_returns.columns)), portfolio_returns.columns)
    # plt.colorbar()
    # plt.show()
    stats_df.rename(columns={'eq_wt': 'EW_GTAA', 'risk_wt': 'RiskWt_GTAA', 'Avg_Universe': 'EW_GTAA_Universe',
                                      'risk_wt_bm': 'RiskWt_GTAA_Universe',
                                      'bm_6040': '60/40_ACWI/AGG', 'qo_momo': 'MomoPortfoli_QO',
                                      'q_momo': 'MomoPortfolio_Q',
                                      'momo_6040': '70/30_QO_MP/RW_GTAA', 'momo_index': '70/30_QQQE/RW_GTAA_bm'},
                             inplace=True)

    stats_df = stats_df[['EW_GTAA', 'EW_GTAA_Universe', 'RiskWt_GTAA', 'RiskWt_GTAA_Universe',
                                           'MomoPortfoli_QO', 'MomoPortfolio_Q', '70/30_QO_MP/RW_GTAA',
                                           '70/30_QQQE/RW_GTAA_bm', '60/40_ACWI/AGG', 'S&P500']]
    # print(stats_df)
    tcor = pd.rolling_corr(portfolio_returns['RiskWt_GTAA'],portfolio_returns['MomoPortfoli_QO'],6)
    tcor.plot()
    plt.show()
    # ts1 = 100 * portfolio_returns.groupby(portfolio_returns.index.year).sum()
    # ts2 = 100 * np.sqrt(12) * portfolio_returns.groupby(portfolio_returns.index.year).std()
    # stats_df.to_csv("C:/Python27/Git/SMA_GTAA/Summary_Statistics.csv")
    # ts1.to_csv("C:/Python27/Git/SMA_GTAA/Return_Summary.csv")
    # ts2.to_csv("C:/Python27/Git/SMA_GTAA/Risk_Summary.csv")





