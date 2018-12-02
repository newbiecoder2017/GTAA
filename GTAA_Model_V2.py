# This model trades based on the 10 Month SMA rule rotating through different asset classes.
# universe consittuents
# DBC - DB Liquid Commoties Index etf
# GLD - Gold
# IVV  - U.S. Stocks (Fama French top 30% by market capitalization)
# IEV- European Stocks (Stoxx 350 Index)
# EWJ - Japanese Stocks (MSCI Japan)
# EEM - Emerging Market Stocks (MSCI EM)
# IYR - U.S. REITs (Dow Jones U.S. Real Estate Index)
# RWX - International REITs (Dow Jones Int’l Real Estate Index)
# IEF - Intermediate Treasuries (Barclays 7-10 Year Treasury Index)
# TLT - Long Treasuries (Barclays 20+ Year Treasury Index)
# BIL - 1-3 Month T Bill
# SHY - Barclays Capital U.S. 1-3 Year Treasury Bond Index
# PGAIX - PIMCO Global Multi-Asset Fund Institutional Class
# GYLD - Arrow Dow Jones Global Yield ETF
# ACWI - iShares MSCI ACWI ETF
# AGG - iShares Core U.S. Aggregate Bond ETF  AGG
# IJH - iShares Core S&P Mid Cap ETF
# IJR - iShares Core S&P Small Cap ETF
# IXUS - ishares Core MSCI Total INTL Stock ETF
# HDV - Ishares Core High Dividend ETF
# EMB - iShares J P Morgan USD Emerging Mkt Bonds ETF
# HYG - iShares iBoxx$ High Yield Corporate Bond ETF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr
import seaborn as sns
import time
import datetime
# from scipy import stats
import statsmodels.api as sm
yf.pdr_override()  # <== that's all it takes :-)
pd.set_option('precision', 4)
pd.options.display.float_format = '{:.3f}'.format
# sns.set_palette(sns.color_palette("Paired"))


# Function to pull data from yahoo
def pull_data(s):
    return pdr.get_data_yahoo(s, start="2000-12-31", end="2018-11-30")['Adj Close']


def read_price_file(frq='BM'):
    df_price = pd.read_csv("C:/Python27/Git/SMA_GTAA/GTAA/adj_close_v2.csv", index_col='Date', parse_dates=True)
    df_price = df_price.resample(frq, closed='right').last()
    # df_price = df_price['2011':]
    return df_price


def model_portfolios(cut_off=0.0, wList = [0.25,0.25,0.25,0.25]):
    df = pd.read_csv("C:/Python27/Git/SMA_GTAA/GTAA/adj_close_v2.csv", index_col='Date', parse_dates=True)

    df = df['12-2012':]
    # calculating the daily return for benchmarks
    rframe = df.resample('BM', closed='right').last().pct_change()

    bmivv = rframe.IVV

    bmgal = rframe.GAL

    bmacwi = rframe.ACWI

    bmbil = rframe.BIL

    bmmstar = rframe.MAHIP

    bmiyld = rframe.IYLD

    bm_60_40 = 0.6 * bmacwi + 0.4 * rframe.AGG

    def clean_universe(df, rs='BM', per=1, cutoff=0.5):
        # resampling price frame
        resamp_df = df.resample(rs, closed='right').last()

        # calculating the resampled price returns
        ret_frame = resamp_df.pct_change(per)

        # calculating the daily returns
        riskChg = df.pct_change()

        # calculating the rolling std deviations and re-sample the df
        risk_df = riskChg.rolling(30).apply(np.std).resample(rs, closed='right').last()

        # returning the risk asdjusted return frames
        return ret_frame / risk_df

    # 1 month risk adjusted  frame
    df_1m = clean_universe(df, rs='BM', cutoff=0.5, per=1)

    # Drop the benchmark from the return frame. Add symbols to the list to eliminate from thereturn frame
    rframe.drop(['GAL', 'BIL', 'GYLD','ACWI','AGG','IYLD', 'MAHIP','IXUS'], inplace=True, axis=1)

    df_1m.drop(['GAL', 'BIL', 'GYLD', 'ACWI','AGG', 'IYLD', 'MAHIP','IXUS'], inplace=True, axis=1)

    df.drop(['GAL', 'BIL', 'GYLD','AGG','IYLD', 'ACWI','MAHIP','IXUS'], inplace=True, axis=1)

    # 3 month risk adjusted frame
    df_3m = clean_universe(df, rs='BM', cutoff=0.5, per=3)

    # 6 month risk adjusted  frame
    df_6m = clean_universe(df, rs='BM', cutoff=0.5, per=6)

    # 12 month risk adjusted  frame
    df_12m = clean_universe(df, rs='BM', cutoff=0.5, per=12)

    # Zscore the risk adjusted return frames
    zs_1m = pd.DataFrame([(df_1m.iloc[i] - df_1m.iloc[i].mean())/df_1m.iloc[i].std() for i in range(len(df_1m))])
    zs_3m = pd.DataFrame([(df_3m.iloc[i] - df_3m.iloc[i].mean())/df_3m.iloc[i].std() for i in range(len(df_3m))])
    zs_6m = pd.DataFrame([(df_6m.iloc[i] - df_6m.iloc[i].mean())/df_6m.iloc[i].std() for i in range(len(df_6m))])
    zs_12m = pd.DataFrame([(df_12m.iloc[i] - df_12m.iloc[i].mean())/df_12m.iloc[i].std() for i in range(len(df_12m))])

    zs_1m = zs_1m.clip(lower=-3.0, upper=3.0, axis=1)
    zs_3m = zs_3m.clip(lower=-3.0, upper=3.0, axis=1)
    zs_6m = zs_6m.clip(lower=-3.0, upper=3.0, axis=1)
    zs_12m = zs_12m.clip(lower=-3.0, upper=3.0, axis=1)

    def return_persistence(data):
        return (data > 0).sum() - (data < 0).sum()

    # Generate the dataframe for Persistence return Factor from 1 month data frame
    persistence_long = df_1m.rolling(6).apply(return_persistence)
    persistence_short = df_1m.rolling(3).apply(return_persistence)
    # composte frame for the long and short persistence factors
    composite_persistence = 0.9 * persistence_long + 0.1 * persistence_short
    # Generate the zscore of composite persistence dataframe
    persistence_zscore = pd.DataFrame([(composite_persistence.iloc[i] - composite_persistence.iloc[i].mean()) / composite_persistence.iloc[i].std() for i in range(len(composite_persistence))])

    persistence_zscore = persistence_zscore.clip(lower=-3.0, upper=3.0, axis=1)
    # Genrate the composite zscore of return frames for different period returns frame
    rank_comp = wList[0] * zs_1m + wList[1] * zs_3m + wList[2] * zs_6m + wList[3] * zs_12m

    # Generate 1 month forward return based on the filtered composite zscore retrun dataframe
    df_portfolio = rframe.shift(-1)[rank_comp >= cut_off]

    # Using the persistence zscore dataframe to generate the position weights
    persistence_zscore = persistence_zscore[rank_comp >= cut_off]
    df_weights = pd.DataFrame([persistence_zscore.iloc[i] / abs(persistence_zscore.iloc[i]).sum() for i in range(len(persistence_zscore))]).abs()

    # Generate the weighted portfolio returns
    df_portfolio = df_weights * df_portfolio

    # Realigining the index of the portfolio
    df_portfolio = df_portfolio.shift(1)

    # calculate the portfolio return series and benchmark. Annual expense of 35bps is deducted monthly from the portfolio
    df_portfolio['Average'] = df_portfolio.sum(axis=1) - .000625   # 50bp of fees and transaction cost
    df_portfolio['bmIVV'] = bmivv
    df_portfolio['bmACWI'] = bmacwi
    df_portfolio['bmGAL'] = bmgal
    df_portfolio['bmBIL'] = bmbil
    df_portfolio['bm_60_40'] = bm_60_40
    df_portfolio['bmIYLD'] = bmiyld
    df_portfolio['bmMStar_AAHI'] = bmmstar
    return df_portfolio, df_weights


def drawdown(s):


    # Get SPY data for past several years
    bm_data = s

    # We are going to use a trailing 252 trading day window

    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    roll_max = bm_data.rolling(center=False, min_periods=1, window=12).max()

    daily_drawdown = bm_data / roll_max - 1.0

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    #     Max_Daily_Drawdown = pd.rolling_min(Daily_Drawdown, window, min_periods=1)

    max_daily_drawdown = daily_drawdown.rolling(center=False, min_periods=1, window=12).min()

    return daily_drawdown.mean(), max_daily_drawdown.min(), daily_drawdown


def regression_fit(port, bm, rfr):

    # risk free rate
    # rfr = rfr['01-2012':].fillna(0)
    # port = port['01-2012':]
    excess_return = port - rfr
    # excess returns
    ey = excess_return.fillna(0)
    ex = bm-rfr.fillna(0)
    ex = sm.add_constant(ex)

    # scipy.stats regression
    model = sm.OLS(ey, ex)
    result = model.fit()
    # result.params[0], result.params.loc['const'], result.rsquared_adj, result.pvalues, result.tvalues.loc['const']
    # intercept = (1+intercept)**12 - 1

    return [result.params[0], result.params.loc['const'], result.rsquared_adj, result.pvalues.loc['const'], result.tvalues.loc['const']]


def backtest_metrics(returnsframe, rfr):
    returnsframe.loc[:,'RFR'] = rfr
    cummulative_return = (1 + returnsframe).cumprod()
    cpr = cummulative_return[-1:]
    N = len(returnsframe) / 12

    # Annualized returns
    AnnReturns = (cpr.pow(1 / N) - 1)
    RFR_AnnRet = AnnReturns.RFR
    AnnReturns = AnnReturns.drop(['RFR'], axis = 1)
    returnsframe = returnsframe.drop(['RFR'], axis = 1)
    cummulative_return = cummulative_return.drop(['RFR'], axis = 1)
    # Annualized Risk
    AnnRisk = (np.sqrt(12) * returnsframe.std())

    def returns_risk(retFrame, per):

        # 1Yr Return
        returnsframe_N = retFrame[-per:]
        N = len(returnsframe_N) / 12
        cpr_N = (1 + returnsframe_N).cumprod()
        annRet_N = (cpr_N[-1:].pow(1/N) - 1)
        std_N = np.sqrt(12) * returnsframe_N.std()
        return annRet_N.values.tolist(), std_N

    ret_12m, std_12m = returns_risk(returnsframe, 12 )
    ret_36m, std_36m = returns_risk(returnsframe, 36)
    ret_60m, std_60m = returns_risk(returnsframe, 60)

    # Sharpe Ratio with 2.5% annualized RFR
    AnnSharpe = (AnnReturns - RFR_AnnRet.iloc[0]) / AnnRisk

    # The Sortino ratio takes the asset's return and subtracts the risk-free rate, and then divides that amount by the asset's downside deviation. MAR is 5%
    df_thres = returnsframe - 0.05
    df_thres[df_thres > 0] = 0
    downward_risk = (np.sqrt(12) * df_thres.std())
    sortino_ratio = (AnnReturns- RFR_AnnRet.iloc[0]) / downward_risk
    # AnnSharpe = (AnnReturns-0.025) / AnnRisk

#   # Calulate Average Daily Drawdowns and Max DD
#

    dd = [drawdown(cummulative_return)[0]]

    mdd = [drawdown(cummulative_return)[1]]

    daily_dd = drawdown(cummulative_return)[2]

    # Calulate the win ratio and Gain to Loss ratio
    up = returnsframe[returnsframe > 0].count() / returnsframe.count()
    # down = returnsframe[returnsframe < 0].count() / returnsframe.count()
    average_up = returnsframe[returnsframe > 0].mean()
    average_down = returnsframe[returnsframe < 0].mean()
    gain_to_loss = (average_up) / (-1 * average_down)

    # MAR ratio, annualised return over MDD. Higher the better
    mar_ratio = abs(AnnReturns / [mdd[0][i] for i in range(len(mdd[0]))])

    # Annualisec return over average annual DD
    sterling_ratio = abs(AnnReturns / [mdd[0][i] for i in range(len(mdd[0]))])

    metric_df = pd.DataFrame(AnnReturns.values.tolist(), index = ['AnnRet(%)','AnnRisk(%)','AnnSharpe(2.5%)','Avg_DD(%)','MaxDD(%)','WinRate(%)','Gain_to_Loss','RoMDD','Sortino(5%)',
                                                                  'Sterling_Ratio(over MDD)','beta','ann_alpha','R_squared','p_value', 'tvalue',
                                                                  '1YrReturns', '1YrRisk','3YrReturns', '3YrRisk', '5YrReturns', '5YrRisk'],
                                                                    columns = ['Average', 'bmGAL','bmIVV','bmACWI','bm_60_40', 'bmIYLD', 'bmMStar_AAHI' ])
    metric_df.loc['AnnRet(%)'] = round(metric_df.loc['AnnRet(%)'], 3)*100
    metric_df.loc['AnnRisk(%)'] = [100*i for i in AnnRisk]
    metric_df.loc['AnnSharpe(2.5%)'] = AnnSharpe.values.tolist()[0]
    metric_df.loc['Avg_DD(%)'] =  [round(float(i), 3) for i in [abs(i) * 100 for i in dd[0]]]
    metric_df.loc['MaxDD(%)'] = [round(float(i), 3) for i in [abs(i) * 100 for i in mdd[0]]]
    metric_df.loc['WinRate(%)'] = [100* round(float(i), 3) for i in up.values.tolist()]
    metric_df.loc['Gain_to_Loss'] = [round(float(i),3) for i in gain_to_loss.values.tolist()]
    metric_df.loc['RoMDD'] = [round(abs(i),3) for i in mar_ratio.values.tolist()[0]]
    metric_df.loc['Sortino(5%)'] = sortino_ratio.values.tolist()[0]
    metric_df.loc['Sterling_Ratio(over MDD)'] = [round(abs(i),3) for i in sterling_ratio.values.tolist()[0]]
    metric_df.loc['1YrReturns'] = [i*100.00 for i in ret_12m[0]]
    metric_df.loc['1YrRisk'] = [100 * i for i in std_12m.values.tolist()]
    metric_df.loc['3YrReturns'] = [i*100.00 for i in ret_36m[0]]
    metric_df.loc['3YrRisk'] = [100 * i for i in std_36m.values.tolist()]
    metric_df.loc['5YrReturns'] = [i*100.00 for i in ret_60m[0]]
    metric_df.loc['5YrRisk'] = [100 * i for i in std_60m.values.tolist()]
    metric_df.loc['Total Return'] = returnsframe['12-2012':].cumsum()[-1:].values[0].tolist()

    return metric_df, daily_dd
    # df.loc['Total Return'] = returnsframe.cumsum()[-1:].values[0].tolist()
    # return metric_df
def dollar_retuns(data):


    idx = data.index.tolist()
    for i in range(len(data)):
        data = data.fillna(0)
        if i == 0:
            data.loc[idx[i],'Portfolio'] = 10000
            data.loc[idx[i], 'S&P 500'] = 10000
            data.loc[idx[i], 'GAL'] = 10000
            data.loc[idx[i], '60/40 Allocation'] = 10000
            data.loc[idx[i], 'IYLD'] = 10000
            data.loc[idx[i], 'MStar_AAHI'] = 10000
        else:
            data.loc[idx[i], 'Portfolio'] = (1 + data.loc[idx[i], 'Average']) * data.loc[idx[i-1], 'Portfolio']
            data.loc[idx[i], 'S&P 500'] = (1 + data.loc[idx[i], 'bmIVV']) * data.loc[idx[i-1], 'S&P 500']
            data.loc[idx[i], 'GAL'] = (1 + data.loc[idx[i], 'bmGAL']) * data.loc[idx[i-1], 'GAL']
            data.loc[idx[i], '60/40 Allocation'] = (1 + data.loc[idx[i], 'bm_60_40']) * data.loc[idx[i-1], '60/40 Allocation']
            data.loc[idx[i], 'IYLD'] = (1 + data.loc[idx[i], 'bmIYLD']) * data.loc[idx[i - 1], 'IYLD']
            data.loc[idx[i], 'MStar_AAHI'] = (1 + data.loc[idx[i], 'bmMStar_AAHI']) * data.loc[idx[i - 1], 'MStar_AAHI']


    # return data[['Portfolio','S&P 500','GAL','60/40 Allocation','IYLD','MStar_AAHI']]
    return data[['Portfolio', 'GAL', 'IYLD','MStar_AAHI']]

if __name__ == "__main__":

    # universe list for the model
    universe_list = ['DBC', 'GLD', 'IVV', 'IEV', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT', 'BIL', 'SHY','ACWI','AGG','GYLD','GAL','IJH','IJR','IXUS','HDV','IYLD','EMB', 'HYG']
    # trading_universe = ['DBC', 'GLD', 'IVV', 'IEV', 'EWJ', 'EEM', 'IYR', 'RWX', 'IEF', 'TLT']

    # Universe Adj.Close dataframe - MAHIP : Copy IYLD
    # df = pd.DataFrame({s:pull_data(s) for s in universe_list})
    # df.to_csv("C:/Python27/Git/SMA_GTAA/GTAA/adj_close_v2.csv")


    adjusted_price = read_price_file('BM')
    modBiL = adjusted_price.BIL.pct_change()

    # read_price_file('BM')
    n1 = 0.0
    n2 = 0.0
    n3 = 0.9
    n4 = 0.1
    model, wts = model_portfolios(cut_off=0.1, wList=[n1,n2,n3,n4])

    # Try with BIL and GYLD, w/o BIL and GYLD and combinations
    # best persistence set is 0.1, 0.9 - Long/Short
    # Best sets are  [0.1,0,0.8,0.1], [0.0,0,0.9,.1], [0.0,0,0.8,0.2] an combinations

    # print(model.tail())
    # print(wts.tail())
    # print(buy_wts[-1:])
    #
    #
    # #DataFrame for all the portfolios and benchmarks
    # portfolio_returns = pd.DataFrame({'eq_wt' : eq_wt_portfolio, 'risk_wt' : risk_wt_portfolio, 'S&P500' : bm_ret['IVV'], 'Avg_Universe' : bm_ret[trading_universe].mean(axis=1),
    #                                   'risk_wt_bm' :risk_wt_benchmark, 'bm_6040' : bm_6040_index, 'qo_momo' : momo_df['qo_rebal'],
    #                                   'q_momo': momo_df['q_rebal'], 'momo_6040' : momo_6040, 'momo_index' : momo_6040_index}, index = risk_wt_portfolio.index)
    #
    # # Remove the first row with NaN's
    # portfolio_returns = portfolio_returns[1:]
    #
    # #BackTest Statistics for all the portfolios and indexes
    stats_df, daily_dd = backtest_metrics(model[['Average','bmGAL','bmIVV','bmACWI','bm_60_40', 'bmIYLD','bmMStar_AAHI']], rfr = modBiL)
    portfolio_returns = model[['Average','bmGAL','bmIVV','bmACWI', 'bm_60_40','bmIYLD','bmMStar_AAHI']]
    portfolio_returns.to_csv("C:/Python27/Git/SMA_GTAA/GTAA/returns.csv")
    stats_df.loc['Best_Month', :] = [100 * float(i) for i in portfolio_returns.max().values.tolist()]
    stats_df.loc['Worst_Month', :] = [100 * float(i) for i in portfolio_returns.min().values.tolist()]
    stats_df.loc['Best_Year', :] = [100 * float(i) for i in portfolio_returns.groupby(portfolio_returns.index.year).sum().max()]
    stats_df.loc['Worst_Year', :] = [100 * float(i) for i in portfolio_returns.groupby(portfolio_returns.index.year).sum().min()]

    #Regression stats for all portfolios and indices
    for c in stats_df.columns:

        stats_df[c].loc[['beta','ann_alpha','R_squared','p_value','tvalue']] = regression_fit(portfolio_returns[c][-36:], model.bmIYLD[-36:].fillna(0), model.bmBIL[-36:].fillna(0))
    # cutt_off=1.5
    # stats_df.to_csv("C:/Python27/Git/SMA_GTAA/GTAA/"+str(n1)+"_"+str(n2)+"_"+str(n3)+"_"+str(n4)+"_"+str(cutt_off)+".csv")
    print(stats_df)
    # # print("Trade Recommendation: ", buy_list)
    # trade_reco = pd.DataFrame([v for i, v in buy_list], index=[i for i, v in buy_list], columns=['Weights'])

    # safe_assests = wts[['SHY', 'TLT', 'GLD']].fillna(0).sum(axis=1).rolling(6).mean()
    # risky_assests = wts[['IVV', 'IEV', 'EWJ', 'EEM', 'IJH', 'IYR', 'IXUS']].fillna(0).sum(axis=1).rolling(6).mean()
    # portfolio_returns[['bmIVV','bmACWI']].cumsum().plot()
    # safe_assests.plot(kind='area')
    # risky_assests.plot(kind='area')
    # plt.grid()
    # plt.title("Safe Asset Exposure vs S&P 500")
    # plt.show()

    # Allocations Plot
    # x = wts[-1:].dropna(axis=1).values
    # y = wts[-1:].dropna(axis=1).columns
    # plt.pie(x[0], labels=y, shadow=False, startangle=90, autopct='%1.1f%%')
    # plt.title("Allocations as of %s" %(portfolio_returns.index[-1:][0].strftime('%m/%d/%Y')))
    # plt.show()
    wts.to_csv("C:/Python27/Git/SMA_GTAA/GTAA/weights.csv")
    print(wts.tail())

    # DrawDown Plot
    # daily_dd.fillna(0).rolling(6).mean().plot(color='rgbc')
    # plt.title("6m Rolling DrawDowns")
    # plt.grid()
    # plt.legend()
    # plt.ylabel("% Drawdown")
    # plt.savefig("C:/Python27/Git/SMA_GTAA/drawdowns_LTH.png")
    # plt.show()


    # Plot the rolling weights
    # y = np.vstack([wts[c].fillna(0) for c in wts.columns])
    # plt.stackplot(wts.index, y, labels = ['SHY'])


    # safe assets plot
    # t = wts[['TLT', 'SHY', 'AGG', 'IEF']].sum(axis=1)
    # t.fillna(0).rolling(6).mean().plot()
    # plt.title("Safe_Assets_Allocations")

    # risky assets plot
    # wts[['IVV', 'EWJ', 'EEM', 'IEV']].fillna(0).rolling(6).mean().plot(color = 'rgb')
    # plt.title("Risky_Assets_Allocations")

    # equity/bond allocations plot
    # wts[['IVV', 'TLT','IEF','SHY']].fillna(0).rolling(6).mean().plot(color = 'rgbc')
    # plt.title("US_Equity_Bonds_Allocations")

    # All Equity Bond Allocations
    # global_equity = wts[['IVV','IEV','EWJ','EEM','IJH','IJR', 'IXUS']]
    # global_equity.loc[:,'Average'] =  global_equity.sum(axis=1)
    # global_debt = wts[['IEF', 'TLT', 'SHY', 'AGG','IEF']]
    # global_debt.loc[:, 'Average'] = global_debt.sum(axis=1)
    # new_df_plot = pd.DataFrame(columns = ['Global_Equity', 'Global_Bonds'])
    # new_df_plot.loc[:, 'Global_Equity'] = global_equity['Average'].fillna(0)
    # new_df_plot.loc[:,'Global_Bonds'] = global_debt['Average'].fillna(0)
    # new_df_plot.rolling(6).mean().plot(color = 'rgb')
    # plt.title("Global_Equity_Bonds_Allocations")
    # plt.legend()
    # plt.grid()
    # plt.show()


    # plt.ylabel("% allocation (rolling 6 months avg)")
    # plt.legend()
    # plt.grid()
    # plt.savefig("C:/Python27/Git/SMA_GTAA/Global_equity_bond_LTH.png")

    # wts[['GLD','SHY','AGG']].fillna(0).rolling(6).mean().plot()
    # plt.legend()
    #  plt.grid()

    # plt.show()

    # Portfolio Return Plot
    # portfolio_returns = portfolio_returns[['Average','bmGAL', 'bm_60_40']]
    # perf_yr = 100 * portfolio_returns.groupby(portfolio_returns.index.year).sum()
    # perf_yr.plot(kind='bar')
    # plt.title("Historical Performance - Portfolio vs Benchmark")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # #Returns grouped by year
    # portfolio_returns.rename(columns = {'eq_wt':'EW_GTAA', 'risk_wt':'RiskWt_GTAA', 'Avg_Universe':'EW_GTAA_Universe', 'risk_wt_bm':'RiskWt_GTAA_Universe',
    #                                     'bm_6040':'60/40_ACWI/AGG', 'qo_momo':'MomoPortfoli_QO', 'q_momo':'MomoPortfolio_Q',
    #                                     'momo_6040':'70/30_QO_MP/RW_GTAA','momo_index':'70/30_QQQE/RW_GTAA_bm'}, inplace = True)
    #
    # portfolio_returns = portfolio_returns[['EW_GTAA', 'EW_GTAA_Universe', 'RiskWt_GTAA', 'RiskWt_GTAA_Universe',
    #                                        'MomoPortfoli_QO', 'MomoPortfolio_Q', '70/30_QO_MP/RW_GTAA',
    #                                        '70/30_QQQE/RW_GTAA_bm', '60/40_ACWI/AGG', 'S&P500']]
    print(100 * portfolio_returns.groupby(portfolio_returns.index.year).sum())
    # # print(100 * np.sqrt(12) * portfolio_returns.groupby(portfolio_returns.index.year).std())
    # # portfolio_returns.cumsum().plot()
    # # plt.legend()
    # # plt.grid()
    # # plt.show()
    #
    # #correaltion Plot
    # plt.matshow(portfolio_returns.corr())
    # plt.xticks(range(len(portfolio_returns.columns)), portfolio_returns.columns)
    # plt.yticks(range(len(portfolio_returns.columns)), portfolio_returns.columns)
    # plt.colorbar()
    # plt.show()
    # stats_df.rename(columns={'eq_wt': 'EW_GTAA', 'risk_wt': 'RiskWt_GTAA', 'Avg_Universe': 'EW_GTAA_Universe',
    #                                   'risk_wt_bm': 'RiskWt_GTAA_Universe',
    #                                   'bm_6040': '60/40_ACWI/AGG', 'qo_momo': 'MomoPortfoli_QO',
    #                                   'q_momo': 'MomoPortfolio_Q',
    #                                   'momo_6040': '70/30_QO_MP/RW_GTAA', 'momo_index': '70/30_QQQE/RW_GTAA_bm'},
    #                          inplace=True)
    #
    # stats_df = stats_df[['EW_GTAA', 'EW_GTAA_Universe', 'RiskWt_GTAA', 'RiskWt_GTAA_Universe',
    #                                        'MomoPortfoli_QO', 'MomoPortfolio_Q', '70/30_QO_MP/RW_GTAA',
    #                                        '70/30_QQQE/RW_GTAA_bm', '60/40_ACWI/AGG', 'S&P500']]
    # # print(stats_df)
    # Correlation Plot
    # tcor = pd.rolling_corr(portfolio_returns['Average'],portfolio_returns['bmGAL'],6)
    # tcor.plot(color = 'r')
    # plt.title("Rolling Correlation Plot - Portfolio vs SPDR Global Asset Allocation Fund (GAL)", color='blue',fontweight='heavy')
    # plt.grid()
    # plt.show()
    # ts1 = 100 * portfolio_returns.groupby(portfolio_returns.index.year).sum()
    # ts2 = 100 * np.sqrt(12) * portfolio_returns.groupby(portfolio_returns.index.year).std()
    # print(ts1)
    # print(ts2)
    # # stats_df.to_csv("C:/Python27/Git/SMA_GTAA/GTAA/Summary_Statistics.csv")
    # # ts1.to_csv("C:/Python27/Git/SMA_GTAA/GTAA/Return_Summary.csv")
    # # ts2.to_csv("C:/Python27/Git/SMA_GTAA/GTAA/Risk_Summary.csv")
    # print(wts.tail(10))

    # ollar Growth of a Portfolio
    dollar_returns = dollar_retuns(portfolio_returns)
    dollar_returns.plot(linewidth = 2.0)
    plt.legend()
    plt.grid()
    plt.ylabel("in dollars($)")
    plt.title("Growth of a $10000 Portfolio ",color='blue',fontweight='heavy')
    # plt.savefig("C:/Python27/Git/SMA_GTAA/Sectors/equity_curve.jpg", facecolor=fig.get_facecolor(), transparent=True)
    plt.show()


