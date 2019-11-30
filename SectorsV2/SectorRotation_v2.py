# This model trades based on the 10 Month SMA rule rotating through different asset classes.
# universe consittuents
# RWR - SPDR REITs
# XLRE - SPDR Real Estate
# XLC - SPDR Communication Services
# XLB - SPDR Materials
# XLI - SPDR Industrials
# XLY - Consumer Discretionary
# XLP - Consumer Staples
# XLE - Energy
# XLF - Financials
# XLU - Utilities
# XLV - Healthcare
# XLK - Technology
# IYT - Transportation
# BIL - 1-3 Month T Bill
# SHY - Barclays Capital U.S. 1-3 Year Treasury Bond Index


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
from scipy import stats
import statsmodels.api as sm
yf.pdr_override() # <== that's all it takes :-)
pd.set_option('precision',4)
pd.options.display.float_format = '{:.3f}'.format
import time
import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
# sns.set_palette(sns.color_palette("hls", 20))

today = datetime.datetime.today().strftime('%m/%d/%y')

#Function to pull data from yahoo
def pull_data(s):

    return pdr.get_data_yahoo(s, start="2000-12-31", end="2019-11-30")['Adj Close']

def read_price_file(frq = 'BM'):
    df_price = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/adj_close_sectors.csv", index_col='Date', parse_dates=True)
    df_price = df_price.resample(frq, closed='right').last()
    return df_price

def model_portfolios(cut_off=0.0, wList=[0.25,0.25,0.25,0.25], mod='cash'):
    df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/adj_close_sectors.csv", index_col='Date', parse_dates=True)
    # df = df['01-2015':]
    #calculating the daily return for benchmarks
    rframe = df.resample('BM', closed='right').last().pct_change()
    bmSPY = rframe.SPY
    bmbil = rframe.BIL


    def clean_universe(df, rs='BM', per=1, cutoff=0.5):
        # resampling price frame
        resamp_df = df.resample(rs, closed='right').last()

        # calculating the resampled price returns in excess of the benchmark
        ret_frame = resamp_df.pct_change(per)
        ret_frame1 = pd.DataFrame({s: ret_frame[s] - ret_frame.SPY for s in ret_frame.columns}, index=ret_frame.index)
        if per==1:
            ret_frame2 = ret_frame1.rolling(12).mean().sub(ret_frame)
        else:
            ret_frame2 = ret_frame1.rolling(per).mean().sub(ret_frame)

        # calculating the daily returns
        riskChg = df.pct_change()

        # calculating the rolling std deviations and re-sample the df
        risk_df = riskChg.rolling(30).apply(np.std, raw=True).resample(rs, closed='right').last()

        # returning the risk asdjusted return frames
        return ret_frame2 / risk_df


    #1 month risk adjusted  frame
    df_1m = clean_universe(df, rs='BM', cutoff=0.5, per=1)

    #Drop the benchmark from the return frame. Add symbols to the list to eliminate from the return frame
    if mod == 'nocash':

        rframe.drop(['SPY','BIL','SHY'], inplace=True, axis=1)

        df_1m.drop(['SPY','BIL','SHY'], inplace=True, axis=1)
    else:
        rframe.drop(['SPY', 'BIL'], inplace=True, axis=1)

        df_1m.drop(['SPY', 'BIL'], inplace=True, axis=1)

    # df.drop(['SPY','BIL'], inplace=True, axis=1)

    # 3 month risk adjusted frame
    df_3m = clean_universe(df, rs='BM', cutoff=0.5, per=3)

    # 6 month risk adjusted  frame
    df_6m = clean_universe(df, rs='BM', cutoff=0.5, per=6)

    # 12 month risk adjusted  frame
    df_12m = clean_universe(df, rs='BM', cutoff=0.5, per=12)
    #delete these
    if mod == 'nocash':

        df.drop(['SPY', 'BIL','SHY'], inplace=True, axis=1)
        df_3m.drop(['SPY', 'BIL','SHY'], inplace=True, axis=1)
        df_6m.drop(['SPY', 'BIL','SHY'], inplace=True, axis=1)
        df_12m.drop(['SPY', 'BIL','SHY'], inplace=True, axis=1)
    else:
        df.drop(['SPY', 'BIL'], inplace=True, axis=1)
        df_3m.drop(['SPY', 'BIL'], inplace=True, axis=1)
        df_6m.drop(['SPY', 'BIL'], inplace=True, axis=1)
        df_12m.drop(['SPY', 'BIL'], inplace=True, axis=1)


    #Zscore the risk adjusted return frames Cross Sectional

    zs_1m = pd.DataFrame([(df_1m.iloc[i] - df_1m.iloc[i].mean())/df_1m.iloc[i].std() for i in range(len(df_1m))])
    zs_3m = pd.DataFrame([(df_3m.iloc[i] - df_3m.iloc[i].mean())/df_3m.iloc[i].std() for i in range(len(df_3m))])
    zs_6m = pd.DataFrame([(df_6m.iloc[i] - df_6m.iloc[i].mean())/df_6m.iloc[i].std() for i in range(len(df_6m))])
    zs_12m = pd.DataFrame([(df_12m.iloc[i] - df_12m.iloc[i].mean())/df_12m.iloc[i].std() for i in range(len(df_12m))])

    zs_1m = zs_1m.clip(lower=-3.0, upper=3.0, axis=1)
    zs_3m = zs_3m.clip(lower=-3.0, upper=3.0, axis=1)
    zs_6m = zs_6m.clip(lower=-3.0, upper=3.0, axis=1)
    zs_12m = zs_12m.clip(lower=-3.0, upper=3.0, axis=1)

    def return_persistence(data):
        return ((data > 0).sum() - (data < 0).sum())

    #Generate the dataframe for Persistence return Factor from 1 month data frame
    persistence_short = df_1m.rolling(3).apply(return_persistence, raw=True)
    persistence_inter = df_1m.rolling(6).apply(return_persistence, raw=True)
    persistence_long= df_1m.rolling(12).apply(return_persistence, raw=True)

    # composte frame for the long and short persistence factors use long wts  = 0.9 and short wts = 0.1 for less drawdown
    composite_persistence = 0.0* persistence_short  + 1.0* persistence_inter + 0.0 * persistence_long

    # Generate the zscore of composite persistence dataframe Cross Sectional
    persistence_zscore = pd.DataFrame([(composite_persistence.iloc[i] - composite_persistence.iloc[i].mean()) / composite_persistence.iloc[i].std() for i in range(len(composite_persistence))])

    persistence_zscore = persistence_zscore.clip(lower=-3.0, upper=3.0, axis=1)
    #Genrate the composite zscore of return frames for different period returns frame
    rank_comp = wList[0] * zs_1m + wList[1] * zs_3m + wList[2] * zs_6m + wList[3] * zs_12m

    # Generate 1 month forward return based on the filtered composite zscore retrun dataframe
    df_portfolio = rframe.shift(-1)[rank_comp >= cut_off]
    alt_df_portfolio = rframe.shift(-1)[rank_comp < cut_off]

    # Using the persistence zscore dataframe to generate the position weights
    pers_score = persistence_zscore
    persistence_zscore = persistence_zscore[rank_comp >= cut_off]
    alt_persistence_zscore = pers_score[rank_comp < cut_off]

    df_weights = pd.DataFrame([persistence_zscore.iloc[i] / abs(persistence_zscore.iloc[i]).sum() for i in range(len(persistence_zscore))]).abs()
    alt_df_weights = pd.DataFrame([alt_persistence_zscore.iloc[i] / abs(alt_persistence_zscore.iloc[i]).sum() for i in range(len(alt_persistence_zscore))]).abs()

    print(df_weights.sum(axis=1).tail())
    # print(alt_df_weights.sum(axis=1).tail())

    # Generate the weighted portfolio returns
    df_portfolio = df_weights * df_portfolio
    alt_df_portfolio = alt_df_weights * alt_df_portfolio

    # Realigining the index of the portfolio
    df_portfolio = df_portfolio.shift(1)
    # alt_df_portfolio = alt_df_portfolio.shift(1)

    # calculate the portfolio return series and benchmark. Annual expense of 35bps is deducted monthly from the portfolio
    df_portfolio['Average'] = df_portfolio.sum(axis=1) - .00083  #100bp of fees and transaction cost
    # df_portfolio['alt_Average'] = alt_df_portfolio.sum(axis=1) - .00083  # 100bp of fees and transaction cost
    df_portfolio['bmSPY'] = bmSPY
    df_portfolio['bmBIL'] = bmbil
    # df_portfolio[['Average','alt_Average','bmSPY']].cumsum().plot()
    # plt.grid()
    return df_portfolio, df_weights,rframe

def drawdown(s):

    # Get SPY data for past several years
    SPY_Dat = s

    # We are going to use a trailing 252 trading day window

    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    Roll_Max = SPY_Dat.rolling(center=False, min_periods=6, window=12).max()

    Daily_Drawdown = SPY_Dat / Roll_Max - 1.0

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    # Max_Daily_Drawdown = pd.rolling_min(Daily_Drawdown, window, min_periods=1)

    Max_Daily_Drawdown = Daily_Drawdown.rolling(center=False, min_periods=1, window=12).min()

    return Daily_Drawdown.mean(), Max_Daily_Drawdown.min(),Daily_Drawdown

def regression_fit(port, bm, rfr):

    excess_return = port - rfr
        # excess returns
    eY = excess_return.fillna(0)
    eX = bm-rfr.fillna(0)
    eX = sm.add_constant(eX)

    # scipy.stats regression
    model = sm.OLS(eY,eX)
    result = model.fit()
    # result.params[0], result.params.loc['const'], result.rsquared_adj, result.pvalues, result.tvalues.loc['const']
    # intercept = (1+intercept)**12 - 1

    return [result.params[0], 100*((1+result.params.loc['const'])**12-1), result.rsquared_adj, result.pvalues.loc['const'], result.tvalues.loc['const']]

def backtest_metrics(returnsframe, rfr):
    returnsframe['RFR'] = rfr
    cummulative_return = (1 + returnsframe).cumprod()
    cpr = cummulative_return[-1:]
    N = len(returnsframe) / 12

    #Annualized returns
    AnnReturns = (cpr.pow(1 / N) - 1)
    RFR_AnnRet = AnnReturns.RFR
    AnnReturns = AnnReturns.drop(['RFR'], axis = 1)
    returnsframe = returnsframe.drop(['RFR'], axis = 1)
    cummulative_return = cummulative_return.drop(['RFR'], axis = 1)
    #Annualized Risk
    AnnRisk = (np.sqrt(12) * returnsframe.std())

    def returns_risk(retFrame, per):

        #1Yr Return
        if per<=len(retFrame):

            returnsframe_N = retFrame[-per:]
            N = len(returnsframe_N) / 12
            cpr_N = (1 + returnsframe_N).cumprod()
            annRet_N = (cpr_N[-1:].pow(1/N) - 1)
            std_N = np.sqrt(12) * returnsframe_N.std()
            return annRet_N.values.tolist(), std_N
        else:
            none_ret = [[0.0,0.0,0.0]]
            none_std = pd.Series([0.0,0.0,0.0], index = retFrame.columns)
            return none_ret, none_std

    ret_12m, std_12m = returns_risk(returnsframe, 12)
    ret_36m, std_36m = returns_risk(returnsframe, 36)
    ret_60m, std_60m = returns_risk(returnsframe, 60)
    ret_120m, std_120m = returns_risk(returnsframe, 120)

    #Sharpe Ratio with 2.5% annualized RFR
    AnnSharpe = (AnnReturns - RFR_AnnRet.iloc[0]) / AnnRisk

    #The Sortino ratio takes the asset's return and subtracts the risk-free rate, and then divides that amount by the asset's downside deviation. MAR is 5%
    df_thres = returnsframe - 0.05
    df_thres[df_thres > 0] = 0
    downward_risk = (np.sqrt(12) * df_thres.std())
    sortino_ratio = (AnnReturns- RFR_AnnRet.iloc[0]) / downward_risk
    # AnnSharpe = (AnnReturns-0.025) / AnnRisk

  #Calulate Average Daily Drawdowns and Max DD
    dd = [drawdown(cummulative_return)[0]]
    mdd = [drawdown(cummulative_return)[1]]
    dailyDD = [drawdown(cummulative_return)[2]]
    drawdown_df = pd.DataFrame(dailyDD[0])
    drawdown_df = round(100.0 * drawdown_df[['portfolio','bmSPY']],2)
    drawdown_df = drawdown_df.rename(columns= {'portfolio':'Portfolio','bmSPY':'S&P 500'})
    drawdown_df[['Portfolio','S&P 500']].plot()
    plt.legend()
    plt.grid()
    plt.ylabel(("percent(%)"))
    plt.title("Portfolio vs S&P 500 Rolling Drawdown - 12 Month")
    plt.savefig("C:/Python27/Git/SMA_GTAA/Sectors/drawDown.jpg", transparent=True)
    plt.show()

    # Calulate the win ratio and Gain to Loss ratio
    up = returnsframe[returnsframe > 0].count() / returnsframe.count()
    # down = returnsframe[returnsframe < 0].count() / returnsframe.count()
    average_up = returnsframe[returnsframe > 0].mean()
    average_down = returnsframe[returnsframe < 0].mean()
    gain_to_loss = (average_up) / (-1 * average_down)

    #MAR ratio, annualised return over MDD. Higher the better
    mar_ratio = abs(AnnReturns / [mdd[0][i] for i in range(len(mdd[0]))])

    # Annualisec return over average annual DD
    sterling_ratio = abs(AnnReturns / [mdd[0][i] for i in range(len(mdd[0]))])

    #Information ratio
    IR_df = pd.DataFrame({c: returnsframe[c] - returnsframe.bmSPY for c in returnsframe.columns})
    IR_df = IR_df[['cashModel','noCashModel','portfolio','bmSPY','EW']]
    info_ratio = (IR_df.mean() / IR_df.std()).fillna(0)

    metric_df = pd.DataFrame(AnnReturns.values.tolist(), index = ['AnnRet(%)','AnnRisk(%)','AnnSharpe','Avg_DD(%)','MaxDD(%)','WinRate(%)','Gain_to_Loss','RoMDD','Sortino(5%)',
                                                                  'Sterling_Ratio(over MDD)','beta','ann_alpha','R_squared','p_value', 'tvalue','info_ratio',
                                                                  '1YrReturns', '1YrRisk','3YrReturns', '3YrRisk', '5YrReturns', '5YrRisk','10YrReturns','10YrRisk'],
                                                                    columns = ['cashModel','noCashModel','portfolio','bmSPY','EW'])
    metric_df.loc['AnnRet(%)'] = round(metric_df.loc['AnnRet(%)'], 3)*100
    metric_df.loc['AnnRisk(%)'] = [100*i for i in AnnRisk]
    metric_df.loc['AnnSharpe'] = AnnSharpe.values.tolist()[0]
    metric_df.loc['Avg_DD(%)'] =  [round(float(i), 3) for i in [abs(i) * 100 for i in dd[0]]]
    metric_df.loc['MaxDD(%)'] = [round(float(i), 3) for i in [abs(i) * 100 for i in mdd[0]]]
    metric_df.loc['WinRate(%)'] = [100* round(float(i), 3) for i in up.values.tolist()]
    metric_df.loc['Gain_to_Loss'] = [round(float(i),3) for i in gain_to_loss.values.tolist()]
    metric_df.loc['RoMDD'] = [round(abs(i),3) for i in mar_ratio.values.tolist()[0]]
    metric_df.loc['Sortino(5%)'] = sortino_ratio.values.tolist()[0]
    metric_df.loc['Sterling_Ratio(over MDD)'] = [round(abs(i),3) for i in sterling_ratio.values.tolist()[0]]
    metric_df.loc['info_ratio'] = [i for i in info_ratio.values.tolist()]
    metric_df.loc['1YrReturns'] = [i*100.00 for i in ret_12m[0]]
    metric_df.loc['1YrRisk'] = [100 * i for i in std_12m.values.tolist()]
    metric_df.loc['3YrReturns'] = [i*100.00 for i in ret_36m[0]]
    metric_df.loc['3YrRisk'] = [100 * i for i in std_36m.values.tolist()]
    metric_df.loc['5YrReturns'] = [i*100.00 for i in ret_60m[0]]
    metric_df.loc['5YrRisk'] = [100 * i for i in std_60m.values.tolist()]
    metric_df.loc['10YrReturns'] = [i*100.00 for i in ret_120m[0]]
    metric_df.loc['10YrRisk'] = [100 * i for i in std_120m.values.tolist()]
    metric_df.loc['Total Return'] = returnsframe.cumsum()[-1:].values[0].tolist()
    return metric_df

def cash_scaling_model():
    # cs_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/adj_close_v2.csv", index_col='Date', parse_dates=True)[['IVV', 'BIL']]
    # cs_df = cs_df.resample('BM', closed='right').last()
    #
    # roll_win = 7
    #
    # rolling_12m = cs_df / cs_df.shift(roll_win) - 1
    #
    # rolling_12m['avg_er'] = cs_df['IVV'] - cs_df['IVV'].rolling(roll_win).mean()
    # rolling_12m.dropna(inplace=True)
    #
    # rolling_12m['ER'] = rolling_12m.IVV - rolling_12m.BIL
    #
    # rolling_12m['c_exc_ret'] = np.where(rolling_12m['ER'] > 0, 1, 0)
    #
    # rolling_12m['c_avg_ret'] = np.where(rolling_12m['avg_er'] > 0, 1, 0)
    #
    # rolling_12m['compp'] = rolling_12m['c_exc_ret'] + rolling_12m['c_avg_ret']
    #
    # c1 = rolling_12m['compp'] >= 2
    # # c2 = rolling_12m['compp'] > 0
    # # c3 = rolling_12m['compp'] < 2
    #
    # #use this to for a 3 way model, cash, nocash and BIL
    # # rolling_12m['composite'] = np.where(c1,1,np.where(c2&c3,0.5,0))
    #
    # rolling_12m['composite'] = np.where(c1, 1, 0)
    #reading the month end 3 month T-Bill rate from FRED DBT3
    ir_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/interest_rates_fred.csv", index_col='DATE', parse_dates=True)
    ir_df[ir_df['DTB3'] == '.'] = np.nan
    ir_df = ir_df.astype(float) * .01
    ir_df.fillna(method='ffill', inplace=True)

    #create and index for 3 month treasury bill
    ir_df['MV'] = 100.0
    ir_df['MV2'] = ir_df['MV'].shift(1) * (1 + ir_df['DTB3'])

    # reading the S&P 500 price data
    sp_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/sp500_yahoo.csv", index_col='Date', parse_dates=True)[['Adj Close']]
    # appending the T-Bill Index to the S&P 500 Dataframe
    sp_df['BIL'] = ir_df['MV2']
    # Resampling the S&P 500 dataframe to align with month end dates
    cs_df = sp_df.resample('BM', closed='right').last()
    # calculating the 6 months change for S&P 500 and TBil
    roll_win = 6
    rolling_12m = cs_df / cs_df.shift(roll_win) - 1

    # Condition 1 - Last month close greater 6 months average close
    rolling_12m['avg_er'] = cs_df['Adj Close'] - cs_df['Adj Close'].rolling(roll_win).mean()
    rolling_12m.dropna(inplace=True)

    #condition 2 -  Last 6 months returns of SP500 in excess of Excess 6 month Tbil returns
    rolling_12m['ER'] = rolling_12m['Adj Close'] - rolling_12m.BIL

    rolling_12m['c_exc_ret'] = np.where(rolling_12m['ER'] > 0, 1, 0)

    rolling_12m['c_avg_ret'] = np.where(rolling_12m['avg_er'] > 0, 1, 0)

    rolling_12m['compp'] = rolling_12m['c_exc_ret'] + rolling_12m['c_avg_ret']

    c1 = rolling_12m['compp'] >= 2

    rolling_12m['composite'] = np.where(c1, 1, 0)

    rolling_12m.to_csv("C:/Python27/Git/SMA_GTAA/Sectors/cashscaler.csv")

    #Plot the risk on and risk off signals
    dts = rolling_12m[rolling_12m['composite'] == 0].index

    ls = [i for i in range(len(dts))]
    for i in ls:
        #     p = plt.axvspan(dts[i],dts[i+1], facecolor='r', alpha=0.3)
        plt.axvline(x=dts[i])
        plt.axis([dts[0], dts[-1], -1, 1])

    sp_df['Adj Close'].resample('BM', closed='right').last().pct_change().cumsum().plot(color = 'r')
    # plt.legend(['SP500'])
    plt.title("Risk On/OFF Indicator vs S&P500 PR", color='crimson',fontweight='heavy')
    plt.savefig("C:/Python27/Git/SMA_GTAA/Sectors/cashScalingPlot.jpg", transparent=True)
    plt.show()

    # Rolling IC between the 1 month Sp500 fwd returns and composite signal
    # rolling_12m['change'] = cs_df['Adj Close'].pct_change()
    # X = rolling_12m['change'].shift(-1)
    # Y = rolling_12m['composite']
    # print(X.corr(Y))
    # X.rolling(roll_win).corr(Y.rolling(roll_win)).fillna(0).plot()
    # plt.grid()
    # plt.title("Rolling IC - Monthly")
    # plt.show()

    return rolling_12m

def startegy_switch():

    df_cash = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/returns_cash.csv", index_col=[0], parse_dates=True)

    df_cash = df_cash.rename(columns={'Average': 'cashModel'})

    df_nocash = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/returns_nocash.csv", index_col=[0], parse_dates=True)

    df_nocash = df_nocash.rename(columns={'Average': 'noCashModel'})

    df_cashscaler = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/cashscaler.csv", index_col=[0], parse_dates=True)

    df_combined = pd.concat([df_cash, df_nocash['noCashModel']], axis=1)

    df_combined = pd.concat([df_combined, df_cashscaler['composite']], axis=1)

    df_combined.dropna(inplace=True)

    df_combined['newComp'] = df_combined['composite'].shift(1)

    df_combined['portfolio'] = (df_combined['noCashModel'] * df_combined['newComp']) + ((1 - df_combined['newComp']) * df_combined['cashModel'])

    df_combined = df_combined[['cashModel', 'noCashModel', 'portfolio', 'bmSPY', 'EW']]

    return df_combined

def dollar_retuns(data):
    data.loc[:,'Portfolio'] = 10000
    data.loc[:,'S&P 500'] = 10000
    for i in range(len(data)):
        if i == 0:
            data['Portfolio'].iloc[i] = 10000
            data['S&P 500'].iloc[i] = 10000
        else:
            data['Portfolio'].iloc[i] = (1 + data['portfolio'].iloc[i]) * data['Portfolio'].iloc[i - 1]
            data['S&P 500'].iloc[i] = (1 + data['S&P500'].iloc[i]) * data['S&P 500'].iloc[i - 1]
    return data[['Portfolio','S&P 500']]

if __name__ == "__main__":

    # universe list for the model
    universe_list = ['XLRE', 'XLB', 'XLI', 'XLY', 'XLP', 'XLE', 'XLF', 'XLU', 'XLV', 'XLK', 'XLC', 'BIL', 'SHY','SPY']

    # Universe Adj.Close dataframe
    df = pd.DataFrame({s:pull_data(s) for s in universe_list})
    df.to_csv("C:/Python27/Git/SMA_GTAA/Sectors/adj_close_sectors.csv")

    adjusted_price = read_price_file('BM')
    modBiL = adjusted_price.BIL.pct_change()

    # read_price_file('BM')
    # n1 = 0.5
    # n2 = 0.5
    # n3 = 0.0
    # n4 = 0.0

    def rotation_models(w, mod):

        print("*****************Strategy with %s*************************" %mod)
        # model, wts = model_portfolios(cut_off=0.2, wList=[n1,n2,n3,n4])
        model, wts,eqPort = model_portfolios(cut_off=0.2, wList=w, mod=mod)
        model['EW'] = eqPort.mean(axis=1)

        # model = model['2014':]

        # Remove the first row with NaN's
        # portfolio_returns = portfolio_returns[1:]

        portfolio_returns = model[['Average','bmSPY','EW']]
        portfolio_returns.to_csv("C:/Python27/Git/SMA_GTAA/Sectors/returns_"+mod+".csv")
        wts.to_csv("C:/Python27/Git/SMA_GTAA/Sectors/weights_" + mod + ".csv")
        return wts

    # for no shy best weights is [0.0,0.0,0.7,0.3] and with shy best is [0.0,0.0,0.3,0.7]
    nocash_df = pd.DataFrame(rotation_models([0.0, 0.0, 0.7, 0.3], mod='nocash'))
    cash_df = pd.DataFrame(rotation_models([0.0, 0.0, 0.3, 0.7], mod='cash')) #.3/.7
    cs_model = cash_scaling_model()

    fig = plt.figure()

    if cs_model['composite'][-1:][0] == 1:
         print("Market Pulse : RISK ON")
         print("Recommended Trades ",nocash_df.iloc[-1].dropna())
         print(nocash_df.tail())
         x = nocash_df[-1:].dropna(axis=1).values
         y = list(nocash_df[-1:].dropna(axis=1).columns)
         plt.pie(x[0], labels=y, shadow=False, startangle=90,autopct='%1.1f%%')
         # plt.legend(loc='best')
         plt.title("Allocations as of %s" %str(today),fontweight = 'heavy',color='darkblue')
         plt.savefig("C:/Python27/Git/SMA_GTAA/Sectors/pie.jpg", facecolor=fig.get_facecolor(), transparent=True)
         plt.show()

    else:
         print("Market Pulse : RISK OFF")
         print("Recommended Trades ", cash_df.iloc[-1].dropna())
         print(cash_df.tail())
         x = cash_df[-1:].dropna(axis=1).values
         y = list(cash_df[-1:].dropna(axis=1).columns)
         plt.pie(x[0], labels=y, shadow=False, startangle=90,autopct='%1.1f%%')
         # plt.legend(loc='best')
         plt.title("Allocations as of %s" %str(today),fontweight = 'heavy')
         plt.savefig("C:/Python27/Git/SMA_GTAA/Sectors/allocations.jpg", facecolor=fig.get_facecolor(), transparent=True,color='darkblue')
         plt.show()

    all_portfolios = startegy_switch()
    all_portfolios.to_csv("C:/Python27/Git/SMA_GTAA/Sectors/portfolio_returns.csv")

    # BackTest Statistics for all the portfolios and indexes
    stats_df = backtest_metrics(all_portfolios, rfr=modBiL)

    # Regression stats for all portfolios and indices
    for c in stats_df.columns:

        stats_df[c].loc[['beta','ann_alpha','R_squared','p_value','tvalue']] = regression_fit(all_portfolios[c], all_portfolios.bmSPY.fillna(0), all_portfolios.RFR.fillna(0))

    all_portfolios.drop(['RFR'], inplace= True, axis = 1)
    stats_df.loc['Best_Month', :] = [100 * float(i) for i in all_portfolios.max().values.tolist()]
    stats_df.loc['Worst_Month', :] = [100 * float(i) for i in all_portfolios.min().values.tolist()]
    stats_df.loc['Best_Year', :] = [100 * float(i) for i in all_portfolios.groupby(all_portfolios.index.year).sum().max()]
    stats_df.loc['Worst_Year', :] = [100 * float(i) for i in all_portfolios.groupby(all_portfolios.index.year).sum().min()]

    # portfolio_returns = portfolio_returns[['Average','bmSPY','EW']]
    # print(100 * all_portfolios.groupby(all_portfolios.index.year).sum())
    all_portfolios.rename(columns = {'bmSPY': 'S&P500'}, inplace=True)
    return_by_year = all_portfolios.add(1).cumprod().groupby(all_portfolios.index.year).last().pct_change()
    return_by_year.to_csv("C:/Python27/Git/SMA_GTAA/Sectors/returns_by_year.csv")
    print(100 *return_by_year)
    print(100 * all_portfolios['2019'].groupby(all_portfolios['2019'].index.month).sum())
    # print(100 * np.sqrt(12) *all_portfolios.groupby(all_portfolios.index.year).std())
    # print(100 * all_portfolios)

    # Bar plot for portfolio and SP500
    plot_perf = 100 * all_portfolios[['portfolio','S&P500']]['2002':].add(1).cumprod().groupby(all_portfolios[['portfolio','S&P500']]['2002':].index.year).last().pct_change()
    plot_perf.plot(kind ='bar')
    plt.grid()
    plt.legend()
    plt.ylabel("in percentage(%)")
    plt.title("Portfolio Net Perfomance vs. S&P 500 PR",color='blue',fontweight='heavy')
    plt.savefig("C:/Python27/Git/SMA_GTAA/Sectors/bar_chart.jpg", facecolor=fig.get_facecolor(), transparent=True)
    plt.show()

    # Portfolio Return Plot
    dollar_returns = dollar_retuns(all_portfolios[['portfolio','S&P500']])
    dollar_returns.plot(linewidth = 2.0)
    plt.legend()
    plt.grid()
    plt.ylabel("in dollars($)")
    plt.title("Growth of a $10000 Portfolio ",color='blue',fontweight='heavy')
    plt.savefig("C:/Python27/Git/SMA_GTAA/Sectors/equity_curve.jpg", facecolor=fig.get_facecolor(), transparent=True)
    plt.show()

    # all_portfolios['07-2017':].cumsum().plot()
    # plt.legend()
    # plt.grid()
    # plt.title("Equity Curve - YTD")
    # plt.show()
    #
    # correlation Plot
    # plt.matshow(all_portfolios.corr())
    # plt.xticks(range(len(all_portfolios.columns)), all_portfolios.columns)
    # plt.yticks(range(len(all_portfolios.columns)), all_portfolios.columns)
    # plt.colorbar()
    # plt.title("Strategy Correlation Box")
    # plt.show()
    #
    # Rolling Correlation vs Benchamrk
    tcor = all_portfolios['portfolio'].rolling(window=6).corr(other=all_portfolios['S&P500'])
    tcor.plot(linewidth=1.5,color='red')
    plt.grid()
    plt.title("Rolling Correlation vs S&P 500",color='blue',fontweight='heavy')
    plt.savefig("C:/Python27/Git/SMA_GTAA/Sectors/correlation.jpg", facecolor=fig.get_facecolor(), transparent=True)
    plt.show()
    #
    #Distribution of returns
    # all_portfolios['portfolio'].hist()
    # plt.show()
    # #saving files
    stats_df.to_csv("C:/Python27/Git/SMA_GTAA/Sectors/Summary_Statistics.csv")
    print(stats_df)



#   cutt_off=0.2
#   stats_df.to_csv("C:/Python27/Git/SMA_GTAA/Sectors/"+str(n1)+"_"+str(n2)+"_"+str(n3)+"_"+str(n4)+"_"+str(cutt_off)+".csv")
#   print(stats_df)


    #Plot the rolling weights
    # wts = wts['2002-12':]  #
    # labels= wts.columns
    # y = np.vstack([wts[c].fillna(0) for c in wts.columns])
    # plt.stackplot(wts.index, y, labels = labels)
    # # wts.fillna(0).rolling(12).mean().plot()
    # plt.legend()
    # plt.grid()

# wlist = [[.25, .25, .25, .25], [0.5,0.5,0.0,0.0], [0.0, 0.5, 0.25, 0.25], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.8, 0.2], [0.0, 0.0, 0.2, 0.8],
#          [0.0, 0.0, 0.7, 0.3], [0.0, 0.0, 0.3, 0.7], [0.0, 0.5, 0.0, 0.5], [0.0, 0.5, 0.5, 0.0],[0.0,0.0,0.9,0.1],[0.0,0.0,0.1,0.9],[0.3,0.0,0.7,0.0]]
# t1 = pd.DataFrame()
# for w in wlist:
#     df_test = pd.DataFrame(test_por(w))
#     df_test.loc['model', :] = str(w)
#     t1 = pd.concat([t1, df_test], axis=1)
#
# t1.to_csv("C:/Python27/Git/SMA_GTAA/Sectors/mod_comp.csv")


