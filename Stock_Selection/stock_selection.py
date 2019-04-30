# This model trades based on the 10 Month SMA rule rotating through different asset classes.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fix_yahoo_finance as yf
from scipy import stats
import datetime
import json
import requests
import statsmodels.api as sm
yf.pdr_override()  # <== that's all it takes :-)
pd.set_option('precision', 4)
pd.options.display.float_format = '{:.3f}'.format
import time

today = datetime.datetime.today().strftime('%m/%d/%y')

# Function to pull data from yahoo
def alphavantage_close_price(symbol):
    response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=" + symbol + "&outputsize=full&apikey=15TKBJQWRQ4PRD7D")
    todos = json.loads(response.text)
    time.sleep(15)
    return pd.DataFrame(todos['Time Series (Daily)']).T['5. adjusted close']


def read_price_file(frq='BM'):
    df_price = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/XLK/SPY_Tech_Stocks.csv", index_col='Date', parse_dates=True)
    df_price = df_price.resample(frq, closed='right').last()
    df_price = df_price['2001':]
    return df_price


def model_portfolios(wList=[0.25, 0.25, 0.25, 0.25]):
    periods='2013'
    df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/XLK/SPY_Tech_Stocks.csv", index_col='Date', parse_dates=True)
    df = df[periods:]
    # calculating the daily return for benchmarks
    rframe = df.resample('BM', closed='right').last().pct_change()
    bench_mark = rframe.XLK
    bmbil = rframe.BIL

    def clean_universe(df, rs='BM', per=1):
        # resampling price frame
        resamp_df = df.resample(rs, closed='right').last()

        # calculating the resampled price returns in excess of the benchmark
        ret_frame = resamp_df.pct_change(per)

        ret_frame1 = pd.DataFrame({s: ret_frame[s] - ret_frame.XLK for s in ret_frame.columns}, index=ret_frame.index)

        if per == 1:
            ret_frame2 = ret_frame1.rolling(12).mean().sub(ret_frame)
        else:
            ret_frame2 = ret_frame1.rolling(per).mean().sub(ret_frame)

        # calculating the daily returns
        riskChg = df.pct_change()

        # calculating the rolling std deviations and re-sample the df
        risk_df = riskChg.rolling(30).apply(np.std).resample(rs, closed='right').last()

        # returning the risk asdjusted return frames
        return ret_frame2 / risk_df

    # 1 month risk adjusted  frame
    df_1m = clean_universe(df, rs='BM', per=1)

    # Drop the benchmark from the return frame. Add symbols to the list to eliminate from the return frame
    rframe.drop(['XLK', 'BIL'], inplace=True, axis=1)

    df_1m.drop(['XLK', 'BIL'], inplace=True, axis=1)

    # 3 month risk adjusted frame
    df_3m = clean_universe(df, rs='BM', per=3)

    # 6 month risk adjusted  frame
    df_6m = clean_universe(df, rs='BM', per=6)

    # 12 month risk adjusted  frame
    df_12m = clean_universe(df, rs='BM', per=12)

    df.drop(['XLK', 'BIL'], inplace=True, axis=1)
    df_3m.drop(['XLK', 'BIL'], inplace=True, axis=1)
    df_6m.drop(['XLK', 'BIL'], inplace=True, axis=1)
    df_12m.drop(['XLK', 'BIL'], inplace=True, axis=1)

    # Zscore the risk adjusted return frames Cross Sectional

    zs_1m = pd.DataFrame([(df_1m.iloc[i] - df_1m.iloc[i].mean()) / df_1m.iloc[i].std() for i in range(len(df_1m))])
    zs_3m = pd.DataFrame([(df_3m.iloc[i] - df_3m.iloc[i].mean()) / df_3m.iloc[i].std() for i in range(len(df_3m))])
    zs_6m = pd.DataFrame([(df_6m.iloc[i] - df_6m.iloc[i].mean()) / df_6m.iloc[i].std() for i in range(len(df_6m))])
    zs_12m = pd.DataFrame([(df_12m.iloc[i] - df_12m.iloc[i].mean()) / df_12m.iloc[i].std() for i in range(len(df_12m))])

    zs_1m = zs_1m.clip(lower=-3.0, upper=3.0, axis=1)
    zs_3m = zs_3m.clip(lower=-3.0, upper=3.0, axis=1)
    zs_6m = zs_6m.clip(lower=-3.0, upper=3.0, axis=1)
    zs_12m = zs_12m.clip(lower=-3.0, upper=3.0, axis=1)

    def return_persistence(data):
        return ((data > 0).sum() - (data < 0).sum())

    # Generate the dataframe for Persistence return Factor from 1 month data frame

    persistence_short = df_1m.rolling(3).apply(return_persistence)
    persistence_inter = df_1m.rolling(6).apply(return_persistence)
    persistence_long = df_1m.rolling(12).apply(return_persistence)

    # composte frame for the long and short persistence factors use long wts  = 0.9 and short wts = 0.1 for less drawdown
    composite_persistence = 0.4 * persistence_short + 0.6 * persistence_inter + 0.0 * persistence_long

    # Generate the zscore of composite persistence dataframe Cross Sectional
    persistence_zscore = pd.DataFrame([(composite_persistence.iloc[i] - composite_persistence.iloc[i].mean()) / composite_persistence.iloc[i].std()
         for i in range(len(composite_persistence))])

    persistence_zscore = persistence_zscore.clip(lower=-3.0, upper=3.0, axis=1)

    # Genrate the composite zscore of return frames for different period returns frame
    rank_comp = wList[0] * zs_1m + wList[1] * zs_3m + wList[2] * zs_6m + wList[3] * zs_12m

    #quintle analysis
    q_cut = rank_comp.quantile(q=[0.2,0.4,0.6,0.8,0.9,1.0],axis=1,numeric_only=True).T

    def q_bucket(rankdf,qcut,q):
        if q==0.2:
            boolean_df = pd.DataFrame({s: rankdf[s] <=qcut[q] for s in rankdf.columns},index = rankdf.index)
            return rankdf[boolean_df]
        elif q==0.9:
            # q = round(q - 0.2, 1)
            boolean_df= pd.DataFrame({s: rankdf[s] > qcut[q] for s in rankdf.columns},index=rankdf.index)
            return rankdf[boolean_df]
        else:
            q_low=round(q-0.2,1)
            boolean_df= pd.DataFrame({s: ((rankdf[s] > qcut[q_low]) & (rankdf[s] <= qcut[q])) for s in rankdf.columns}, index = rankdf.index)
            return rankdf[boolean_df]

    q_one = q_bucket(rank_comp,q_cut,0.2)
    q_two = q_bucket(rank_comp, q_cut, 0.4)
    q_three = q_bucket(rank_comp, q_cut, 0.6)
    q_four = q_bucket(rank_comp, q_cut, 0.8)
    q_five = q_bucket(rank_comp, q_cut, 0.9)

    qone_pers = persistence_zscore[q_one.notnull()]
    qtwo_pers = persistence_zscore[q_two.notnull()]
    qthree_pers = persistence_zscore[q_three.notnull()]
    qfour_pers = persistence_zscore[q_four.notnull()]
    qfive_pers = persistence_zscore[q_five.notnull()]


    wts_one = pd.DataFrame([qone_pers.iloc[i] / abs(qone_pers.iloc[i]).sum() for i in range(len(qone_pers))]).abs()
    wts_two = pd.DataFrame([qtwo_pers.iloc[i] / abs(qtwo_pers.iloc[i]).sum() for i in range(len(qtwo_pers))]).abs()
    wts_three = pd.DataFrame([qthree_pers.iloc[i] / abs(qthree_pers.iloc[i]).sum() for i in range(len(qthree_pers))]).abs()
    wts_four = pd.DataFrame([qfour_pers.iloc[i] / abs(qfour_pers.iloc[i]).sum() for i in range(len(qfour_pers))]).abs()
    wts_five = pd.DataFrame([qfive_pers.iloc[i] / abs(qfive_pers.iloc[i]).sum() for i in range(len(qfive_pers))]).abs()

    # calculate the weights from the Sector Rotation weight model
    cash_scaler = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/cashscaler.csv", parse_dates=True, index_col=[0])
    cash_scaler = cash_scaler[periods:]
    sector_wts_cash = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/weights_cash.csv", parse_dates=True, index_col=[0])
    sector_wts_cash = sector_wts_cash[periods:]
    sector_wts_nocash = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/weights_nocash.csv", parse_dates=True, index_col=[0])
    sector_wts_nocash = sector_wts_nocash[periods:]

    # reindex weights dataframe to align with cash_scale time period,
    sector_wts_cash = sector_wts_cash.loc[cash_scaler.index[0]:]
    sector_wts_nocash = sector_wts_nocash.loc[cash_scaler.index[0]:]
    cash_scaled_sector_wts = pd.DataFrame(index=cash_scaler.index)
    cash_scaled_sector_wts['Sector_wt'] = np.where(cash_scaler.composite == 0, sector_wts_cash.XLK, sector_wts_nocash.XLK)
    # scaled weights,input : cash_scaler,sector_wts_cash,sector_wts_nocash,bm
    def final_weights_scaled_cash_model(quintileWeights, cashScaler=cash_scaler, cashScaledSecWts = cash_scaled_sector_wts):

        df_weights_q= quintileWeights.loc[cashScaler.index[0]:]
        cash_wts_q = df_weights_q.multiply(cashScaledSecWts['Sector_wt'], axis=0)
        final_wts_q = cash_wts_q[cash_wts_q.applymap(lambda x: x >= 0.00005)]
        final_wts_q= pd.DataFrame([final_wts_q.iloc[i] / abs(final_wts_q.iloc[i]).sum() for i in range(len(final_wts_q))]).abs()
        return final_wts_q

    final_wts_one = final_weights_scaled_cash_model(wts_one)
    final_wts_two = final_weights_scaled_cash_model(wts_two)
    final_wts_three = final_weights_scaled_cash_model(wts_three)
    final_wts_four = final_weights_scaled_cash_model(wts_four)
    final_wts_five = final_weights_scaled_cash_model(wts_five)
    #Quintile returns
    quint_ret_1 = rframe.shift(-1)[q_one.notnull()].multiply(final_wts_one)
    quint_ret_2 = rframe.shift(-1)[q_two.notnull()].multiply(final_wts_two)
    quint_ret_3 = rframe.shift(-1)[q_three.notnull()].multiply(final_wts_three)
    quint_ret_4 = rframe.shift(-1)[q_four.notnull()].multiply(final_wts_four)
    quint_ret_5 = rframe.shift(-1)[q_five.notnull()].multiply(final_wts_five)
    q5_ew = rframe.shift(-1)[q_five.notnull()].mean(axis=1)
    q1_ew = rframe.shift(-1)[q_one.notnull()].mean(axis=1)

    #quintile returns wtd average returns
    quintile_returns = pd.DataFrame(index=rank_comp.index)
    quintile_returns['q1'] = quint_ret_1.shift(1).sum(axis=1)
    quintile_returns['q2'] = quint_ret_2.shift(1).sum(axis=1)
    quintile_returns['q3'] = quint_ret_3.shift(1).sum(axis=1)
    quintile_returns['q4'] = quint_ret_4.shift(1).sum(axis=1)
    quintile_returns['q5'] = quint_ret_5.shift(1).sum(axis=1)
    quintile_returns['eq_wt5'] = q5_ew.shift(1)
    quintile_returns['eq_wt1'] = q1_ew.shift(1)
    quintile_returns['bench_mark'] = bench_mark
    quintile_returns.index = pd.to_datetime(quintile_returns.index)
    grouped = quintile_returns[periods:].groupby(quintile_returns[periods:].index.year).sum()
    print(grouped)

    # pd.plotting.scatter_matrix(quintile_returns, alpha=0.5, figsize=(8, 8), diagonal='hist')
    quintile_returns['2015':].add(1).cumprod().plot()

    #excess return
    q_delta = pd.DataFrame({s: quintile_returns[s] - quintile_returns['eq_wt5'] for s in quintile_returns.columns})
    grp_delta = q_delta.groupby(q_delta.index.year).mean()
    grp_delta.plot(kind='bar')
    plt.grid()
    plt.plot()
    plt.show()

    def max_dd(returns):
        """returns is a series"""
        r = returns.add(1).cumprod()
        dd = r.div(r.cummax()).sub(1)
        mdd = dd.min()
        # end = dd.argmin()
        # start = r.loc[:end].argmax()
        return mdd

    def max_dd_df(returns):
        """returns is a dataframe"""
        series = lambda x: pd.Series(x, ['Draw Down'])
        return returns.apply(max_dd).apply(series)

    print(max_dd(quintile_returns))


    return quintile_returns

if __name__ == "__main__":

    # tickers_df = pd.read_csv("C:/Users/yprasad/Dropbox/SPY_All_Holdings.csv")
    # sector_list = tickers_df.Sector.unique()
    # for sec in sector_list[6:7]:
    #
    #     sec_symbols = tickers_df[tickers_df.Sector == sec]['Identifier']
    #     sec_symbols = sec_symbols.tolist()
    #     df = pd.DataFrame({s: alphavantage_close_price(s) for s in sec_symbols})
    #     sec = sec.strip()
    #     df.to_csv("C:/Python27/Examples/SPY_"+sec+".csv")


    model = model_portfolios(wList=[0.0, 0.7, 0.3, 0.0])
    # wts.index.name = 'Date'
    # model['EW'] = eqPort.mean(axis=1)
    # portfolio_returns = model[['Average', 'bench_mark', 'EW', 'alt_Average']]
    # portfolio_returns.index.name = 'Date'
    # portfolio_returns.to_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/XLK/returns.csv")
    # wts.to_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/XLK//weights.csv")
