import pandas as pd
import numpy as np
import json
import requests
import time
import datetime
import matplotlib.pyplot as plt

now = datetime.datetime.now().strftime('%Y-%m-%d')
now = pd.to_datetime(now)
now = now.strftime('%Y-%m-%d')


class Strategy_Backtest():

    def __init__(self, sector):

        self.sector = sector
        self.price_frame = pd.read_csv(
            "C:/Python27/Git/SMA_GTAA/Stock_Selection/" + self.sector + "/SPY_" + self.sector + ".csv"
            , index_col='Date', parse_dates=True)

    def alpha_vantage_api(self, symbol):
        self.symbol = symbol
        print(self.symbol)
        response = requests.get(
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=" + self.symbol +
            "&outputsize=full&apikey=15TKBJQWRQ4PRD7D")
        todos = json.loads(response.text)
        time.sleep(15)
        return pd.DataFrame(todos['Time Series (Daily)']).T['5. adjusted close']

    def call_alphavantage_api(self):
        tickers_df = pd.read_csv("C:\Python27\Git\SMA_GTAA\Stock_Selection\SPY_All_Holdings.csv")
        sector_list = tickers_df.Sector.unique()
        sector_list = ['Energy']
        for sec in sector_list:
            if sec == 'Communication Services':
                bm_ticker = 'XLC'
                filename = sec.replace(" ", "")
            elif sec == 'Consumer Discretionary':
                bm_ticker = 'XLY'
                filename = sec.replace(" ", "")
            elif sec == 'Consumer Staples':
                bm_ticker = 'XLP'
                filename = sec.replace(" ", "")
            elif sec == 'Energy':
                bm_ticker = 'XLE'
                filename = sec.replace(" ", "")
            elif sec == 'Financials':
                bm_ticker = 'XLF'
                filename = sec.replace(" ", "")
            elif sec == 'Health Care':
                bm_ticker = 'XLV'
                filename = sec.replace(" ", "")
            elif sec == 'Industrials':
                bm_ticker = 'XLI'
                filename = sec.replace(" ", "")
            elif sec == 'Information Technology':
                bm_ticker = 'XLK'
                filename = sec.replace(" ", "")
            elif sec == 'Materials':
                bm_ticker = 'XLB'
                filename = sec.replace(" ", "")
            elif sec == 'Real Estate':
                bm_ticker = 'XLRE'
                filename = sec.replace(" ", "")
            elif sec == 'Utilities':
                bm_ticker = 'XLU'
                filename = sec.replace(" ", "")
            else:
                print("Error occured in call_alphavantage_api sector list, check the sectors in the file")

            it_symbols = tickers_df[tickers_df.Sector == sec]['Identifier']
            it_symbols = it_symbols.to_list()
            it_symbols.append(bm_ticker)
            it_symbols = [s.replace('.', '-') for s in it_symbols]
            df = pd.DataFrame({s: self.alpha_vantage_api(s) for s in it_symbols})
            df.index.name = 'Date'
            df.to_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/" + filename + "/SPY_" + filename + ".csv")
            return "Completed"

    def sampled_data(self, res_period):
        self.res_period = res_period
        self.resampled_frame = self.price_frame.resample(self.res_period, closed='right').last()
        return self.resampled_frame

    def sampled_returns_momo(self, retFrame, long_per=12, short_per=1):
        self.retFrame = retFrame
        self.long_per = long_per
        self.short_per = short_per
        self.sampled_rf = self.retFrame.pct_change(self.long_per) \
            .subtract(self.retFrame.pct_change(self.short_per))
        return self.sampled_rf

    def excess_returns(self, retFrame, bm_ticker):
        self.retFrame = retFrame
        self.bm_ticker = bm_ticker
        self.bm_frame = self.retFrame[self.bm_ticker]
        self.excess_ret = self.retFrame.subtract(self.bm_frame, axis=0)
        self.excess_ret.drop(self.bm_ticker, axis=1, inplace=True)
        return self.excess_ret

    def winLoss(self, dseries):
        self.dseries = dseries
        return (dseries > 0).sum()

    def returns_persistence(self, excRet, window=12):
        self.window = window
        self.excRet = excRet
        min_periods = round(self.window / 2.0)
        return self.excRet.rolling(self.window, min_periods=min_periods).apply(self.winLoss, raw=True)

    def qcut_grouping(self, series, bins=5):
        self.series = series
        self.bins = bins
        self.labels = ['Q' + str(b) for b in range(1, self.bins + 1)]
        self.bin_df = pd.qcut(self.series, self.bins, labels=self.labels, retbins=False, duplicates='drop')
        return self.bin_df

    def bucket_signal(self, buckets=5):
        self.buckets = buckets
        steps = 1. / self.buckets
        bins = np.arange(0, 1, steps)
        self.labels = ['Q' + str(b) for b in range(1, len(bins) + 1)]

    def qcut(self, s, q=5):
        self.q = q
        self.s = s
        labels = ['q{}'.format(i) for i in range(1, self.q + 1)]
        return pd.qcut(self.s, self.q, labels=labels)

    def quintile_returns(self, returns, cut_frame, qcut=5):
        self.returns = returns
        self.cut_frame = cut_frame
        self.qcut = qcut
        cols = ['q{}'.format(i) for i in range(1, self.qcut + 1)]
        #        cols = list(set(self.cut_frame.values.unique()))
        cols.sort()
        bucket_returns = pd.DataFrame(index=self.returns.index, columns=cols)
        fwd_ret = self.returns.shift(-1)
        cut_frame = cut_frame.unstack()
        bucket_returns = bucket_returns[cut_frame.index[0]:]
        fwd_ret = fwd_ret[cut_frame.index[0]:]
        r = {}
        for c in cols:
            print(c)
            t = fwd_ret[self.cut_frame == c].mean(axis=1)
        return bucket_returns.shift(1)

    def max_dd(self, returns):
        """returns is a series"""
        self.returns = returns
        r = self.returns.add(1).cumprod()
        dd = r.div(r.cummax()).sub(1)
        mdd = dd.min()
        end = dd.idxmin()
        start = r.loc[:end].argmax()
        return mdd, start, end

    def max_dd_df(self, returns):

        """returns is a dataframe"""
        self.returns = returns
        series = lambda x: pd.Series(x, ['Draw Down', 'Start', 'End'])
        return self.returns.apply(self.max_dd).apply(series)

    def frequency_of_time_series(self, df):
        self.df = df
        start, end = self.df.index.min(), self.df.index.max()
        delta = end - start
        return round((len(self.df) - 1.) * 365.25 / delta.days, 2)

    def annualized_return(self, df):
        self.df = df
        freq = self.frequency_of_time_series(self.df)
        return self.df.add(1).prod() ** (1 / freq) - 1

    def annualized_volatility(self, df):
        self.df = df
        freq = self.frequency_of_time_series(self.df)
        return self.df.std().mul(freq ** .5)

    def sharpe_ratio(self, df):
        self.df = df
        return self.annualized_return(df) / self.annualized_volatility(df)

    def describe(self, df):
        self.df = df
        r = self.annualized_return(self.df).rename('Return')
        v = self.annualized_volatility(self.df).rename('Volatility')
        s = self.sharpe_ratio(self.df).rename('Sharpe')
        skew = self.df.skew().rename('Skew')
        kurt = self.df.kurt().rename('Kurtosis')
        desc = self.df.describe().T
        return pd.concat([r, v, s, skew, kurt, desc], axis=1).T.drop('count')


def factor_analysis(bm, sector, long_per=12, short_per=1, pers_win=6, qcut=5):
    qcut = qcut

    bm = bm

    backtest = Strategy_Backtest(sector=sector)

    rs = backtest.sampled_data('BM')

    sampled_return_frame = backtest.sampled_returns_momo(retFrame=rs, long_per=long_per, short_per=short_per)

    excess_over_bm = backtest.excess_returns(retFrame=sampled_return_frame, bm_ticker=bm)

    exc_ret_rank = excess_over_bm.rank(axis=1, method='average', numeric_only=True, ascending=False, pct=False)

    persistent_exc_ret = backtest.returns_persistence(excess_over_bm, window=pers_win)

    pers_rank = persistent_exc_ret.rank(axis=1, method='average', numeric_only=True, ascending=False, pct=False)

    combined_rank = (exc_ret_rank + pers_rank) / 2

    combined_rank = combined_rank['2016':]

    cut_frame = combined_rank.stack().groupby(level=0).apply(backtest.qcut, qcut)

    # shifting one month backwards returns to align with previous month position
    returns = rs.pct_change().shift(-1)
    bm_rets = returns[bm].shift(1)
    unstack_frame = cut_frame.unstack()
    cols = ['q{}'.format(i) for i in range(1, qcut + 1)]

    # holding dataframe
    q10_holdings = unstack_frame[unstack_frame == 'q10']
    q10_holdings.to_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/" + sector + "/" + sector + "_holdings.csv")
    print(q10_holdings.stack())

    returns = pd.DataFrame({s: returns[unstack_frame == s].mean(axis=1) for s in cols})

    # shifting the bucket returns one month forward to align the index
    returns = returns.shift(1)
    returns.to_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/" + sector + "/" + sector + "_returns.csv")

    ret_with_bm = returns
    ret_with_bm.loc[:, bm] = bm_rets['2016':]
    print(ret_with_bm.groupby(ret_with_bm.index.year).sum().dropna())
    fot = backtest.frequency_of_time_series(ret_with_bm)
    ar = backtest.annualized_return(ret_with_bm)
    avol = backtest.annualized_volatility(ret_with_bm)
    sr = backtest.sharpe_ratio(ret_with_bm)
    des = backtest.describe(ret_with_bm)
    print(des)

    draw_downs = backtest.max_dd_df(ret_with_bm)
    print(draw_downs)
    fig, axes = plt.subplots(10, 1, figsize=(12, 20))
    for i, ax in enumerate(axes[::-1]):
        returns.iloc[:, i].add(1).cumprod().plot(ax=ax)
        sd, ed = draw_downs[['Start', 'End']].iloc[i]
        ax.axvspan(sd, ed, alpha=0.1, color='r')
        ax.set_ylabel(returns.columns[i])

    fig.suptitle('Maximum Draw Down', fontsize=18)
    fig.tight_layout()
    plt.subplots_adjust(top=.95)

    top = returns.q10
    delta = returns.subtract(top, axis=0)
    delta.drop(bm, axis=1, inplace=True)
    median = delta.median()
    mean = delta.mean()

    returns.add(1).cumprod().plot(figsize=(15, 6))
    plt.grid()
    plt.legend()
    dframe = pd.DataFrame({'Mean': mean, 'Median': median}, index=mean.index)
    dframe.plot(kind='bar', figsize=(15, 6))
    plt.grid()
    plt.legend()
    plt.show()
    return returns


def sector_bucket_returns(sector):
    sec_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/" + sector + "/" + sector + "_returns.csv",
                         index_col='Date', parse_dates=True)
    return sec_df['q10']


if __name__ == "__main__":

    #    backtest = Strategy_Backtest()
    #
    #    state = backtest.call_alphavantage_api()

    def run_factor_analysis():

        qcut = 10

        ##        q10
        fact_ret_energy = factor_analysis(bm='XLE', sector='Energy', long_per=6, short_per=1, pers_win=6, qcut=qcut)
        #
        ##        q10 - q2 outperforms and has betteer statistics. Needs more work.
        fact_ret_health = factor_analysis(bm='XLV', sector='HealthCare', long_per=6, short_per=1, pers_win=3, qcut=qcut)
        #
        ##        q10
        fact_ret_technology = factor_analysis(bm='XLK', sector='InformationTechnology', long_per=18, short_per=1,
                                              pers_win=18, qcut=qcut)
        #
        ##        q9 : q9 is better in returns and dd. potential improvement
        fact_ret_industrials = factor_analysis(bm='XLI', sector='Industrials', long_per=12, short_per=1, pers_win=6,
                                               qcut=qcut)
        #
        ##        q9: q9 is slightly better in terms of returns, q10 sharpe for best use 15 for long and pers
        fact_ret_financials = factor_analysis(bm='XLF', sector='Financials', long_per=12, short_per=1, pers_win=12,
                                              qcut=qcut)
        #
        ##        q10
        fact_ret_condisc = factor_analysis(bm='XLY', sector='ConsumerDiscretionary', long_per=12, short_per=1,
                                           pers_win=12, qcut=qcut)
        #
        ##        q10
        fact_ret_comm = factor_analysis(bm='XLC', sector='CommunicationServices', long_per=3, short_per=1, pers_win=3,
                                        qcut=qcut)
        #
        ##        q10 : big DD vs q2, for best use long = 12, pers = 15
        fact_ret_constpl = factor_analysis(bm='XLP', sector='ConsumerStaples', long_per=12, short_per=1, pers_win=12,
                                           qcut=qcut)
        #
        ##        q10
        fact_ret_materials = factor_analysis(bm='XLB', sector='Materials', long_per=12, short_per=1, pers_win=12,
                                             qcut=qcut)
        #
        ##        q10
        fact_ret_realestate = factor_analysis(bm='XLRE', sector='RealEstate', long_per=12, short_per=1, pers_win=12,
                                              qcut=qcut)
        #
        ##         q10, for best result use long = 6, pers = 6
        fact_ret_utilities = factor_analysis(bm='XLU', sector='Utilities', long_per=12, short_per=1, pers_win=12,
                                             qcut=qcut)


    # run factor analysis first before genrating trade reco for the month.
    #    run_factor_analysis()

    sector_list = ['CommunicationServices', 'ConsumerDiscretionary', 'ConsumerStaples', 'Energy', 'Financials',
                   'HealthCare', 'Industrials',
                   'InformationTechnology', 'Materials', 'RealEstate', 'Utilities']
    start = '2016-12'
    end = '2019-06'
    returns_df = pd.DataFrame({s: sector_bucket_returns(s) for s in sector_list}, index=None)

    cash_scaler = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/cashscaler.csv", index_col='Date', parse_dates=True)
    cash_scaler.index.name = 'Date'
    cash_scaler = cash_scaler.resample('BM', closed='right').last()

    wts_cash = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/weights_cash.csv", index_col=[0], parse_dates=True)
    wts_cash.index.name = 'Date'
    wts_cash = wts_cash.resample('BM', closed='right').last()

    wts_no_cash = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/weights_nocash.csv", index_col=[0], parse_dates=True)
    wts_no_cash.index.name = 'Date'
    wts_no_cash = wts_no_cash.resample('BM', closed='right').last()

    sec_returns = pd.read_csv("C:/Python27/Git/SMA_GTAA/Sectors/adj_close_sectors.csv", index_col='Date',
                              parse_dates=True)

    shy_ret = sec_returns['SHY'].resample('BM', closed='right').last().pct_change()

    returns_df['SHY'] = shy_ret

    returns_df = returns_df[start:end]
    cash_scaler = cash_scaler[start:end]
    wts_cash = wts_cash[start:end]
    wts_no_cash = wts_no_cash[start:end]

    sec_to_ticker_dict = {'CommunicationServices': 'XLC', 'ConsumerDiscretionary': 'XLY', 'ConsumerStaples': 'XLP',
                          'Energy': 'XLE',
                          'Financials': 'XLF', 'HealthCare': 'XLV', 'Industrials': 'XLI',
                          'InformationTechnology': 'XLK',
                          'Materials': 'XLB', 'RealEstate': 'XLRE', 'Utilities': 'XLU'}

    returns_df.rename(columns=sec_to_ticker_dict, inplace=True)

    portfolio_wts = pd.DataFrame(index=cash_scaler.index, columns=returns_df.columns)

    for k, v in cash_scaler.iterrows():
        if v['composite'] == 1:
            portfolio_wts.loc[k, :] = wts_no_cash.loc[k, :]
        else:
            portfolio_wts.loc[k, :] = wts_cash.loc[k, :]

    shifted_returns = returns_df.shift(-1)
    port_return = portfolio_wts.multiply(shifted_returns, axis=1).sum(axis=1).shift(1)
    sector_trade_reco = portfolio_wts[-1:].T.sort_values(by=portfolio_wts.index[-1]).dropna()

    ticker = sector_trade_reco.index.to_list()


    def _trades_reco_(df, sector):

        _noh_ = df[-1:].notnull().sum().sum()
        _wts_ = sector.loc[t].div(_noh_)
        _wts_ = [_wts_] * _noh_
        sym = df[-1:].notnull().sum()
        sym = sym[sym == 1].index.tolist()
        _dict_ = dict(zip(sym, _wts_))
        return _dict_


    dict_trades = {}
    for t in ticker:

        if t == 'XLC':
            xlc_df = pd.read_csv(
                "C:/Python27/Git/SMA_GTAA/Stock_Selection/CommunicationServices/CommunicationServices_holdings.csv",
                index_col='Date', parse_dates=True)
            xlc_df = xlc_df[start:end]
            dict_trades.append(_trades_reco_(xlc_df, sector_trade_reco))

        elif t == 'XLY':
            xly_df = pd.read_csv(
                "C:/Python27/Git/SMA_GTAA/Stock_Selection/ConsumerDiscretionary/ConsumerDiscretionary_holdings.csv",
                index_col='Date', parse_dates=True)
            xly_df = xly_df[start:end]
            dict_trades.update(_trades_reco_(xly_df, sector_trade_reco))

        elif t == 'XLP':
            xlp_df = pd.read_csv(
                "C:/Python27/Git/SMA_GTAA/Stock_Selection/ConsumerStaples/ConsumerStaples_holdings.csv",
                index_col='Date', parse_dates=True)
            xlp_df = xlp_df[start:end]
            dict_trades.update(_trades_reco_(xlp_df, sector_trade_reco))

        elif t == 'XLE':
            xle_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/Energy/Energy_holdings.csv",
                                 index_col='Date', parse_dates=True)
            xle_df = xle_df[start:end]
            dict_trades.update(_trades_reco_(xle_df, sector_trade_reco))

        elif t == 'XLF':
            xlf_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/Financials/Financials_holdings.csv",
                                 index_col='Date', parse_dates=True)
            xlf_df = xlf_df[start:end]
            dict_trades.update(_trades_reco_(xlf_df, sector_trade_reco))

        elif t == 'XLV':
            xlv_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/HealthCare/HealthCare_holdings.csv",
                                 index_col='Date', parse_dates=True)
            xlv_df = xlv_df[start:end]
            dict_trades.update(_trades_reco_(xlv_df, sector_trade_reco))


        elif t == 'XLI':
            xli_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/Industrials/Industrials_holdings.csv",
                                 index_col='Date', parse_dates=True)
            xli_df = xli_df[start:end]
            dict_trades.update(_trades_reco_(xli_df, sector_trade_reco))


        elif t == 'XLK':
            xlk_df = pd.read_csv(
                "C:/Python27/Git/SMA_GTAA/Stock_Selection/InformationTechnology/InformationTechnology_holdings.csv",
                index_col='Date', parse_dates=True)
            xlk_df = xlk_df[start:end]
            dict_trades.update(_trades_reco_(xlk_df, sector_trade_reco))


        elif t == 'XLB':
            xlb_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/Materials/Materials_holdings.csv",
                                 index_col='Date', parse_dates=True)
            xlb_df = xlb_df[start:end]
            dict_trades.update(_trades_reco_(xlb_df, sector_trade_reco))


        elif t == 'XLRE':
            xlre_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/RealEstate/RealEstate_holdings.csv",
                                  index_col='Date', parse_dates=True)
            xlre_df = xlre_df[start:end]
            dict_trades.update(_trades_reco_(xlre_df, sector_trade_reco))

        elif t == 'XLU':
            xlu_df = pd.read_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/Utilities/Utilities_holdings.csv",
                                 index_col='Date', parse_dates=True)
            xlu_df = xlu_df[start:end]
            dict_trades.update(_trades_reco_(xlu_df, sector_trade_reco))

        else:
            print("Error: ticker_list from sector_trade_reco is empty")

        to_trade = pd.DataFrame(dict_trades)
        to_trade = to_trade.applymap(lambda x: "{:.2%}".format(x))
        tickers_df = pd.read_csv("C:/Users/yprasad/Dropbox/SPY_All_Holdings.csv")
        tickers_df.set_index('Identifier', inplace=True)
        to_trade = to_trade.T
        idx = to_trade.index.tolist()
        for sec in idx:
            to_trade.loc[sec, 'Sector'] = tickers_df.loc[sec, 'Sector']
        print(to_trade)
        to_trade.to_html("C:/Python27/Git/SMA_GTAA/Stock_Selection/Trades_" + now + ".html", index_names=False,
                         justify='center')

#    q10_holdings.to_csv("C:/Python27/Git/SMA_GTAA/Stock_Selection/"+sector+"/"+sector+"_holdings.csv")


#    qcut=5
#    bm = 'XLE'
#    backtest = Strategy_Backtest(sector='Energy')
#    rs = backtest.sampled_data('BM')
#    sampled_return_frame = backtest.sampled_returns_momo(retFrame=rs, long_per = 12, short_per = 1)
#    excess_over_bm = backtest.excess_returns(retFrame = sampled_return_frame, bm_ticker = bm)
#    exc_ret_rank = excess_over_bm.rank(axis=1, method='average', numeric_only=True, ascending=False, pct=False)
#    persistent_exc_ret = backtest.returns_persistence(excess_over_bm, window=6)
#    pers_rank = persistent_exc_ret.rank(axis=1, method='average', numeric_only=True, ascending=False, pct=False)
#    combined_rank = (exc_ret_rank + pers_rank)/2
#    combined_rank = combined_rank['2001':]
#
#    cut_frame = combined_rank.stack().groupby(level=0).apply(backtest.qcut,qcut)
#    returns = rs.pct_change().shift(-1)
#    bm_rets = returns[bm].shift(1)
#    unstack_frame=cut_frame.unstack()
#    cols = ['q{}'.format(i) for i in range(1,qcut+1)]
#    q1_holdings = unstack_frame[unstack_frame=='q1']
#    returns = pd.DataFrame({s:returns[unstack_frame==s].mean(axis=1) for s in cols})
#    returns = returns.shift(1)
#
#    ret_with_bm = returns
#    ret_with_bm.loc[:, bm] = bm_rets['2001':]
#    fot = backtest.frequency_of_time_series(ret_with_bm)
#    ar = backtest.annualized_return(ret_with_bm)
#    avol = backtest.annualized_volatility(ret_with_bm)
#    sr = backtest.sharpe_ratio(ret_with_bm)
#    des = backtest.describe(ret_with_bm)
#    print(des)
#
#     draw_downs = backtest.max_dd_df(ret_with_bm)
#     print(draw_downs)
#     fig, axes = plt.subplots(5, 1, figsize=(12, 8))
#     for i, ax in enumerate(axes[::-1]):
#         returns.iloc[:, i].add(1).cumprod().plot(ax=ax)
#         sd, ed = draw_downs[['Start', 'End']].iloc[i]
#         ax.axvspan(sd, ed, alpha=0.1, color='r')
#         ax.set_ylabel(returns.columns[i])
#
#     fig.suptitle('Maximum Draw Down', fontsize=18)
#     fig.tight_layout()
#     plt.subplots_adjust(top=.95)
#
#    top = returns.q5
#    delta = returns.subtract(top,axis=0)
#    delta.drop(bm, axis=1, inplace=True)
#    median = delta.median()
#    mean = delta.mean()
#
#    returns.add(1).cumprod().plot(figsize=(15,6))
#    plt.grid()
#    plt.legend()
#    dframe = pd.DataFrame({'Mean':mean, 'Median':median}, index=mean.index)
#    dframe.plot(kind='bar', figsize=(15,6))
#     plt.grid()
#     plt.legend()
#     plt.show()
