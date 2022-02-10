import streamlit as st
import psycopg2
from psycopg2 import sql
import pandas as pd
import numpy as np
import copy

from conn import api

@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None, 'builtins.weakref': lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["local"])


@st.cache(ttl=600,allow_output_mutation=True, hash_funcs={psycopg2.extensions.connection: lambda _: None})
def run_query(query, conn):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

@st.cache(allow_output_mutation=True, hash_funcs={psycopg2.extensions.connection: lambda _: None})
def get_portfolio_holding(benchmark_name, conn):

    # get benchmark weight
    weights = pd.read_sql(api.get_benchmark_weight(benchmark_name), con = conn) 
    weights = weights.groupby(by=["ISINCode"]).sum()  # incase there are duplciates ETFs in a portfolio

    
    # get benchmark holding dictionary
    etfs = pd.read_sql(api.get_etf_details_for_portfolio(weights.index), conn).set_index("ISINCode")

    etfs = etfs.drop_duplicates() # delete this once duplicated ETFs are resolved in a portfolio
    etfs = pd.concat([etfs, weights], axis=1).sort_values("Weight", ascending=False)

    holding_etfs = etfs.to_dict(orient='index') 
    #TODO: When converting from pandas to dictionary, Weight 3.5 converts into 3.50000001
    for k in holding_etfs:
        holding_etfs[k]['Weight'] = round(holding_etfs[k]['Weight'],2)

    # get benchmark rank
    ranks = {}
    rank_categories = ["Portfolio", "Cost", "Return", "AUM", "Tracking Error", "Volume", "Rank"]
    benchmark_ranks = run_query(api.get_benchmark_rank(benchmark_name), conn)[0]
    for i in range(1, len(benchmark_ranks)):
        ranks[rank_categories[i]] = benchmark_ranks[i]
    
    
    # get benchmark holding type
    benchmark_holding_type = run_query(api.get_benchmark_holding_type(benchmark_name), conn)
    benchmark_holding_top10 = run_query(api.get_benchmark_holding_top10(benchmark_name), conn)
    
    holdings = benchmark_holding_type + benchmark_holding_top10

    
    return holding_etfs, ranks, holdings


#@st.cache(hash_funcs={psycopg2.extensions.connection: lambda _: None})
def init_data(conn):
    # get benchmark portfolios
    benchmarks = pd.read_sql(api.get_benchmarks(), con = conn)
    benchmarks = benchmarks.rename(columns ={'count': '# Holdings', 'Rank5': 'Ranks', 'Return': '1 Month %'})

    # get benchmark risk & return
    benchmark_perf = pd.read_sql(api.get_benchmark_risk_return(), con=conn)

    benchmarks_names = benchmarks['Portfolio'].unique()

    default_benchmark = benchmarks_names[0]

    #get full ETF list
    etfs_all = run_query(api.get_unique_fund_list(), conn)
    #etfs_all.insert(0, [''])


    return benchmarks, benchmark_perf ,benchmarks_names, etfs_all



@st.cache(hash_funcs={psycopg2.extensions.connection: lambda _: None})
def calc_rank(isins, weights, conn):
    ranks = pd.read_sql(api.get_ranks(isins), con=conn).set_index("ISINCode")

    weights_pd = pd.DataFrame.from_dict(weights, orient = 'index')/100

    ranks_pd = ranks.reindex(weights_pd.index)
    ranks_pd = ranks_pd.fillna(0)  # TODO: no rank assumed 0 ranking
    port_rank = np.matrix(ranks_pd.values * weights_pd.values).sum(axis=0).astype(int)  
    port_rank_pd = pd.DataFrame(port_rank, columns = ranks_pd.columns)

    return port_rank_pd

@st.cache(hash_funcs={psycopg2.extensions.connection: lambda _: None})
def calc_holding_top10(isins, weights, base_portfolio_holding_type, base_portfolio_name, benchmark_holding_type, benchmark_name , conn, is_customise = False):

    benchmark_pd = pd.DataFrame(benchmark_holding_type, columns=['Portfolio', 'HoldingType', 'HoldingName', 'port_weight'])
    benchmark_pd['Portfolio'] = 'Benchmark'
    benchmark_pd['Name'] = benchmark_name

    if is_customise == False:
        base_pd = pd.DataFrame(base_portfolio_holding_type, columns=['Portfolio', 'HoldingType', 'HoldingName', 'port_weight'])
        base_pd['Portfolio'] = 'Portfolio'
        base_pd['Name'] = base_portfolio_name
        
        holding_all = base_pd.merge(benchmark_pd, how='outer', on = ["HoldingType", "HoldingName"], suffixes = ('_x', '_y'), copy=True)

    else: 
        # calculate top 10 holdings, when portfolio top10 and benchmark top10 have more than 10 names (non-overlapping), prioritise portfolio
        holding_type = pd.read_sql(api.get_holding_type(isins), con = conn)
        holding_top10 = pd.read_sql(api.get_holding_top10(isins), con = conn)
        holding_pd = pd.concat([holding_type, holding_top10], axis=0, ignore_index=True)

        weights_pd = pd.DataFrame.from_dict(weights, orient = 'index')/100
        weights_pd_idx = weights_pd.reset_index().rename(columns ={'index': "ISINCode", 0: "PortWeight"})
        weights_pd_idx["ISINCode"] = weights_pd_idx["ISINCode"].astype(str)

        holding_pd = holding_pd.merge(weights_pd_idx, on='ISINCode', how='left')
        holding_pd['port_weight']=round(holding_pd['Weight'] * holding_pd["PortWeight"],2)
        holding_pd = holding_pd.groupby(['HoldingType','HoldingName'])['port_weight'].sum().reset_index().sort_values(["HoldingType", "port_weight"], ascending=False)
        holding_pd['Portfolio'] = 'My portfolio'
        holding_pd['Name'] = 'My portfolio'
        holding_all = holding_pd.merge(benchmark_pd, how='outer', on = ["HoldingType", "HoldingName"], suffixes = ('_x', '_y'), copy=True)
    

    holding_top10 = pd.DataFrame()

    for s in holding_all['HoldingType'].unique():
        holding_top10 = pd.concat([holding_top10, holding_all[holding_all['HoldingType'] == s][:10]], axis=0, ignore_index=True)

    holding_top10 = pd.concat([\
        holding_top10.loc[~holding_top10['Portfolio_x'].isna(), ["HoldingType", "HoldingName", 'port_weight_x', 'Portfolio_x', 'Name_x']].rename(columns = {'port_weight_x': 'Weight', 'Portfolio_x': 'Portfolio', 'Name_x': 'Name'}),\
        holding_top10.loc[~holding_top10['Portfolio_y'].isna(), ["HoldingType", "HoldingName", 'port_weight_y', 'Portfolio_y', 'Name_y']].rename(columns = {'port_weight_y': 'Weight', 'Portfolio_y': 'Portfolio', 'Name_y': 'Name'})],\
        axis=0, ignore_index=True)

    holding_top10 = holding_top10.rename(columns ={'HoldingType': 'Type', 'HoldingName': 'Holding'})
    holding_top10['Weight'] = holding_top10['Weight']/100
  
    
    return holding_top10

@st.cache(hash_funcs={psycopg2.extensions.connection: lambda _: None})
def calc_holding_all(isins, weights, conn):
    weights_pd = pd.DataFrame.from_dict(weights, orient = 'index')/100
    weights_pd_idx = weights_pd.reset_index().rename(columns ={'index': "ISINCode", 0: "PortWeight"})

    holding_all_pd = pd.read_sql(api.get_holding_all(isins), con = conn)
    holding_all_pd = holding_all_pd.merge(weights_pd_idx,  on="ISINCode", how='left')
    holding_all_pd['port_weight'] = round(holding_all_pd['Weight'] * holding_all_pd['PortWeight'], 4)
    #holding_all_pd = holding_all_pd.fillna('')

    #TODO: once find Notes or Bond name, can eliminated the != 'Note condition'
    holding_all_pd = holding_all_pd[(holding_all_pd['port_weight']>0) & (holding_all_pd['InstrumentDescription']!= '-' ) & (holding_all_pd['InstrumentDescription']!= 'Note' )]
    holding_all_pd = holding_all_pd.groupby(['InstrumentDescription', 'Country','Flag' ,'Sector'])['port_weight'].sum().reset_index().sort_values('port_weight', ascending=False)
    
    holding_all_pd = holding_all_pd[holding_all_pd['port_weight']>=0.01].rename(columns = {'InstrumentDescription': 'Name', 'port_weight': 'Weight'})
    holding_all_pd['Weight'] = holding_all_pd['Weight']/100

    return holding_all_pd



@st.cache(hash_funcs={psycopg2.extensions.connection: lambda _: None})
def get_inception_date(isins, conn):
    return run_query(api.get_inception_date(isins), conn)[0][0]


@st.cache(hash_funcs={psycopg2.extensions.connection: lambda _: None})
def get_ts(isins, start_date, conn):
    price = pd.read_sql(api.get_ts_price(isins), con = conn)
    price = price.pivot(index="TimeStamp", columns="ISINCode", values=["NAV_USD", "TR_USD", "GBP", "EUR"])

    price = price.fillna(method="ffill")

    div = pd.read_sql(api.get_ts_div(isins, start_date), con = conn)
    return price, div


# return portfolio performance (rebased at 100), 
# individual ETFs' performance (rebased at 100), and 
# shares allocated to each ETF at rebalance date(only for PR, used for dividend accumulating calculation)
@st.cache(allow_output_mutation=True)
def calc_return(price, weights_dict, rebalance_frequency, currency):
    navs_pd = price['NAV_USD']
    tr_pd = price['TR_USD']

    if currency == 'GBP':
        navs_pd = navs_pd * price['GBP']
        tr_pd = tr_pd * price['GBP']
    if currency == 'EUR':
        navs_pd = navs_pd * price['EUR']
        tr_pd = tr_pd * price['EUR']

    navs = navs_pd.values 
    tr = tr_pd.values

    
    weights = pd.DataFrame(weights_dict, index=[0])

    weights = weights[price['NAV_USD'].columns].values/100


    dates = pd.to_datetime(price.index, utc=True)

    if rebalance_frequency == 'Monthly':
        period = pd.DatetimeIndex(dates).month
    if rebalance_frequency == 'Quarterly':
        period = pd.DatetimeIndex(dates).quarter
    if rebalance_frequency == 'Annual':
        period = pd.DatetimeIndex(dates).year

    period_next = period[1:]
    rebal_idx = period[:-1] != period_next

    notional = np.zeros((len(navs),1))
    notional_tr = np.zeros((len(tr),1))
    notional[0] = 100
    notional_tr[0] = 100

    shares = np.zeros((0,len(navs[0, :])))
    shares_tr = np.zeros((0,len(tr[0, :])))

    shares_dic = {}
    shares_tr_dic = {}

    for i in range(len(navs)-1):
        if i ==0 or rebal_idx[i]==True:
            shares = np.true_divide(notional[i] * weights, navs[i,:])
            shares_tr= np.true_divide(notional_tr[i] * weights, tr[i, :])
            shares_dic[dates[i].strftime('%Y-%m-%d')] = np.squeeze(shares)
            shares_tr_dic[dates[i].strftime('%Y-%m-%d')] = np.squeeze(shares_tr)
        
        notional[i+1] = np.sum(navs[i+1,:] * shares)
        notional_tr[i+1] = np.sum(tr[i+1, :] * shares_tr)

    performance = pd.DataFrame(np.column_stack([notional, notional_tr, dates]), columns = ['PR', 'TR', 'Dates']).set_index('Dates')

    # calculate individual ETFs performance contribution
    performance_isins_navs = navs_pd * (100 / navs[0, :]) 
    performance_isins_tr = tr_pd * (100 / tr[0, :]) 
    performance_isins = pd.concat([performance_isins_navs, performance_isins_tr], axis=1, keys = ['PR', 'TR'])
    performance_isins.index = dates

    shares_pr_pd = pd.DataFrame.from_dict(shares_dic, orient='index', columns=navs_pd.columns)
    shares_tr_pd = pd.DataFrame.from_dict(shares_tr_dic, orient='index', columns=tr_pd.columns)
    shares_pd = pd.concat([shares_pr_pd, shares_tr_pd], axis=1, keys=['PR','TR'])

    return performance, performance_isins, shares_pd


@st.cache
def calc_div(dividends, shares, currency):
    
    if currency == 'USD':
        div_col = 'Div_USD'
    elif currency == 'GBP':
        div_col = 'Div_GBP'
    else:
        div_col = 'Div_EUR'

    divs_pd = dividends[['TimeStamp','ISINCode',div_col]].pivot_table(index="TimeStamp", columns = "ISINCode")[div_col]
    divs_pd.index = pd.to_datetime(divs_pd.index, utc=True)

    shares_pd = shares[divs_pd.columns]
    shares_pd.index = pd.to_datetime(shares_pd.index, utc=True)

    div_shares = pd.concat([divs_pd,shares_pd], axis=1, join='outer', keys=['Div', 'Shares'])
    total_div = div_shares['Div'] * div_shares['Shares'].shift(1)
    total_div.columns = div_shares['Div'].columns
    total_div['Total'] = total_div.sum(axis=1)

    total_div = total_div[total_div['Total'] != 0]

    return total_div

@st.cache
def convert_performance(port_returns, bench_returns, port_name, bench_name, inception_value, inception_date ):

    port_perf = copy.deepcopy(port_returns)
    bench_perf = copy.deepcopy(bench_returns)

    port_perf['Type'] = port_name
    bench_perf['Type'] = bench_name


    # st.write(port_perf)
    # st.write(bench_perf)
    performance = pd.concat([port_perf, bench_perf], axis=0)
    #st.write(performance)
    perf_pivot = performance.pivot(columns ='Type')[['PR','TR']]

    perf_select = perf_pivot[perf_pivot.index >= pd.to_datetime(inception_date, utc=True)]
    perf_select = perf_select * (inception_value/100)

    perf_select_melt = perf_select.melt(ignore_index=False).rename(columns={None: 'Dividend', 'value': 'Return'})
    perf_select_melt['Dates'] = perf_select_melt.index.strftime('%Y-%m-%d')

    return perf_select, perf_select_melt


@st.cache
def convert_div(port_return, bench_return, port_div, bench_div, port_name, bench_name, inception_value, inception_date):

    # port_income = port_div * (inception_value/100)
    # bench_income = bench_div * (inception_value/100)

    port_cum = port_div.cumsum(axis=0)['Total'].rename('Income')
    bench_cum = bench_div.cumsum(axis=0)['Total'].rename('Income')
    interval = ''

    max_len = max(len(port_return), len(bench_return))
    if  max_len > 52 * 5 * 2:
        interval = 'year'
    elif max_len > 52 * 2 * 2:
        interval = 'quarter'
    else:
        interval = 'month'
        


    def __convert_div(perf, div, interval):
        tr = pd.concat([perf[['Type','PR']], div], axis=1)
        tr[['PR', 'Income']] = tr[['PR','Income']].fillna(method='ffill')
        tr = tr[tr.index >= pd.to_datetime(inception_date, utc=True)]
        tr[['PR','Income']] = tr[['PR','Income']] * (inception_value/100)

        if interval == 'year':
            years = tr.index.year 
            idx = years[:-1] != years[1:] 
        elif interval == 'quarter':
            quarters = tr.index.quarter
            idx = quarters[:-1] != quarters[1:]
        else:
            months = tr.index.month
            idx = months[:-1] != months[1:]

        idx = np.append(idx, True)
        div = tr[idx]

        return tr, div[['Type', 'Income']]

    port_tr, port_income = __convert_div(port_return, port_cum, interval)
    bench_tr, bench_income = __convert_div(bench_return, bench_cum, interval)

    combined_tr = pd.concat([port_tr, bench_tr], axis=0).rename(columns={'PR': 'Return'})
    combined_tr = combined_tr[~combined_tr['Type'].isna()]
    combined_tr['Income'] =combined_tr['Income'].fillna(0)  
    combined_tr['TR'] = combined_tr['Return'] + combined_tr['Income']
    combined_tr['Dates'] = combined_tr.index.strftime('%Y-%m-%d')

    combined_div = pd.concat([port_income, bench_income], axis=0)
    if interval == 'year':    
        combined_div['Dates'] = combined_div.index.year
    else:
        combined_div['Dates'] = combined_div.index.strftime('%Y-%m-%d')
   

    return combined_tr, combined_div


@st.cache
def calc_relative_perf(isin_perf, port_perf, isin_names_dict):

    isin_perf_relative = (isin_perf - port_perf.values.reshape((-1,1)))/100

    isin_perf_melt = isin_perf_relative.melt(ignore_index=False).rename(columns={'value': 'Return'})
    isin_names_pd = pd.DataFrame.from_dict(isin_names_dict, orient='index').reset_index().rename(columns={'index': "ISINCode", 0: 'Name'})

    isin_perf_melt = isin_perf_melt.reset_index().merge(isin_names_pd, how='left', on="ISINCode").set_index("TimeStamp")

    isin_perf_melt['Dates'] = isin_perf_melt.index.strftime('%Y-%m-%d')

    # get top5 and bottom5 performing ETF ISINs
    latest = isin_perf_relative[-1:].melt(ignore_index=True).sort_values('value', ascending=True)["ISINCode"]
    bottom5 = latest.head(5)
    top5 = latest.tail(5)
    return isin_perf_melt, top5, bottom5

@st.cache
def calc_mom_perf(isin_perf, port_perf, isin_names_dict, port_name):
    # MoM returns
    months = isin_perf.index.month 
    idx = months[:-1] != months[1:]
    idx = np.append(idx, True)
    isin_perf_month = isin_perf[idx]
    isin_perf_month = isin_perf_month/isin_perf_month.shift(1) -1
    isin_perf_month = isin_perf_month[1:]

    isin_perf_month_melt = isin_perf_month.melt(ignore_index=False).rename(columns={'value': 'Return'})
    isin_names_pd = pd.DataFrame.from_dict(isin_names_dict, orient='index').reset_index().rename(columns={'index': "ISINCode", 0: 'Name'})

    isin_perf_month_melt = isin_perf_month_melt.reset_index().merge(isin_names_pd, how='left', on="ISINCode").set_index("TimeStamp")

    isin_perf_month_melt['Dates'] = isin_perf_month_melt.index.strftime('%Y-%m-%d')


    port_perf_month = port_perf[idx]
    port_perf_month = port_perf_month / port_perf_month[0] - 1
    port_perf_month = port_perf_month[1:].reset_index().rename(columns={'PR': 'Return'})
    port_perf_month['Dates'] = port_perf_month['Dates'].dt.strftime('%Y-%m-%d')
    port_perf_month['Name'] = port_name

    return isin_perf_month_melt, port_perf_month

#@st.cache
def calc_calender_return(perf_pivot):
    years = perf_pivot.index.year
    calendar_years = pd.unique(years)
    idx = years[:-1] != years[1:]
    idx = np.append(idx, True)
    perf_calendar = perf_pivot[idx]
    perf_calendar = perf_calendar / perf_calendar.shift(1) -1 
    perf_calendar = perf_calendar[1:]
    perf_calendar = perf_calendar.melt(ignore_index=False).rename(columns={None: 'Dividend', 'value': 'Return'})

    perf_calendar['Year'] = calendar_label = [str(i) + ' YTD' if i==max(calendar_years) else str(i) for i in perf_calendar.index.year]
    return perf_calendar

#@st.cache
def calc_cumulative_return(perf_pivot):
    period_idx = np.array([5, 14, 28, 54, 53*3+1, 53*5+2, 53*10+3])
    period = np.array(['1 month', '3 month', '6 month', '1 year', '3 year', '5 year', '10 year'])
    period_num = np.array([0, 0, 0, 1, 3, 5, 10])
    valid_idx = period_idx[period_idx<len(perf_pivot)]
    cum_idx = -1*valid_idx
    cum_label = period[period_idx<len(perf_pivot)]
    cum_num = period_num[period_idx<len(perf_pivot)]

    perf_cum = np.true_divide(perf_pivot[-1:].values, perf_pivot.iloc[cum_idx,:].values) -1
    perf_cum = pd.DataFrame(perf_cum, columns = perf_pivot.columns, index = cum_label)


    # only select period > 1Y for annualised return
    perf_annual = perf_cum[cum_num>=1]
    cum_num_valid = cum_num[cum_num>=1]
    
    if len(cum_num_valid) > 0:
        perf_annual_np = np.power(perf_annual.values + 1, 1/np.expand_dims(cum_num_valid, axis=1)) -1
        perf_annual = pd.DataFrame(perf_annual_np, columns = perf_annual.columns, index=cum_label[cum_num>=1])
        perf_annual = perf_annual.melt(ignore_index=False).reset_index().rename(columns={'index': 'Period', 'value': 'Return'})

    else:
        perf_annual = pd.DataFrame()
    perf_cum = perf_cum.melt(ignore_index=False).reset_index().rename(columns={'index': 'Period', 'value': 'Return'})
    
    return perf_cum, perf_annual, cum_label, cum_label[cum_num>=1]

#@st.cache
def calc_drawdown(data_pivot):
    max_pd = data_pivot.rolling(len(data_pivot), min_periods=1).max()
    drawdown = data_pivot / max_pd - 1

    drawdown = drawdown.melt(ignore_index=False).rename(columns={None: 'Dividend', 'value': 'Drawdown'})
    drawdown['Dates'] = drawdown.index.strftime('%Y-%m-%d')

    return drawdown


@st.cache
def get_crisis(file_path):
    crisis = pd.read_csv(file_path)
    crisis['Start'] = pd.to_datetime(crisis['Start'], utc=True)
    crisis['End'] = pd.to_datetime(crisis['End'], utc=True)

    return crisis


#@st.cache
def calc_vol(data, frequency):
    return_wk = data / data.shift(1) -1
    vol = return_wk.rolling(frequency).std(ddof=0)*np.sqrt(252)
    vol = vol.dropna(how='all')
    vol = vol.melt(ignore_index=False).rename(columns={None: 'Dividend', 'value': 'Volatility'})
    vol['Dates'] = vol.index.strftime('%Y-%m-%d')

    return vol

def calc_updown(data):
    ret = data/data.shift(1) - 1
    ret = ret.melt(ignore_index=False).rename(columns={'value': 'Return'})
    ret['Dates'] = ret.index.strftime('%Y-%m-%d')
    ret['Sign'] = ret['Return'].apply(lambda x: 'Positive' if x>0 else 'Negative')

    return ret

        
def submit_user_portfolio(records, conn):
    cur = conn.cursor()
    update = """ INSERT INTO users("UserName", "Email", "Portfolio","Frequency","Holdings")
                VALUES(%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING 
            """
    cur.execute(update, records)

    conn.commit()
    cur.close()

    return 


