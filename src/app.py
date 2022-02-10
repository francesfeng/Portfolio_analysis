import streamlit as st
import pandas as pd
import numpy as np 

import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import os

import psycopg2 

import datetime
import copy
import json

from conn import api
import style
from conn.data import init_connection
from conn.data import run_query
from conn.data import init_data
from conn.data import get_portfolio_holding
from conn.data import calc_rank
from conn.data import calc_holding_top10
from conn.data import calc_holding_all
from conn.data import get_inception_date
from conn.data import get_ts
from conn.data import calc_return
from conn.data import calc_div
from conn.data import convert_performance
from conn.data import convert_div
from conn.data import calc_relative_perf
from conn.data import calc_mom_perf
from conn.data import calc_calender_return
from conn.data import calc_cumulative_return
from conn.data import calc_drawdown
from conn.data import get_crisis
from conn.data import calc_vol
from conn.data import calc_updown
from conn.data import submit_user_portfolio

from components import create_benchmark_table
from components import create_frontier_graph
from components import load_default_portfolio
from components import customise_portfolio
from components import draw_holding_graph
from components import draw_full_holding_graph
from components import draw_radar_graph
from components import draw_pie_graph
from components import performance_graph
from components import performance_bar_graph
from components import performance_overlay_graph
from components import performance_grouped_bar_graph
from components import drawdown_graph
from components import distribution_graph


st.set_page_config(layout="wide")
#https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config


alt.themes.register("lab_theme", style.lab_theme)
alt.themes.enable("lab_theme")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("src/style.css")


#------------------------------------------------------------------------------------Establish db connection

conn = init_connection()


benchmarks, benchmark_perf, benchmark_names, etfs_lst = init_data(conn)

if 'benchmark' not in st.session_state:
	st.session_state.benchmark = benchmark_names[1]

if 'base_name' not in st.session_state:
	st.session_state.base_name = benchmark_names[0]


benchmark_etfs, benchmark_ranks, benchmark_holdings = get_portfolio_holding(st.session_state.benchmark, conn)
base_etfs, base_ranks, base_holdings = get_portfolio_holding(st.session_state.base_name, conn)


if 'base_port' not in st.session_state:
	st.session_state.base_port = {k: v for k, v in base_etfs.items()}
if 'base_rank' not in st.session_state:
	st.session_state.base_rank = base_ranks
if 'base_holding_type' not in st.session_state:
	st.session_state.base_holding_type = base_holdings
if 'base_weight' not in st.session_state:
	st.session_state.base_weight = {k:v['Weight'] for k, v in base_etfs.items()}


if 'bench_port' not in st.session_state:
	st.session_state.bench_port = {k: v for k, v in benchmark_etfs.items()}
if 'bench_rank' not in st.session_state:
	st.session_state.bench_rank = benchmark_ranks
if 'bench_holding_type' not in st.session_state:
	st.session_state.bench_holding_type = benchmark_holdings
if 'bench_weight' not in st.session_state:
	st.session_state.bench_weight = {k:v['Weight'] for k, v in benchmark_etfs.items()}
	

if 'portfolio' not in st.session_state:
	st.session_state.portfolio = copy.deepcopy(st.session_state.base_port)

if 'weights' not in st.session_state:
	st.session_state.weights = copy.deepcopy(st.session_state.base_weight)

if 'port_weight' not in st.session_state:
	st.session_state.port_weight = sum([v*100 for k, v in st.session_state.base_weight.items()])/100




if 'port_run' not in st.session_state:
	st.session_state.port_run = True

if 'customise' not in st.session_state:
	st.session_state.customise = False

if 'msg' not in st.session_state:
	st.session_state.msg = ['','']

if 'analyse_msg' not in st.session_state:
	st.session_state.analyse_msg = ['', '']

if 'search_etf' not in st.session_state:
	st.session_state.search_etf = []

if 'previous_added_etfs' not in st.session_state:
	st.session_state.previous_added_etfs = []

#------------------------------------------------------------------------------------Benchmark portfolios

with st.expander("Portfolio Overview", expanded = not(st.session_state.customise)):
	col_bench1, col_bench2 = st.columns(2)
	with col_bench1:
		st.write('##### Portfolios')
		st.write("<div style='height: 500px; overflow: auto; width: fit-content'>" +
	             create_benchmark_table(benchmarks).to_html() +
	             "</div>", unsafe_allow_html=True)


	with col_bench2:
		st.write('##### Risk return frontier')
		#st.write(benchmark_rr)
		alt_frontier = create_frontier_graph(benchmark_perf)

		st.altair_chart(alt_frontier, use_container_width=True)


#------------------------------------------------------------------------------------Portfolios callback function
def change_base():
	name = st.session_state.base_name
	etfs, ranks, holdings = get_portfolio_holding(name, conn)
	st.session_state.base_port = {k: v for k, v in etfs.items()}
	st.session_state.base_weight = {k: v['Weight'] for k, v in etfs.items()}
	st.session_state.base_rank = ranks
	st.session_state.base_holding_type = holdings

	reset_portfolio()
	return

def change_benchmark():
	name = st.session_state.benchmark
	etfs, ranks, holdings = get_portfolio_holding(name, conn)
	st.session_state.bench_port = {k: v for k, v in etfs.items()}
	st.session_state.bench_weight = {k: v['Weight'] for k, v in etfs.items()}
	st.session_state.bench_rank = ranks
	st.session_state.bench_holding_type = holdings

	st.session_state.msg = ['', '']
	st.session_state.analyse_msg = ['','']

	return

def add_etf():
	st.session_state.port_run = False
	st.session_state.msg = ['','']

	if len(st.session_state.search_etf) == 0:
		return

	etf_exist = []
	etf_add = []
	etf_isins = []
	etf_remove = []

	for i in st.session_state.search_etf:
		if i[0] in st.session_state.previous_added_etfs:
				etf_remove += [i[0]]

		elif i[0] in st.session_state.portfolio:
				etf_exist += [i[1]]
		else:
			etf_isins += [i[0]]
			etf_add += [i[1]]
			if i[0] not in st.session_state:
				st.session_state[i[0]] = 0.00


	if len(etf_isins) > 0:
		etfs_info = pd.read_sql(api.get_etf_details_for_portfolio(etf_isins), con = conn).set_index("ISINCode")
		etfs_dict = etfs_info.to_dict(orient='index')

		for k, v in etfs_dict.items():
			st.session_state.portfolio[k] = v
			st.session_state.portfolio[k]['Weight'] = 0.00


		st.session_state.msg = ['Success', ' '.join(etf_add) + ' is added to the portfolio']

	if len(etf_exist) > 0:
		st.session_state.msg = ['Fail', ' '.join(etf_exist) + ' exists in the portfolio']
		
	st.session_state.previous_added_etfs = [i[0] for i in st.session_state.search_etf]

	return


@st.cache
def remove_etf(etf_isin, etf_name):
	old_weight = st.session_state[etf_isin]
	del st.session_state.portfolio[etf_isin]
	
	del st.session_state[etf_isin]
	st.session_state.port_weight -= old_weight

	st.session_state.msg = ['Info', etf_name + ' has been removed from the portfolio']
	st.session_state.analyse_msg = ['','']

	st.session_state.port_run = False
	return


def empty_portfolio():
	for k in st.session_state.portfolio:
		del st.session_state[k]
	st.session_state.portfolio = {}
	st.session_state.port_weight = 0
	st.session_state.msg = ['', '']
	st.session_state.port_run = False

	st.session_state.msg = ['', '']
	st.session_state.analyse_msg = ['','']
	st.session_state.previous_added_etfs = [i[0] for i in st.session_state.search_etf]

	return


def get_total_weight(etf_isin):
	st.session_state.port_run = False
	old_weight = st.session_state.portfolio[etf_isin]["Weight"]
	new_weight = round(st.session_state[etf_isin],2)

	if st.session_state.port_weight - old_weight + new_weight > 100:
		st.session_state.msg = ['Fail', "Total portfolio weight cannot exceed 100%"]
		revised_weight = 100 - st.session_state.port_weight + old_weight
		st.session_state.portfolio[etf_isin]["Weight"] = revised_weight
		st.session_state[etf_isin] = revised_weight
		st.session_state.port_weight = 100

	else:
		st.session_state.portfolio[etf_isin]["Weight"] = new_weight
		st.session_state[etf_isin] = new_weight
		st.session_state.port_weight = st.session_state.port_weight - old_weight + new_weight
		st.session_state.msg = ['', '']

	st.session_state.msg = ['', '']
	st.session_state.analyse_msg = ['','']

def reset_portfolio():
	st.session_state.port_run = False
	st.session_state.portfolio = copy.deepcopy(st.session_state.base_port)
	st.session_state.port_weight = sum([v*100 for k, v in st.session_state.base_weight.items()])/100
	st.session_state.weights = copy.deepcopy(st.session_state.base_weight)

	st.session_state.previous_added_etfs = [i[0] for i in st.session_state.search_etf]

	st.session_state.msg = ['', '']
	st.session_state.analyse_msg = ['','']

	return


def run_portfolio():

	if st.session_state.port_weight == 0:
		st.session_state.analyse_msg = ['Fail', "Please assign weight for each ETF allocation before analyse the portfolio"]
		st.session_state.port_run = False
		return	

	if st.session_state.port_weight < 100:
		weights = [v['Weight'] for k, v in st.session_state.portfolio.items()]
		if 0 in weights:
			st.session_state.analyse_msg = ['Fail', "There are ETFs with 0 allocation, please assign a weight before running the portfolio"]
			st.session_state.port_run = False
			return
		else:
			new_weights = [round(i/st.session_state.port_weight*100,2) for i in weights]
			new_weights[-1] = 100.00- sum(new_weights[:-1])

			for i, p in enumerate(st.session_state.portfolio):
				st.session_state.portfolio[p]['Weight'] = new_weights[i]
				st.session_state[p] = new_weights[i]
			st.session_state.port_weight = 100
			st.session_state.analyse_msg = ['Info',"Each ETF allocation has been re-weighted to meet total weight of 100%"]
	
	st.session_state.port_run = True
	st.session_state.weights = {k:v["Weight"] for k, v in st.session_state.portfolio.items()}
	st.session_state.msg = ['', '']

	return


#-----------------------------------------------------------------------------------------------------------------Portfolio builder


with st.expander("Portfolio Holdings", expanded=True):

	col_bench_select, col_bench_select2, col_bench_select3 = st.columns([7, 7, 2])

	col_bench_select.selectbox("Select a portfolio", benchmarks['Portfolio'], \
										index=0, key='base_name', on_change=change_base)

	with col_bench_select2:
		st.write("")
		st.write("")
		st.checkbox("Customise the portfolio", key = 'customise')

	with col_bench_select3:
		st.write("")
		st.write("")
		st.empty()
		
	if st.session_state.customise:

		# search ETFs
		with col_bench_select3:
			st.button("↩️ Reset", help="Reset to portfolio's original constituents", on_click = reset_portfolio, key='reset')

		col_search1, col_search2 = st.columns([3,1])

		with col_search1:
			etf_add = st.multiselect("Search ETFs to add to portfolio",etfs_lst,format_func=lambda x: """{} | {} | {}""".format(x[1], x[2],x[3]) if len(x) > 1 else '', key='search_etf', on_change = add_etf)
			
		# portfolio builder
		col_port1, warning_msg, _, col_port2 = st.columns([2, 10, 2, 2])

		if len(st.session_state.portfolio) > 0:
			with col_port1:
				st.write('')
				st.write('**Holdings**', "(", len(st.session_state.portfolio), ")" , key="holding_count")

			with warning_msg:
				if st.session_state.msg[0] == 'Success':
					st.success(st.session_state.msg[1])
				elif st.session_state.msg[0] == 'Fail':
					st.warning(st.session_state.msg[1])
				elif st.session_state.msg[0] == 'Info':
					st.info(st.session_state.msg[1])
				else:
					st.empty()

			with col_port2:
				st.write('')
				st.button("🗑️ Delete all", help="Remove all EFFs in this portfolio", on_click = empty_portfolio, key='empty')
			

			customise_portfolio(st.session_state.portfolio, get_total_weight, remove_etf)
			
			st.write('')
			
			col_analyse1, col_analyse2 = st.columns([2, 10])
			with col_analyse1:
				st.write('')
				st.button("💹 Analyse my portfolio", on_click = run_portfolio)
				st.write('')
				st.write('')
			with col_analyse2:
				if st.session_state.analyse_msg[0] == 'Success':
					st.success(st.session_state.analyse_msg[1])
				elif st.session_state.analyse_msg[0] == 'Fail':
					st.warning(st.session_state.analyse_msg[1])
				elif st.session_state.analyse_msg[0] == 'Info':
					st.info(st.session_state.analyse_msg[1])
				else:
					st.empty()
		else:
			st.warning('The portfolio is empty')

	else:
		st.session_state.port_run = True
		load_default_portfolio(st.session_state.base_port)


if st.session_state.port_run == True:
	#---------------------------------------------------------------------------------------------------------------------------- Rank
	port_name = 'Portfolio' if st.session_state.customise==False else 'My portfolio'
	with st.expander("Ranks", expanded=True):
		col_rankt1, col_rankt2, col_rankt3, col_rankt4, _ ,col_rankt5 = st.columns([2,1,1,1,1,6])


		with col_rankt1:
			st.write(" ")
			st.write(" ")
			st.write("**Average Ranking**")
			st.write(" ")
			st.write(" ")
			st.write(" ")

		with col_rankt2:
			if st.session_state.customise:
				port_rank_pd = calc_rank([k for k in st.session_state.portfolio], st.session_state.weights ,conn)
				st.metric(label="My portfolio", value=port_rank_pd.values[0][-1], delta="") #.css-nebmhc
			else:
				st.metric(label="Portfolio", value=st.session_state.base_rank['Rank'], delta="") 
		with col_rankt3:
			st.write("**vs.**")
		with col_rankt4:
			st.metric(label="Benchmark", value=st.session_state.bench_rank['Rank'], delta="")
		

		with col_rankt5:
			st.selectbox("Benchmark portfolio to compare", benchmark_names, key="benchmark", on_change=change_benchmark)

		col_rank1, col_rank2= st.columns(2)

		rank_categories = [k for k in st.session_state.base_rank][:-1]
		base_rank = port_rank_pd.values[0][:-1] if st.session_state.customise else [v for k, v in st.session_state.base_rank.items()]
		with col_rank1:	
			fig_radar = draw_radar_graph(
						base_rank, 
						[v for k, v in st.session_state.bench_rank.items() ], 
						rank_categories, 
						port_name,
						'Benchmark')	
			st.plotly_chart(fig_radar, use_container_width=True)


		with col_rank2:
			bench_names = pd.DataFrame.from_dict(st.session_state.bench_port, orient='index')[['FundName', 'Weight', 'Sector']]
			
			fig_pie = draw_pie_graph(bench_names)
			st.plotly_chart(fig_pie, use_container_width=True)

		

	#---------------------------------------------------------------------------------------------------------------------- Holding
	with st.expander('Holdings', expanded=True):

		holding_type = calc_holding_top10([k for k in st.session_state.portfolio], st.session_state.weights, \
											st.session_state.base_holding_type, st.session_state.base_name,\
											st.session_state.bench_holding_type, st.session_state.benchmark,\
											conn, st.session_state.customise)

		holding_select = st.radio(label = '', options = ['Main','Other Characters','Full Holdings'])
		st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


		if holding_select == 'Main':
			type1 = 'Top10'
			type2 = 'Sector'
			type3 = 'Geography'

			title1 = 'Top 10 Holdings'
			title2 = 'Sector'
			title3 = 'Country'
		elif holding_select == 'Other Characters':
			type1 = 'AssetClass'
			type2 = 'Currency'
			type3 = 'Exchange'
			type4 = 'Credit Rating'

			title1 = 'Asset Class'
			title2 = 'Currency'
			title3 = 'Exchange'
			title4 = 'Credit Rating (Bond holdings only)'
		else:
			None

		_, legend1 = draw_holding_graph(holding_type[holding_type['Type'] == 'Sector'], 'Weight', 'Portfolio', 'Holding', [port_name, 'Benchmark'] ,'', st.session_state.customise)
		if holding_select != 'Full Holdings':
			alt1, _ = draw_holding_graph(holding_type[holding_type['Type'] == type1], 'Weight', 'Portfolio', 'Holding', [port_name, 'Benchmark'] ,title1, st.session_state.customise)
			alt2, _ = draw_holding_graph(holding_type[holding_type['Type'] == type2], 'Weight', 'Portfolio', 'Holding', [port_name, 'Benchmark'], title2, st.session_state.customise)
			alt3,_ = draw_holding_graph(holding_type[holding_type['Type'] == type3], 'Weight', 'Portfolio', 'Holding', [port_name, 'Benchmark'], title3, st.session_state.customise)

			st.altair_chart(legend1)
			if holding_select == 'Main':
				col_h1, _, col_h2, _, col_h3, _ = st.columns([3, 1, 3, 1, 3, 1])
			else:
				col_h1, _, col_h2, _, col_h3, _, col_h4, _ = st.columns([3, 1, 3, 1, 3, 1, 3, 1])
				alt4,_ = draw_holding_graph(holding_type[holding_type['Type'] == type4], 'Weight', 'Portfolio', 'Holding', [port_name, 'Benchmark'], title4, st.session_state.customise)

			col_h1.altair_chart(alt1.configure_header(labelOrient = "right").configure_axisY(disable=True), use_container_width=True)
			col_h2.altair_chart(alt2.configure_header(labelOrient = "right").configure_axisY(disable=True), use_container_width=True)
			col_h3.altair_chart(alt3.configure_header(labelOrient = "right").configure_axisY(disable=True), use_container_width=True)

			if holding_select == 'Other Characters':
				col_h4.altair_chart(alt4.configure_header(labelOrient = "right").configure_axisY(disable=True), use_container_width=True)

		else:
			if st.session_state.customise:
				holding_all = calc_holding_all([k for k in st.session_state.weights], st.session_state.weights, conn)
			else:
				holding_all = calc_holding_all([k for k in st.session_state.base_port], st.session_state.base_weight, conn)

			st.altair_chart(draw_full_holding_graph(holding_all), use_container_width=True)


	#------------------------------------------------------------------------------------------------------------------ Performance
	if st.session_state.customise:
		port_isins = [k for k in st.session_state.weights]
		port_weights = st.session_state.weights
	else:
		port_isins = [k for k in st.session_state.base_port]
		port_weights = st.session_state.base_weight
	
	inception_date = get_inception_date(port_isins, conn)


	with st.expander("Historical Performance", expanded=True):
		col_perf1, col_perf2, col_perf3, col_perf4,col_perf5, col_perf6 = st.columns([3,2,2,1,2,3])
		rebal = col_perf1.selectbox("Portfolio rebalance frequency", ["Monthly", "Quarterly", "Annual"], key='rebalance')
		div = col_perf2.selectbox("Dividend treatment", ["Reinvest", "Distribute"], key='dividend')
		start_value = col_perf3.number_input("Start to invest", min_value=0, value = 10000,step=100, key='start_value')
		currency = col_perf4.selectbox(".", ["GBP", "EUR", "USD"], key='currency')
		start_date = col_perf5.date_input("Inception date", value=inception_date, min_value=inception_date, key='start_date')
		compare = col_perf6.multiselect("Compare to", ["Benchmark", "S&P"], default="Benchmark" )

		
		prices, divs = get_ts(port_isins, inception_date.strftime('%Y-%m-%d'), conn)
		port_returns, isin_perf, shares = calc_return(prices, port_weights, st.session_state.rebalance, st.session_state.currency)

		bench_prices, bench_divs = get_ts([k for k in st.session_state.bench_weight], inception_date.strftime('%Y-%m-%d'), conn)
		bench_returns, _ , bench_shares = calc_return(bench_prices, st.session_state.bench_weight,st.session_state.rebalance, st.session_state.currency)
		return_type = 'TR' if st.session_state.dividend == 'Reinvest' else 'PR'	
		

		performance_pivot, performance = convert_performance(port_returns, bench_returns, port_name, 'Benchmark', \
												st.session_state.start_value,\
												st.session_state.start_date) 
		
		if st.session_state.dividend == 'Reinvest':
			
			alt_perf = performance_graph(performance[performance['Dividend'] == return_type], 'Return', [port_name, 'Benchmark'], ',.4r')
			st.altair_chart(alt_perf.properties(title="Investment Performance"), use_container_width=True)

		else:
			port_income = calc_div(divs, shares['PR'], st.session_state.currency)
			bench_income = calc_div(bench_divs, bench_shares['PR'], st.session_state.currency)

			tr_combined, div_combined = convert_div(port_returns, bench_returns, \
										port_income, bench_income, \
										port_name, 'Benchmark', \
										st.session_state.start_value,\
										st.session_state.start_date
										)
			alt_perf = performance_graph(tr_combined, 'Return', [port_name, 'Benchmark'], ',.2r')
			alt_tr = performance_bar_graph(tr_combined, 'Income', [port_name, 'Benchmark'],',.2r')
			st.altair_chart(alt_perf.properties(title='Investment Performance',height=400), use_container_width=True)
			st.altair_chart(alt_tr.properties(height=300), use_container_width=True)
			
	#---------------------------------------------------------------------------------------------------------------- Calendar Performance
	calendar_year = pd.unique(performance.index.year)
	perf_calendar = calc_calender_return(performance_pivot[return_type])
	perf_cum, perf_annual, cum_label, ann_label = calc_cumulative_return(performance_pivot[return_type])
	return_lst = ['Cumulative Return']

	if len(perf_annual) > 0:
		return_lst += ['Annualised Return']
	if len(perf_calendar) > 0:
		return_lst += ['Calender Return']

	with st.expander('Performance Overview', expanded=True):
		col_cal1, col_cal2 = st.columns(2)
		return_select = col_cal1.radio("", return_lst, 0)
		st.altair_chart(legend1, use_container_width=True)

		if return_select == 'Cumulative Return':
			# perf_cum = perf_cum[perf_cum['Dividend'] == return_type]
			min_return = perf_cum['Return'].min()
			max_return = perf_cum['Return'].max()
			
			for i, col in enumerate(st.columns(len(cum_label))):
				with col:
					alt_cum_return = performance_grouped_bar_graph(perf_cum[perf_cum['Period'] == cum_label[i]], 'Return', \
												cum_label[i], [port_name, 'Benchmark'] ,cum_label, [min_return, max_return], True if i == 0 else False)
					st.altair_chart(alt_cum_return, use_container_width=True)
		elif return_select == 'Annualised Return':
			# perf_annual = perf_annual[perf_annual['Dividend'] == return_type]
			min_return = perf_annual['Return'].min()
			max_return = perf_annual['Return'].max()
			for i, col in enumerate(st.columns(len(ann_label))):
				with col:
					alt_annual_return = performance_grouped_bar_graph(perf_annual[perf_annual['Period'] == ann_label[i]], 'Return', \
												ann_label[i], [port_name, 'Benchmark'], ann_label, [min_return, max_return], True if i == 0 else False)
					st.altair_chart(alt_annual_return, use_container_width=True)
		else:

			cal_label = perf_calendar['Year'].unique()
			min_return = perf_calendar['Return'].min()
			max_return = perf_calendar['Return'].max()

			for i, col in enumerate(st.columns(len(cal_label))):
				with col:
					alt_calendar_return = performance_grouped_bar_graph(perf_calendar[perf_calendar['Year'] == cal_label[i]], 'Return', \
													cal_label[i], [port_name, 'Benchmark'], cal_label, [min_return, max_return], \
													True if i == 0 else False, False)
					st.altair_chart(alt_calendar_return, use_container_width=True)
		

	#-------------------------------------------------------------------------------------------------------- Performance contribution

	if st.session_state.customise == True:
		isin_names = {k: v['FundName'] for k, v in st.session_state.portfolio.items()}
	else:
		isin_names = {k: v['FundName'] for k, v in st.session_state.base_port.items()}
	isin_perf_relative, top5, bottom5 = calc_relative_perf(isin_perf['PR'], port_returns['PR'], isin_names)
	
	
 	
	with st.expander("ETFs Performance contributions", expanded=True):
		col_rel1, col_rel2, col_rel3 = st.columns([2, 2, 3])
		with col_rel1:
			st.write('')
			relative_type = st.radio('', ['Relative performance to portfolio', 'Month-on-Month Returns'])
		with col_rel2:
			st.write('')
			perf_type = st.radio('',['Top 5 ETFs', 'Bottom 5 ETFs', 'Customise'])
		with col_rel3:
			if perf_type == 'Top 5 ETFs':
				etfs_select = st.multiselect('',isin_names, default=top5, format_func=lambda x: isin_names[x], disabled=True)
			elif perf_type == 'Bottom 5 ETFs':
				etfs_select = st.multiselect('',isin_names, default=bottom5, format_func=lambda x: isin_names[x], disabled=True)
			else:
				etfs_select = st.multiselect('',isin_names, default=[k for k in isin_names][:3], format_func=lambda x: isin_names[x])


		if relative_type == 'Relative performance to portfolio':
			alt_perf_isins = performance_graph(isin_perf_relative[isin_perf_relative['ISINCode'].isin(etfs_select)], 'Return', [isin_names[k] for k in etfs_select], '.1%', 'Name')
			st.altair_chart(alt_perf_isins.properties(height=500), use_container_width=True)

		else: 
			isin_perf_mom, port_perf_mom = calc_mom_perf(isin_perf['PR'], port_returns['PR'], isin_names, port_name)
			alt_perf_isins_mon = performance_overlay_graph(isin_perf_mom[isin_perf_mom['ISINCode'].isin(etfs_select)], \
										port_perf_mom,'Return',[isin_names[k] for k in etfs_select], [port_name], '.1%', 'Name')

	
			st.altair_chart(alt_perf_isins_mon.properties(height=500) , use_container_width=True)


	
	#------------------------------------------------------------Mar drawdown
	

	crisis = get_crisis('./data/events.csv')
	
	with st.expander("Drawdown", expanded=True):
		col_dd1, col_dd2, _ = st.columns([1,1,2])
		with col_dd1:
			start_date = st.date_input("From:", value=inception_date, min_value=inception_date)
		with col_dd2:
			end_date = st.date_input("To", min_value=start_date)

		performance_dd = performance_pivot[(performance_pivot.index >= pd.to_datetime(start_date, utc=True))&(performance_pivot.index <= pd.to_datetime(end_date, utc=True))]

		performance_dd = performance_dd[return_type]
		drawdown = calc_drawdown(performance_dd)

		crisis = crisis[crisis['Start'] >= pd.to_datetime(start_date, utc=True)]

		alt_dd = drawdown_graph(drawdown, crisis, port_name)
		st.altair_chart(alt_dd, use_container_width=True)

	#------------------------------------------------------------ Volatility

	with st.expander("Volatility", expanded=True):
		st.radio("", ['Monthly', 'Quarterly', 'Annual'], index=0, key='vol_frequency')
		

		if st.session_state.vol_frequency == 'Monthly':
			vol_freq = 4
		elif st.session_state.vol_frequency == 'Quarterly':
			vol_freq = 12
		else:
			vol_freq = 52
		
		vol = calc_vol(performance_pivot[return_type], vol_freq)
		alt_vol = performance_graph(vol, 'Volatility',[port_name, 'Benchmark'], '.0%')
		st.altair_chart(alt_vol, use_container_width=True)


	#------------------------------------------------------------ Positive/Negative Return Distribution

	def submit_portfolio():
		records = (st.session_state.username, st.session_state.user_email,st.session_state.user_portfolio,\
											st.session_state.user_frequency, json.dumps(st.session_state.weights))
		submit_user_portfolio(records, conn)

		return


	with st.expander("Up/Down Weeks Count", expanded=True):
		ret = calc_updown(performance_pivot[return_type])
		alt_dis = distribution_graph(ret)
		st.altair_chart(alt_dis, use_container_width=True)

	if st.session_state.customise == True:
		with st.form("salve_portfolio"):
			st.write("Save my portfolio to receive regular performance update by email")
			col_save1, col_save2, col_save3, col_save4 = st.columns(4)
			user_name = col_save1.text_input("Username", key='username')
			user_email = col_save2.text_input("Email", key='user_email')
			user_port = col_save3.text_input("Portfolio Name", key='user_portfolio')
			user_frequency = col_save4.selectbox("How often I want to receive my portfolio update", ['Weekly', 'Bi-weekly', 'Monthly', 'Quarterly'], key='user_frequency')

			submitted = st.form_submit_button("Submit", on_click=submit_portfolio)

			if submitted:
				st.write("Congratulations! You've successfully saved the portfolin. Next update wtill arrive by email on xxx")

