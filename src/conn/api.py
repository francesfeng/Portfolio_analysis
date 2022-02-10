def get_benchmark_frontier():
	return """SELECT * FROM gcp_benchmark_vol 
				LEFT JOIN (SELECT "Portfolio", "Period", "AnnualisedReturn" FROM gcp_benchmark_return) AS price
				USING("Portfolio", "Period") """


def get_benchmarks():
	return """ SELECT * FROM (
				(SELECT "Portfolio", COUNT("Weight") FROM portfolios GROUP BY 1) AS weight
				LEFT JOIN (SELECT "Portfolio", 
						   CASE 
						     WHEN "Rank" >= 80 THEN '⭐⭐⭐⭐⭐'
						   	 WHEN "Rank" >= 60 THEN '⭐⭐⭐⭐'
						   	 WHEN "Rank" >= 40 THEN '⭐⭐⭐'
						   	 WHEN "Rank" >= 20 THEN '⭐⭐'
						     ELSE '⭐'
						   END "Rank5"	   
						   FROM  gcp_benchmark_rank) AS ranks
				USING("Portfolio")
				LEFT JOIN (SELECT "Portfolio", Round("CumulativeReturn"::NUMERIC,3) AS "Return" FROM gcp_benchmark_return WHERE "Period" = '1M') AS return
				USING("Portfolio")
			) AS port_map 
	"""

def get_benchmark_risk_return():
	return """SELECT * FROM (
			(SELECT "Portfolio", "Period", "Volatility" FROM gcp_benchmark_vol) AS vol
			LEFT JOIN (SELECT "Portfolio", "Period", "AnnualisedReturn" FROM gcp_benchmark_return) AS return
			USING("Portfolio", "Period") 
		) AS map"""



def get_unique_fund_list():
	return """SELECT "ISINCode", "FundName", "Sector", "Tickers" FROM gcp_etf_lookup_view"""


def get_latest_date():
	return """ SELECT "DateLatest" FROM gcp_etfs WHERE "DateLatest" IS NOT NULL """


def get_by_isin(isins, table_name, col_names = None):
	isin_lst = '\', \''.join(isins)
	if col_names is not None:
		cols = '\", \"'.join(col_names)
		return """ SELECT \"""" + cols + """\" FROM """ +table_name+ """ WHERE "ISINCode" IN ('""" + isin_lst + """')"""
	else:
		return """ SELECT * FROM """ +table_name+ """ WHERE "ISINCode" IN ('""" + isin_lst + """')"""


def get_etf_details_for_portfolio(isins):
	return get_by_isin(isins, 'gcp_etfs', ["ISINCode", "FundName", "Sector" ,"NAV", "AUM", "Cost", "Rank5"])

def get_ranks(isins):
	return get_by_isin(isins, "calc_rank", ["ISINCode", "CostRank", "ReturnRank", "AUMRank", "TrackingErrorRank", "VolumeRank", "Rank"])


# def get_portfolio_list():
# 	return """SELECT DISTINCT("Portfolio") FROM portfolios"""


# def get_benchmark_weight(name):
# 	return """ SELECT "FundName", "Weight" FROM gcp_benchmarks WHERE "Portfolio" = '""" + eval('name') + """' """

def get_benchmark_rank(name):
	return """ SELECT * FROM gcp_benchmark_rank WHERE "Portfolio" = '""" + eval('name') + """' """


def get_holding_type(isins):
	query = get_by_isin(isins, "latest_holding_type", ["ISINCode", "HoldingType", "HoldingName", "Weight", "Rank"])
	query += """ AND "Rank" <=10 """
	return query

def get_holding_top10(isins):
	query = get_by_isin(isins, "latest_holding_all", ["ISINCode", "InstrumentDescription", "Weight", "Rank"])
	return """SELECT "ISINCode", 'Top10' AS "HoldingType", "InstrumentDescription" AS "HoldingName", "Weight", "Rank" FROM (  """ + query + """ ) AS a WHERE "Rank" <= 10 """

def get_holding_all(isins):
	query = get_by_isin(isins, "latest_holding_all", ["ISINCode", "InstrumentDescription", "Country", "Sector", "Weight", "Rank"])

	return """ SELECT * FROM ( (""" + query  + """ ) AS a LEFT JOIN (SELECT "Country", "Flag" FROM country_list) AS b USING("Country") ) AS map """


def get_benchmark_weight(name):
	return """ SELECT "ISINCode", "Weight" FROM portfolios WHERE "Portfolio" = '""" + eval('name') + """' """


def get_benchmark_names(name):
	return """ SELECT "FundName", "Weight", "Sector" FROM (
				(SELECT * FROM portfolios WHERE "Portfolio" = '""" + eval('name') + """' ) AS port
				LEFT JOIN (SELECT "ISINCode", "FundName", "Sector" FROM gcp_etf_lookup_view ) AS etfs
				USING("ISINCode")
				) AS map"""


def get_benchmark_holding_type(name):
	return """ SELECT "Portfolio", "HoldingType", "HoldingName", "Weight" FROM gcp_benchmark_holding_type WHERE "Portfolio" = '""" + eval('name') + """' AND "Rank" <=10
			 """

def get_benchmark_holding_top10(name):
	return """ SELECT "Portfolio", 'Top10' AS "HoldingType", "HoldingName", "Weight" FROM gcp_benchmark_holding_all WHERE "Portfolio" = '""" + eval('name') + """' AND "Rank" <= 10 ORDER BY "Portfolio", "HoldingType", "Weight" DESC """

def get_inception_date(isins):
	query = get_by_isin(isins, "gcp_ts_nav_tr", ["ISINCode", "TimeStamp", "IsWeek"])
	query += """ AND "IsWeek" = True """
	return """ SELECT MAX("Inception") FROM 
					(SELECT DISTINCT("ISINCode"), FIRST_VALUE("TimeStamp") OVER(PARTITION BY "ISINCode") AS "Inception" FROM (""" + query + """ ) AS c 
					) AS d
				"""
	
def get_ts_price(isins):
	query = get_by_isin(isins, 'gcp_ts_nav_tr', ["ISINCode", "TimeStamp", "NAV_USD", "TR_USD", "IsWeek"])

	query = """ SELECT "ISINCode", "TimeStamp", "NAV_USD", "TR_USD" FROM (""" + query + """ ) AS a WHERE "IsWeek" = True"""

	query_wk = """ SELECT * FROM ( """ + query + """ ) AS b WHERE "TimeStamp" >= 
				(SELECT MAX("Inception") FROM 
					(SELECT DISTINCT("ISINCode"), FIRST_VALUE("TimeStamp") OVER(PARTITION BY "ISINCode") AS "Inception" FROM (""" + query + """ ) AS c 
					) AS d
				)"""

	return """ SELECT * FROM ( """ + query_wk + """ ) AS nav 
				LEFT JOIN (SELECT "TimeStamp", "FX" AS "GBP" FROM fxes where "FXTicker" = 'GBP=') AS gbp
					USING("TimeStamp")
				LEFT JOIN (SELECT "TimeStamp", "FX" AS "EUR" FROM fxes where "FXTicker" = 'EUR=') AS eur
				USING("TimeStamp") """

def get_ts_div(isins, start_date):

	query = get_by_isin(isins, 'gcp_div', ['TimeStamp', 'ISINCode', 'Div_USD', 'Div_GBP', 'Div_EUR'])

	return """ SELECT * FROM ( """ + query + """ ) AS a WHERE "TimeStamp" >= '""" + eval('start_date') + """' """