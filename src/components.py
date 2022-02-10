import streamlit as st
import pandas as pd
import altair as alt

import plotly.express as px
import plotly.graph_objects as go



@st.cache(suppress_st_warning=True)
def create_benchmark_table(benchmarks):
	styles = [
    dict(selector="th", props=[("background-color", "#123FED"),
    							("color", "white"),
                               ("text-align", "center")]),
    dict(selector="caption", props=[("caption-side", "bottom")])
	]

	return benchmarks.style.format({'1 Month %': "{:.1%}"}, na_rep="-")\
				.bar(subset=['1 Month %'], align = 'mid', color = ['#d65f5f', '#0DAA5B'], height = 60)\
				.hide(axis='index')
				#.set_table_styles(styles)
	
@st.cache(allow_output_mutation=True)
def create_frontier_graph(data):
	alt_frontier = alt.Chart(data).mark_point(filled=True, size=200).encode(
		x = alt.X("AnnualisedReturn:Q", scale=alt.Scale(zero=False)),
		y = alt.Y("Volatility:Q",  scale=alt.Scale(zero=False)),
		color = "Period:N",
		tooltip = ['Portfolio', 'Period', alt.Tooltip('AnnualisedReturn', format = '.2%'), alt.Tooltip('Volatility', format = '.2%')]
		).properties(height=500)

	return alt_frontier



def load_default_portfolio(port):
	st.write('**Holdings**', "(", len(st.session_state.portfolio), ")" , key="holding_count")

	col_header1, col_header2, col_header3, col_header4, col_header5, col_header6, col_header7= st.columns([3, 2, 1, 1, 1, 1, 2])

	with col_header1:
		st.write("**ETF Name**", key="col_name")

	with col_header2:
		st.write("**Sector**", key="col_sector")

	with col_header3:
		st.write("**NAV**", key="col_nav")

	with col_header4:
		st.write("**AUM**", key="col_aum")

	with col_header5:
		st.write("**Cost**", key="col_cost")

	with col_header6:
		st.write("**Rank**", key="col_rank")

	with col_header7:
		st.write("**Weight**", "(", st.session_state.port_weight, "% )")


	col_list = ['Name', 'Sector', 'NAV', 'AUM','Cost', 'Rank', 'Weight']
	col_names = ['FundName', 'Sector', 'NAV', 'AUM', 'Cost', 'Rank5', 'Weight']
	for k, v in port.items():
		cols = st.columns([3, 2, 1, 1, 1, 1, 2])
		for j, col in enumerate(cols):
			col.write(v[col_names[j]])
	
	return


def customise_portfolio(port, reweight, remove_etf):

	col_header1, col_header2, col_header3, col_header4, col_header5, col_header6, col_header7, col_header8= st.columns([3, 2, 1, 1, 1, 1, 2, 1])

	with col_header1:
		st.write("**ETF Name**", key="col_name")

	with col_header2:
		st.write("**Sector**", key="col_sector")

	with col_header3:
		st.write("**NAV**", key="col_nav")

	with col_header4:
		st.write("**AUM**", key="col_aum")

	with col_header5:
		st.write("**Cost**", key="col_cost")

	with col_header6:
		st.write("**Rank**", key="col_rank")

	with col_header7:
		st.write("**Weight**", "(", st.session_state.port_weight, "% )")


	total_weight = 0
	col_list = ['Name', 'Sector', 'NAV', 'AUM','Cost', 'Rank']
	col_names = ['FundName', 'Sector', 'NAV', 'AUM', 'Cost', 'Rank5']
	for i, k in enumerate(port):
		cols = st.columns([3, 2, 1, 1, 1, 1, 2, 1])
		for j, col in enumerate(cols[:-2]):
			col.write(port[k][col_names[j]])	

		cols[-2].number_input("", min_value = 0.0, max_value = 100.0, value=port[k]['Weight'], step = 1.0, key=k, on_change=reweight, args=(k, ))
		cols[-1].button("❌", key='delete'+str(i), on_click=remove_etf, args = (k, port[k]['FundName'], ))

	return



@st.cache
def draw_radar_graph(data1, data2, categories, legend1, legend2):
	fig = go.Figure()
	rng = ['#123FED', '#FFC400']

	fig.add_trace(go.Scatterpolar(
		      r=data1,
		      theta=categories,
		      fill='toself',
		      name=legend1,
		      marker_color = rng[0],
		      hovertemplate = "%{theta}: %{r}"
			)
			)
	fig.add_trace(go.Scatterpolar(
      		r=data2,
      		theta=categories,
      		fill='toself',
      		name=legend2,
      		marker_color = rng[1],
      		hovertemplate = "%{theta}: %{r}"
		))
	fig.update_layout(
		  	polar=dict(
		    	radialaxis=dict(
		    		visible=True,
			      	range=[0, 100],
		    	)),
		  	font_family = 'Sans-serif',
		  	font_size = 14,
		  	title_font_family='Sans-serif'
  			
		  #showlegend=True, 	  
		)
	fig.layout.plot_bgcolor = '#F7F8FA'
	return fig


@st.cache
def draw_pie_graph(data):
	fig = go.Figure()
	fig = px.pie(data, values='Weight', names='Sector', 
						hover_data =['FundName'],
						labels = {'FundName': 'ETF'},	
						hole=.3,
						color_discrete_sequence=px.colors.cyclical.Edge)
	fig.update_traces(
		textposition='inside',
		)
	fig.update_layout(
		uniformtext_minsize=12, 
		uniformtext_mode='hide',
		font_family = 'Sans-serif',
		font_size = 14,
		title_font_family='Sans-serif'
		)

	return fig



@st.cache(allow_output_mutation=True)
def draw_holding_graph(data, x_name, y_name, row_name, dom ,title, is_customise=False):
	if is_customise:
		tooltips = [y_name, row_name, alt.Tooltip(x_name, format = '.2%')]
	else:
		tooltips = [y_name, 'Name:N', row_name, alt.Tooltip(x_name, format = '.2%')]

	bars = alt.Chart(data).mark_bar().encode(
				x = alt.X(x_name +':Q', axis = alt.Axis(format='.0%')),
				y = alt.Y(y_name,sort='descending',title=None),
				color = alt.Color(y_name, scale=alt.Scale(domain=dom), legend=None),
				row = alt.Row(row_name+':N', sort=alt.EncodingSortField(field='Weight', order='descending'), title=None, spacing=5),
				tooltip = tooltips
			).properties(title = title)

	legend = alt.Chart(data).mark_bar().encode(
		#x = alt.X(x_name +':Q', axis = alt.Axis(format='.0%')),
		color = alt.Color(y_name, scale=alt.Scale(domain=dom)),
		).properties(
		height=35, width = 250)
	return bars, legend


@st.cache(allow_output_mutation=True)
def draw_full_holding_graph(data):
	slider_page = alt.binding_range(min=1, max=len(data)/20, step=1, name='Number of holdings (20/page):')
	selector_page = alt.selection_single(name="PageSelector", fields=['page'],
                                    bind=slider_page, init={'page': 1})
	base = alt.Chart(data).transform_calculate(
    			combined=alt.datum.Flag + '    ' + alt.datum.Name
    		).encode(
    			x = alt.X('Weight:Q', axis = alt.Axis(format='.1%')),
				y = alt.Y('combined:N', sort='-x')			
    		)
	bars = base.mark_bar(align='left', color='#123FED').encode(
    			tooltip = ['Name', 'Country', 'Sector', alt.Tooltip('Weight', format = '.2%')]
    		).properties(height=alt.Step(30))

	text = base.mark_text(align='right', color='white').encode(
			text = alt.Text('Sector:N')
			)

	return (bars + text).transform_window(
        		rank = 'rank(Weight)',
    		).add_selection(
        		selector_page
    		).transform_filter(
        		'(datum.rank > (PageSelector.page - 1) * 20) & (datum.rank <= PageSelector.page * 20)'
    		).properties(
    			title = """All portfolio holdings"""
    		)


@st.cache(allow_output_mutation=True)
def performance_graph(data, y_col, dom, num_format, color_col = 'Type'):
	nearest = alt.selection(type='single', nearest=True, on='mouseover',
			fields=['Dates'], empty='none')

	base = alt.Chart(data).mark_line().encode(
			x=alt.X('Dates:T', title=None ,axis=alt.Axis(format = ("%d-%b-%Y"))),
			y = alt.Y(y_col + ':Q', title=None, axis=alt.Axis(format= num_format), scale=alt.Scale(zero=False)),
			color=alt.Color( color_col + ':N', scale=alt.Scale(domain=dom))
		)


	selectors = alt.Chart(data).mark_point().encode(
			x='Dates:T',
			opacity=alt.value(0),
		).add_selection(
			nearest
		)

	points = base.mark_point().encode(
			opacity=alt.condition(nearest, alt.value(1), alt.value(0))
		)

	text = base.mark_text(align='left', dx=-50, dy=10).encode(
			text=alt.condition(nearest, alt.Text(y_col +':Q', format=num_format), alt.value(' ')),
		)

	rules = alt.Chart(data).mark_rule(color='#E7E8F0').encode(
			x = 'Dates:T'
		).transform_filter(
		nearest)

	return alt.layer(
			base, selectors, points, text, rules
			).properties(
			height = 500
			)


@st.cache(allow_output_mutation=True)
def performance_bar_graph(data, y_col, dom, num_format, color_col = 'Type'):
	base = alt.Chart(data).mark_bar().encode(
		alt.X('Dates:T', axis=alt.Axis(format = ("%d-%b-%Y"))),
		y = alt.Y(y_col +':Q', axis=alt.Axis(format= num_format ), scale=alt.Scale(zero=False)),
		color=alt.Color(color_col+':N', scale=alt.Scale(domain=dom), legend=None),
		).properties(title='Dividend Income')

	return base

@st.cache(allow_output_mutation=True)
def performance_overlay_graph(data1, data2, y_col, dom1, dom2, num_format, color_col):
	data = pd.concat([data1, data2], axis=0, ignore_index=True)
	dom = data['Name'].unique()

	num = len(data1['Dates'].unique())
	bar_len = 800/num
	if num <=12:
		x_label = alt.X('Dates:O')
	else:
		x_label = alt.X('Dates:T', axis=alt.Axis(format='%b-%Y'))

	selection = alt.selection_single(
    	fields=['Dates'], nearest=True, on='mouseover', empty='none', clear='mouseout'
		)

	base = alt.Chart(data1).mark_bar(size=bar_len).encode(
		x = x_label
		)

	bars = base.encode(
		y = alt.Y(y_col +':Q', stack='zero' ,axis=alt.Axis(format= num_format ), sort=alt.EncodingSortField(field='Name', order='ascending')),
		color=alt.Color(color_col+':N', scale=alt.Scale(domain=dom1)),
		)
	points1 = bars.mark_point().transform_filter(selection)
	
	lines = alt.Chart(data2).mark_line(color='dark').encode(
				x = x_label,
				y = alt.Y(y_col +':Q', axis=alt.Axis(format= num_format,labelAlign='left' ), title='Portfolio returns', ),
				color = alt.Color(color_col+':N', scale=alt.Scale(domain=dom2))
			)
	points2 = lines.mark_point().transform_filter(selection)
	

	rules = alt.Chart(data).encode(x=x_label).transform_pivot(
		'Name', value='Return', groupby=['Dates']
		).mark_rule().encode(
			opacity=alt.condition(selection, alt.value(0.3), alt.value(0)),
			tooltip = [alt.Tooltip(c, type='quantitative', format='.2%') for c in dom]
		).add_selection(selection)



	#return bars.properties(height=500)
	return alt.layer(bars + points1 + rules, lines + points2).resolve_scale(y='independent').properties(height=500)

@st.cache(allow_output_mutation=True)
def performance_grouped_bar_graph(data, y_col, title, dom ,period_order, min_max, y_label_show = True, x_label_show=False):
	bars = alt.Chart(data).mark_bar().encode(
					x = alt.X('Type:N', title = None, sort="descending", axis=alt.Axis(labels=x_label_show, tickSize=0)),
					y = alt.Y(y_col + ':Q', title = None, axis=alt.Axis(format='%', labels=y_label_show), scale=alt.Scale(domain=min_max)),
					color = alt.Color('Type:N',scale=alt.Scale(domain=dom), legend=None),
					#column = alt.Column(column_col + ':O', title=None, sort = period_order),
					tooltip = alt.Tooltip(y_col + ':Q', format=".2%")
					).properties(
						title = title,
						height = 400
					).configure_title(fontSize=14, orient = 'bottom')

	return bars



@st.cache(allow_output_mutation=True)
def drawdown_graph(drawdown, crisis, port_name):
	alt_dd = performance_graph(drawdown, 'Drawdown', [port_name, 'Benchmark'] ,'.1%')	
	alt_crisis = alt.Chart(crisis).mark_rect(color = "#E7E8F0", opacity = 0.2).encode(
			x = 'Start:T',
			x2 = 'End:T',
			)
	alt_crisis_txt = alt_crisis.mark_text(align='left', dy=190, size=18).encode(
			text = alt.Text('Event:O')
			)

	return alt_crisis + alt_dd + alt_crisis_txt

@st.cache(allow_output_mutation=True)
def distribution_graph(data):
	alt_dis = alt.Chart(data).mark_bar().encode(
				x = alt.X('Return:Q', bin = alt.Bin(maxbins=50), axis=alt.Axis(format='.0%')),
				y = 'count()',
				color = alt.Color('Sign:N', scale=alt.Scale(domain=['Positive', 'Negative'], range=["green", "red"]))
			).properties(height=400)
	return alt_dis
			

