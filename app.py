#Importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
# For technical analysis
#import pandas_datareader as web
import functions.functions.bitcoin_project as bp


st.logo('docs/marketwise_logo.jpg')

#Loading the data
bitcoin = pd.read_csv('data/bitcoin-usd.csv', parse_dates=['date'])
sp500 = pd.read_csv('data/sp500.csv', parse_dates=['date'])
gold_cpi = pd.read_csv('data/monthly_data.csv', parse_dates=['date'])


st.set_page_config(page_title='Bitcoin Analysis', 
                   page_icon='docs/marketwise_logo.jpg',
                   initial_sidebar_state='collapsed', 
                   layout='wide', 
                   menu_items={'About': 'https://linkedin.com/in/itumeleng-kesebonye/'})

nav1, nav2, nav3, nav4 = st.columns([0.115, 0.115, 0.115, 0.6])

background = """
<style>
    .stApp {
        background: linear-gradient(-45deg, #F3F8FF, #F5F5F5, #FCFAEE, #EEF7FF);
	    background-size: 400% 400%;
	    animation: gradient 15s ease infinite;
	    height: 100vh;
    }
    
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    .st-emotion-cache-79elbk {
        background: rgba(0,0,0,0));
     }
    
    st-emotion-cache-1mi2ry5 {
        background-color: #FCFAEE;
    }
    
    .st-emotion-cache-1l269bu {
        background-color: rgba(0,0,0,0);
    }
    
    @keyframes gradient {
	0% {
		background-position: 0% 50%;
	}
	50% {
		background-position: 100% 50%;
	}
	100% {
		background-position: 0% 50%;
	}
}

</style>
"""

st.markdown(background, unsafe_allow_html=True)

with nav1:
    st.page_link("app.py", label="BTC Dashboard", icon="🏠")

with nav2:
    st.page_link("pages/services.py", label="Stock Dashboard", icon="1️⃣")

with nav3:
    st.page_link("pages/info.py", label="Information", icon="2️⃣")


image, title = st.columns([0.25, 3])

with image:
    st.image('docs/kanchanara-dRgxo-ujT2U-unsplash.jpg', 
         caption='', width=100)
    
with title:
    st.title('Bitcoin Dashboard')


# Creating date buttons for a specific interaction with graphs
date_buttons = [
    {'count': 1, 'step': 'day', 'stepmode': 'todate', 'label': '1D'},
    {'count': 5, 'step': 'day', 'stepmode': 'todate', 'label': '5D'},
    {'count': 1, 'step': 'month', 'stepmode': 'todate', 'label': '1M'},
    {'count': 3, 'step': 'month', 'stepmode': 'todate', 'label': '3M'},
    {'count': 6, 'step': 'month', 'stepmode': 'todate', 'label': '6M'},
    {'count': 1, 'step': 'year', 'stepmode': 'todate', 'label': 'YTD'}
]

col_1_1, col_1_2 = st.columns(2)

col_2_1, col_2_2, col_2_3, col_2_4 = st.columns(4)



with col_1_1:
    st.subheader('Historic Prices')
    
with col_2_1:
    fig1 = px.line(bitcoin, x='date', y='close', height=350, title='Bitcoin history price')
    fig1.update_traces(line_color='#C80036')
    fig1.update_layout(xaxis_title='Date', yaxis_title='Price ($)', 
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig1.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig1)

with col_2_2:
    fig2 = px.line(sp500, x='date', y='close', height=350, title='S&P 500 history price')
    fig2.update_traces(line_color='#0E46A3')
    fig2.update_layout(xaxis_title='Date', yaxis_title='Price ($)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig2.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig2)
    

with col_2_3:
    fig3 = px.line(gold_cpi, x='date', y='gold_usd', height=350, title='Gold history price')
    fig3.update_traces(line_color='#8644A2')
    fig3.update_layout(xaxis_title='Date', yaxis_title='Price ($)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig3.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig3)
    
with col_2_4:
    fig4 = px.line(gold_cpi, x='date', y='cpi_us', height=350, title='CPI history price')
    fig4.update_traces(line_color='#240A34')
    fig4.update_layout(xaxis_title='Date', yaxis_title='Price ($)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig4.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig4)

col3, _ = st.columns(2)

col_3_1_1, col_3_1_2, col_3_1_3 = st.columns(3)

bitcoin['date'] = pd.to_datetime(bitcoin['date'])
bitcoin = bitcoin.set_index(['date'])
btc_daily_returns = bitcoin['close'].pct_change() * 100
btc_weekly_first = bitcoin['close'].resample('5D').first()
btc_weekly_last = bitcoin['close'].resample('5D').last()

btc_weekly_returns = round(((btc_weekly_last - btc_weekly_first) / btc_weekly_first) * 100, 2) 

weekly_rolling_volatility = btc_daily_returns.rolling(5).std() * np.sqrt(252)

with col3:
    st.subheader('Returns & Votality')

with col_3_1_1:
    btc_daily_returns = btc_daily_returns.dropna()
    fig5 = px.line(btc_daily_returns.reset_index(), x='date', y='close', height=350, title='Daily Returns of Bitcoin')
    fig5.update_traces(line_color="#C80036")
    fig5.update_layout(xaxis_title="Date", yaxis_title='% Returns', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig5.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig5)
    
with col_3_1_2:

    fig6 = px.line(btc_weekly_returns.reset_index(), x='date', y='close', height=350, title='Weekly Returns of Bitcoin')
    fig6.update_traces(line_color='#C80036')
    fig6.update_layout(xaxis_title='Date', yaxis_title='% Returns', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig6.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig6)
    
with col_3_1_3:
    
    fig7 = px.line(weekly_rolling_volatility.reset_index(), x='date', y='close', height=350, title='Weekly Rolling Volatility of Bitcoin')
    fig7.update_traces(line_color='#C80036')
    fig7.update_layout(xaxis_title='Date', yaxis_title='% Volatility', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig7.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig7)
    
    

col_3_2_1, col_3_2_2, col_3_2_3 = st.columns(3)

sp500['date'] = pd.to_datetime(sp500['date'])
sp500 = sp500.set_index(['date'])
sp_daily_returns = sp500['close'].pct_change() * 100
sp_weekly_first = sp500['close'].resample('5D').first()
sp_weekly_last = sp500['close'].resample('5D').last()    

sp_weekly_returns = round(((sp_weekly_last - sp_weekly_first) / sp_weekly_first) * 100, 2)

sp_weekly_rolling_volatility = sp_daily_returns.rolling(5).std() * np.sqrt(252)

with col_3_2_1:
    sp_daily_returns = sp_daily_returns.dropna().reset_index()
    fig5_1 = px.line(sp_daily_returns, x='date', y='close', height=350, title='Daily Returns of S&P 500')
    fig5_1.update_traces(line_color="#0E46A3")
    fig5_1.update_layout(xaxis_title="Date", yaxis_title='% Returns', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig5_1.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig5_1)
    
with col_3_2_2:
    fig6_1 = px.line(sp_weekly_returns.reset_index(), x='date', y='close', height=350, title='Weekly Returns of S&P 500')
    fig6_1.update_traces(line_color='#0E46A3')
    fig6_1.update_layout(xaxis_title='Date', yaxis_title='% Returns', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig6_1.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig6_1)
    
with col_3_2_3:
    fig7_1 = px.line(sp_weekly_rolling_volatility.reset_index(), x='date', y='close', height=350, title='Weekly Rolling Volatility of S&P 500')
    fig7_1.update_traces(line_color='#0E46A3')
    fig7_1.update_layout(xaxis_title='Date', yaxis_title='% Volatility', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig7_1.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig7_1)

    
    


st.subheader('Correlation Scores of Bitcoin and S&P 500')

full_df = bitcoin.merge(sp500, how='left', on='date', suffixes=['_bitcoin', '_sp500'])
fig8 = px.scatter(full_df, x='close_sp500', y='close_bitcoin', title='Relationship between Bitcoin and the  S&P 500 Prices', trendline='lowess', trendline_options=dict(frac=0.08))
fig8.update_traces(marker=dict(color='#B60071'))
fig8.update_layout(xaxis_title='S&P 500 Price $', yaxis_title='Bitcoin Price $', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig8)

col_4_1, col_4_2, col_4_3, col_4_4, col_4_5, col_4_6 = st.columns(6)


with col_4_1:
    st.metric("Pearson Price", 
              value=(round(full_df[['close_sp500', 'close_bitcoin']].corr().iloc[0,1], 3)))

with col_4_2:
    st.metric("Spearman Price", 
              value=(round(full_df[['close_sp500', 'close_bitcoin']].corr(method='spearman').iloc[0,1], 3)))

with col_4_3:
    st.metric("Kendall Price", 
              value=(round(full_df[['close_sp500', 'close_bitcoin']].corr(method='kendall').iloc[0,1], 3)))

with col_4_4:
    st.metric("Pearson Volume",
              value=(round(full_df[['volume_sp500', 'volume_bitcoin']].corr().iloc[0,1], 3)))

with col_4_5:
    st.metric("Spearman Volume",
              value=(round(full_df[['volume_sp500', 'volume_bitcoin']].corr(method='spearman').iloc[0,1], 3)))
    
with col_4_6:
    st.metric("Kendall Volume",
              value=(round(full_df[['volume_sp500', 'volume_bitcoin']].corr(method='kendall').iloc[0,1], 3)))




st.subheader('Average True Range')


bitcoin_atr = bp.calculate_atr(bitcoin)

sp500_atr = bp.calculate_atr(sp500)


col_5_1, col_5_2 = st.columns(2)

with col_5_1:
    
    fig9 = px.line(bitcoin_atr.reset_index(), x='date', y=0, title='Average True Range of Bitcoin Price')
    fig9.update_traces(line_color='#C80036')
    fig9.update_layout(xaxis_title='Date', yaxis_title='Price $', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig9.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig9)
    
with col_5_2:
    fig10 = px.line(sp500_atr.reset_index(), x='date', y=0, title='Average True Range of S&P 500 Price')
    fig10.update_traces(line_color='#0E46A3')
    fig10.update_layout(xaxis_title='Date', yaxis_title='Price $', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig10.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    st.plotly_chart(fig10)
    
    