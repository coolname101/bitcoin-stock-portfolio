#Importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# For technical analysis
import talib as ta
import pandas_datareader as web
import datetime as dt


#Loading the data
bitcoin = pd.read_csv('data/bitcoin-usd.csv', parse_dates=['date'])
sp500 = pd.read_csv('data/sp500.csv', parse_dates=['date'])
gold_cpi = pd.read_csv('data/monthly_data.csv', parse_dates=['date'])


st.set_page_config(page_title='Bitcoin Analysis', 
                   page_icon='docs/marketwise_logo.jpg', 
                   layout='wide', 
                   menu_items={'About': 'https://linkedin.com/in/itumeleng-kesebonye/'})

nav1, nav2, nav3, nav4 = st.columns(4)

background = """
<style>
    .stApp {
        background: rgb(223,104,222);
        background: linear-gradient(90deg, rgba(223,104,222,1) 0%, rgba(153,153,245,1) 35%, rgba(0,255,196,1) 100%);
    }
    
    .st-emotion-cache-1l269bu {
        background-color: #F6F5F5;
    }
</style>
"""

st.markdown(background, unsafe_allow_html=True)

with nav1:
    st.page_link("setup.py", label="Home", icon="🏠")

with nav2:
    st.page_link("pages/services.py", label="Services", icon="1️⃣")

with nav3:
    st.page_link("pages/info.py", label="Information", icon="2️⃣")

with nav4:
    st.page_link("http://www.google.com", label="Google", icon="🌎")


st.title('Bitcoin Stock Analysis')
st.logo('docs/marketwise_logo.jpg')
st.image('docs/kanchanara-dRgxo-ujT2U-unsplash.jpg', 
         caption='Photo by Kanchanara on Splash', width=1000)
st.file_uploader(label='CSV file of the stock to be analysed')

col_1_1, col_1_2 = st.columns(2)

col_2_1, col_2_2, col_2_3, col_2_4 = st.columns(4)



with col_1_1:
    st.subheader('Dashboard')
    
with col_2_1:
    fig1 = px.line(bitcoin, x='date', y='close', height=350, title='Bitcoin history price')
    fig1.update_traces(line_color='#EE4E4E')
    fig1.update_layout(xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig1)

with col_2_2:
    fig2 = px.line(sp500, x='date', y='close', height=350, title='S&P 500 history price')
    fig2.update_traces(line_color='#0E46A3')
    fig2.update_layout(xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig2)
    

with col_2_3:
    fig3 = px.line(gold_cpi, x='date', y='gold_usd', height=350, title='Gold history price')
    fig3.update_traces(line_color='#8644A2')
    fig3.update_layout(xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig3)
    
with col_2_4:
    fig4 = px.line(gold_cpi, x='date', y='cpi_us', height=350, title='CPI history price')
    fig4.update_traces(line_color='#240A34')
    fig4.update_layout(xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig4)


col_3_1, col_3_2, col_3_3 = st.columns(3)

bitcoin['date'] = pd.to_datetime(bitcoin['date'])
bitcoin = bitcoin.set_index(['date'])
btc_daily_returns = bitcoin['close'].pct_change() * 100

with col_3_1:
    btc_daily_returns = btc_daily_returns.dropna()
    fig5 = px.line(btc_daily_returns.reset_index(), x='date', y='close', height=350, title='Daily Returns of Bitcoin')
    fig5.update_traces(line_color="#C80036")
    fig5.update_layout(xaxis_title="Date", yaxis_title='% Returns')
    st.plotly_chart(fig5)
    
with col_3_2:
    weekly_rolling_mean = btc_daily_returns.rolling(5).mean()
    fig6 = px.line(weekly_rolling_mean.reset_index(), x='date', y='close', height=350, title='Weekly Rolling Mean of Bitcoin')
    fig6.update_traces(line_color='#EF5A6F')
    fig6.update_layout(xaxis_title='Date', yaxis_title='% Returns')
    st.plotly_chart(fig6)
    
with col_3_3:
    weekly_rolling_volatility = btc_daily_returns.rolling(5).std() * np.sqrt(252)
    fig7 = px.line(weekly_rolling_volatility.reset_index(), x='date', y='close', height=350, title='Weekly Rolling Volatility of Bitcoin')
    fig7.update_traces(line_color='#E4003A')
    fig7.update_layout(xaxis_title='Date', yaxis_title='% Volatility')
    st.plotly_chart(fig7)


full_df = bitcoin.merge(sp500, how='left', on='date', suffixes=['_bitcoin', '_sp500'])
fig8 = px.scatter(full_df, x='close_sp500', y='close_bitcoin', title='Relationship between Bitcoin and the  S&P 500 Prices', trendline='lowess', trendline_options=dict(frac=0.08))
fig8.update_traces(marker=dict(color='#B60071'))
fig8.update_layout(xaxis_title='S&P 500 Price $', yaxis_title='Bitcoin Price $')
st.plotly_chart(fig8)

st.subheader('Correlation Scores of Bitcoin and S&P 500')
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






def calculate_atr(data, window=5):
  """
  This function calculates the Average True Range (ATR) for a given DataFrame.

  Args:
      data (pandas.DataFrame): DataFrame containing daily price data with columns for 'High', 'Low', 'Close'.
      window (int, optional): The window size for ATR calculation. Defaults to 14.

  Returns:
      pandas.Series: A Series containing the ATR values for each day.
  """
  high_low = data['high'] - data['low']
  high_close = abs(data['high'] - data['close'].shift(1))
  low_close = abs(data['low'] - data['close'].shift(1))
  true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
  atr = true_range.rolling(window=window).mean()
  return atr.dropna()

bitcoin_atr = calculate_atr(bitcoin)
#st.dataframe(bitcoin_atr)
fig9 = px.line(bitcoin_atr.reset_index(), x='date', y=0, title='Average True Range of Bitcoin Price')
fig9.update_traces(line_color='#FF0000')
fig9.update_layout(xaxis_title='Date', yaxis_title='Price $')
st.plotly_chart(fig9)