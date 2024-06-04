import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Bitcoin Analysis', 
                   page_icon='docs/marketwise_logo.jpg', 
                   layout='wide', 
                   menu_items={'About': 'https://linkedin.com/in/itumeleng-kesebonye/'})

#st.page_link("setup.py", label="Home", icon="🏠")
#st.page_link("pages/services.py", label="Page 1", icon="1️⃣")
#st.page_link("pages/info.py", label="Page 2", icon="2️⃣", disabled=True)
#st.page_link("http://www.google.com", label="Google", icon="🌎")


st.title('Bitcoin Stock Analysis')
st.logo('docs/marketwise_logo.jpg')
st.image('docs/kanchanara-dRgxo-ujT2U-unsplash.jpg', 
         caption='Photo by Kanchanara on Splash')
st.file_uploader(label='CSV file of the stock to be analysed')
st.subheader('Data Preview')
col1, col2 = st.columns(2)
with col1:
    st.text('Bitcoin Data')
    bitcoin = pd.read_csv('data/bitcoin-usd.csv', parse_dates=['date'])
    st.dataframe(bitcoin)

with col2:
    st.text('S&P 500 Data')
    sp500 = pd.read_csv('data/sp500.csv', parse_dates=['date'])
    st.dataframe(sp500)

st.text('Gold & CPI Data')  
gold_cpi = pd.read_csv('data/monthly_data.csv', parse_dates=['date'])
st.dataframe(gold_cpi)

fig = plt.figure(figsize=(12,6))
plt.plot(bitcoin['date'], bitcoin['close'], label='bitcoin')
plt.plot(sp500['date'], sp500['close'], label='S&P 500')
plt.plot(gold_cpi['date'], gold_cpi['gold_usd'], label='Gold')
plt.plot(gold_cpi['date'], gold_cpi['cpi_us'], label='CPI')
plt.title('Price History')
plt.xlabel('Date')
plt.ylabel('Price $')
plt.legend()
plt.grid()

st.pyplot(fig)