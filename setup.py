import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

col5, col6 = st.columns(2)

col1, col2, col3, col4 = st.columns(4)



with col5:
    st.subheader('History Prices')
    
with col1:
    bitcoin = pd.read_csv('data/bitcoin-usd.csv', parse_dates=['date'])
    fig1 = px.line(bitcoin, x='date', y='close', height=350, title='Bitcoin history price')
    fig1.update_traces(line_color='#EE4E4E')
    fig1.update_layout(xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig1)

with col2:
    sp500 = pd.read_csv('data/sp500.csv', parse_dates=['date'])
    fig2 = px.line(sp500, x='date', y='close', height=350, title='S&P 500 history price')
    fig2.update_traces(line_color='#0E46A3')
    fig2.update_layout(xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig2)
    
gold_cpi = pd.read_csv('data/monthly_data.csv', parse_dates=['date'])

with col3:
    fig3 = px.line(gold_cpi, x='date', y='gold_usd', height=350, title='Gold history price')
    fig3.update_traces(line_color='#8644A2')
    fig3.update_layout(xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig3)
    
with col4:
    fig4 = px.line(gold_cpi, x='date', y='cpi_us', height=350, title='CPI history price')
    fig4.update_traces(line_color='#240A34')
    fig4.update_layout(xaxis_title='Date', yaxis_title='Price ($)')
    st.plotly_chart(fig4)