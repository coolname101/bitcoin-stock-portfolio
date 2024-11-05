import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, max_error

import functions.functions.bitcoin_project as bp

st.set_page_config(page_title='Bitcoin Analysis', 
                   page_icon='docs/marketwise_logo.jpg',
                   initial_sidebar_state='collapsed', 
                   layout='wide', 
                   menu_items={'About': 'https://linkedin.com/in/itumeleng-kesebonye/'})

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

nav1, nav2, nav3, nav4 = st.columns([0.115, 0.115, 0.115, 0.6])

with nav1:
    st.page_link("app.py", label="BTC Dashboard", icon="🏠")

with nav2:
    st.page_link("pages/services.py", label="Stock Dashboard", icon="1️⃣")

with nav3:
    st.page_link("pages/info.py", label="Information", icon="2️⃣")

image, title = st.columns([0.25, 3])

with image:
    st.image('docs/marketwise_logo.jpg', 
         caption='', width=80)
    
with title:
    st.title('Stock Dashboard')



ticker = ['AAPL', 'GOOG', 'GME', 'INTC', 'GBPJPY=X', 'NVDA', 'BTC-USD', 'USDT-USD']  # List of asset tickers to choose from

col0, _, _ = st.columns([0.3,1,1])

with col0:
    option = st.selectbox("Please select ticker", ticker)  # Dropdown options to select asset of interest




new_df = bp.load_data(option)
new_df = bp.clean_data(new_df)
new_df = bp.feature_engineering(new_df)

# Creating date buttons for a specific interaction with graphs
date_buttons = [
    {'count': 1, 'step': 'day', 'stepmode': 'todate', 'label': '1D'},
    {'count': 5, 'step': 'day', 'stepmode': 'todate', 'label': '5D'},
    {'count': 1, 'step': 'month', 'stepmode': 'todate', 'label': '1M'},
    {'count': 3, 'step': 'month', 'stepmode': 'todate', 'label': '3M'},
    {'count': 6, 'step': 'month', 'stepmode': 'todate', 'label': '6M'},
    {'count': 1, 'step': 'year', 'stepmode': 'todate', 'label': 'YTD'}
]


col1, col2, col3 = st.columns(3)

# Plotting the close price of the asset of interest
fig1 = px.line(new_df, x=new_df.index, y='Close', height=500, title=f'{option} history price')
fig1.update_traces(line_color='#EE4E4E')
fig1.update_layout(xaxis_title='Date', yaxis_title='Price ($)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig1.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})

with col1:
    st.plotly_chart(fig1)

# Plotting the Relative Strength Index
fig2 = go.Figure(go.Scatter(x=new_df.index, y=new_df['rsi'], mode='lines'))
fig2.add_hline(y=70, row=2, col=1, annotation=
               {'text': 'Sell Threshhold', 'font': {'color': 'red'}, 'bgcolor': 'white'}, 
               line_color='red', line_dash='dash')
fig2.add_hline(y=30, row=2, col=1, annotation=
               {'text': 'Buy Threshhold', 'font': {'color': 'green'}, 'bgcolor': 'white'}, 
               line_color='green', line_dash='dash')
fig2.update_layout(height=500, title_text=f"{option} Price Relative Strength Index", paper_bgcolor='rgba(0,0,0,0)', 
                   plot_bgcolor='rgba(0,0,0,0)')
fig2.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
fig2.update_xaxes(title_text="Date")
fig2.update_yaxes(title_text="RSI", tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

with col2:
    st.plotly_chart(fig2)

# Coding colors for negative and positive macd_hist values for easy interpretation
new_df['color'] = np.where(new_df['macd_hist'] < 0, 'red', 'green')

# Plotting the Moving Average Convergence/Divergence Graph
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=new_df.index, y=new_df['macd'], name='MACD', mode='lines'))
fig3.add_trace(go.Scatter(x=new_df.index, y=new_df['macd_signal'], name='MACD Signal', 
                          mode='lines', line_color='orange', line=dict(dash='dash')))
fig3.add_trace(go.Bar(x=new_df.index, y=new_df['macd_hist'], marker_color=new_df['color'], name='MACD Hist'))
fig3.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
fig3.update_layout(height=500, title_text=f"{option} Moving Average Convergence/Divergence", 
                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig3.update_xaxes(title_text="Date")
fig3.update_yaxes(title_text="MACD")

with col3:
    st.plotly_chart(fig3)


# The candlestick slows down the app, below I create a show/hide button 
# so that it is only loaded when necessary.
show_hide = ['Hide', 'Show']

col3_1, _, _ = st.columns([0.3,1,1])

with col3_1:
    show_hide_button = st.selectbox("Select 'Show' to load Candlestick", show_hide)

col4, col5 = st.columns(2)

if show_hide_button == "Show":
    
    # Candlestick Graph
    fig4 = go.Figure(go.Candlestick(x=new_df.index, 
                                open=new_df['Open'], 
                                high=new_df['High'], 
                                low=new_df['Low'], 
                                close=new_df['Close']))
    fig4.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    fig4.update_layout(height=500, title_text=f"{option} Candlestick Indicator", 
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig4.update_xaxes(title_text="Date")
    fig4.update_yaxes(title_text="Price")
    
    with col4:
        st.plotly_chart(fig4)
    
    # Engulfing Plot
    fig5 = go.Figure(go.Scatter(x=new_df.index,
                            y=new_df['engulfing'], 
                            mode='lines'))
    fig5.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}}, 
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig5.update_layout(title_text=f"{option} Engulfing Signal")
    fig5.update_xaxes(title_text="Date")
    fig5.update_yaxes(title_text="Engulfing")
    
    with col5:
        st.plotly_chart(fig5)
    



# MODELING



X_prophet = bp.prepare_data(new_df)

X_prophet = bp.rename_columns(X_prophet)

split = int(len(new_df) * 0.75)

train_prophet, test_prophet = bp.split_train_val(X_prophet, 0.75)

model = bp.training_prophet(train_prophet)


train_prophet = pd.DataFrame(train_prophet)
test_prophet = pd.DataFrame(test_prophet)



forecast = bp.prophet_predict(train_prophet, test_prophet, model)



rmse = round(mean_squared_error(test_prophet['y'], forecast['yhat'][split:], squared=False), 3)
mse = round(mean_squared_error(test_prophet['y'], forecast['yhat'][split:]), 4)
r2 = round(r2_score(test_prophet['y'], forecast['yhat'][split:]), 6)
max_err = round(max_error(test_prophet['y'], forecast['yhat'][split:]), 3)

st.subheader("Before Error reduction")

col6, col7, col8, col9 = st.columns(4)


with col6:
    st.metric("RMSE", rmse)
    
with col7:
    st.metric("MSE", mse)
    
with col8:
    st.metric("R2 Score", r2)
    
with col9:
    st.metric("Max Error", max_err)

error = (mean_squared_error(test_prophet['y'], forecast['yhat'][split:], squared=False))
rmse_1 = round(mean_squared_error(test_prophet['y'], forecast['yhat'][split:] - error, squared=False), 3)
mse_1 = round(mean_squared_error(test_prophet['y'], forecast['yhat'][split:] - error), 4)
r2_1 = round(r2_score(test_prophet['y'], forecast['yhat'][split:] - error), 6)
max_err_1 = round(max_error(test_prophet['y'], forecast['yhat'][split:] - error), 3)

st.subheader("After Error reduction")
col6_1, col7_1, col8_1, col9_1 = st.columns(4)

with col6_1:
    st.metric("RMSE", rmse_1)

with col7_1:
    st.metric("MSE", mse_1)

with col8_1:
    st.metric("R2 Score", r2_1)
    
with col9_1:
    st.metric("Max Error", max_err_1)    

# error = round(mean_squared_error(test_prophet['y'], forecast['yhat'][split:], squared=False), 2)


adjusted = list(forecast['yhat'][split:] - error)
test_prophet['adjusted_yhat'] = adjusted
test_prophet['yhat'] = forecast['yhat'][split:]

comp_df = pd.concat([test_prophet['ds'], test_prophet['y'], test_prophet['yhat'], test_prophet['adjusted_yhat']], axis=1)

col10, col11 =st.columns(2)

# Plotting the training data to observe the errors made
fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=forecast['ds'], y=train_prophet['y'], name='True Price', mode='lines', line_color='#D20062'))
fig6.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'][:split], name='Predicted Price', mode='lines', line_color='#4D96FF'))
fig6.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
fig6.update_layout(height=500, title_text=f"{option} Price Prediction on Training Data", 
                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig6.update_xaxes(title_text="Date")
fig6.update_yaxes(title_text="Price")

with col10:
    st.plotly_chart(fig6)

fig7 = go.Figure()
fig7.add_trace(go.Scatter(x=comp_df['ds'], y=comp_df['y'], name='True Value', mode='lines', line_color='#D20062'))
fig7.add_trace(go.Scatter(x=comp_df['ds'], y=comp_df['yhat'], name='Forecasted Price (Without Error Reduction)', mode='lines', line_color='#9B7EBD'))
fig7.add_trace(go.Scatter(x=comp_df['ds'], y=comp_df['adjusted_yhat'], name='Forecasted Price', mode='lines', line_color='#4D96FF'))
fig7.update_layout(height=500, title_text=f"{option} Price Prediction on Test Data", 
                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig7.update_xaxes(title_text="Date")
fig7.update_yaxes(title_text="Price")

with col11:
    st.plotly_chart(fig7)



col12, _, _ = st.columns([2,1,1])

with col12:
    future_dates_file = st.file_uploader(label="Upload a CSV file with future dates split into 'year', 'month', 'day' columns", type=['csv'])



if future_dates_file is not None:
    
    X_prophet['cap'] = X_prophet['y'] * 1.2
    X_prophet['floor'] = X_prophet['y'] * 0.8
    
    future_dates_df = pd.read_csv(future_dates_file)
    
    dates_length = len(future_dates_df)


    
    model = bp.training_prophet(X_prophet)

    future = model.make_future_dataframe(periods=dates_length, freq='B')
    future_df = pd.DataFrame()
    future['cap'] = X_prophet['cap']
    future['floor'] = X_prophet['floor']
    future['rsi'] = X_prophet['rsi']
    future['macd'] = X_prophet['macd']
    future['macd_signal'] = X_prophet['macd_signal']
    future['macd_hist'] = X_prophet['macd_hist']
    future['engulfing'] = X_prophet['engulfing']
    future_df['cap'] = bp.predict_col(X_prophet, future_dates_df, 'cap')['cap']
    future_df['floor'] = bp.predict_col(X_prophet, future_dates_df, 'floor')['floor']
    future_df['rsi'] = bp.predict_col(X_prophet, future_dates_df, 'rsi')['rsi']
    future_df['macd'] = bp.predict_col(X_prophet, future_dates_df, 'macd')['macd']
    future_df['macd_signal'] = bp.predict_col(X_prophet, future_dates_df, 'macd_signal')['macd_signal']
    future_df['macd_hist'] = bp.predict_col(X_prophet, future_dates_df, 'macd_hist')['macd_hist']
    future_df['engulfing'] = bp.predict_col_class(X_prophet, future_dates_df, 'engulfing')['engulfing'].astype(float)
    #future = pd.concat([future, future_df], axis=0, ignore_index=True)
    future['cap'].iloc[-dates_length:] = future_df['cap']
    future['floor'].iloc[-dates_length:] = future_df['floor']
    future['rsi'].iloc[-dates_length:] = future_df['rsi']
    future['macd'].iloc[-dates_length:] = future_df['macd']
    future['macd_signal'].iloc[-dates_length:] = future_df['macd_signal']
    future['macd_hist'].iloc[-dates_length:] = future_df['macd_hist']
    future['engulfing'].iloc[-dates_length:] = future_df['engulfing']
    #st.dataframe(future)
    
    forecast = model.predict(future)
    
    forecast['adjusted_yhat'] = forecast['yhat'] - error
    
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=X_prophet['ds'], y=X_prophet['y'], name='True Value', mode='lines', line_color='#4D96FF'))
    fig8.add_trace(go.Scatter(x=forecast['ds'][-dates_length:], y=forecast['yhat'][-dates_length:], name='Forecasted Price', mode='lines', line_color='#D20062'))
    fig8.update_layout(height=500, title_text=f"{option} Price Prediction", 
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig8.update_layout({'xaxis': {'rangeselector': {'buttons': date_buttons}}})
    fig8.update_xaxes(title_text="Date")
    fig8.update_yaxes(title_text="Price")
    st.plotly_chart(fig8)

    _, col13, _ = st.columns(3)
    
    with col13:
        st.download_button(label="Download forecast", 
                           data=forecast[['ds', 'yhat', 'adjusted_yhat']].iloc[-dates_length:].to_csv().encode("utf-8"), 
                           file_name=f"{option} Price Forecast.csv")
