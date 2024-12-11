import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import talib as ta
# import mplfinance as mpf
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def load_data(ticker):
    """
    Load yahoo Finance stock/asset historic data given the designated ticker name.
    
    param ticker: yahoo Finance ticker symbol for asset.
    return: dataframe
    
    >>> load_data('APPL')
    DataFrame 
    """
    
    data = yf.Ticker(ticker=ticker)  # Loading the data from Yahoo Finance API using the ticker
    dataframe = data.history(period='max', interval='1d')  # Loading the maximum daily history prices available
    dataframe = pd.DataFrame(dataframe.reset_index())  # Reseting the index to have the date as a column instead of index
    
    return dataframe


@st.cache_data
def clean_data(dataframe):
    """
    Clean the date column by converting to pandas datetime and setting as index 
    for easy time series manipulation and plotting.
    
    param dataframe: a time series dataframe with column 'Date'
    return: cleaned dataframe with column 'Date' as index
    
    >>> clean_data(dataframe=appl_df)
    DataFrame
    """
    
    # Ensuring that the date column is set to pandas datetime data-type for accurate datetime manipulation
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe = dataframe.set_index(['Date'])  # Setting the date column as index
    dataframe = dataframe.asfreq('b')  # Setting the frequency of the dataset to business week days
    dataframe = dataframe.fillna(method='ffill')  # Filling the missing values with a foward fill method
    
    return dataframe


@st.cache_data
def feature_engineering(dataframe):
    """
    Add Relative Strength Index, Moving Average Convergence Divergence and Engulfing
    to the columns for a robust analysis.
    
    param dataframe: a time-series dataframe with columns 'Open', 'High', 'Low' and 'Close'.
    return: dataframe with 'rsi', 'macd', 'macd_signal', 'macd_hist' and 'engulfing' as 
            additional columns
            
    >>> feature_engineering(dataframe=appl_df)
    DataFrame
    """
    
    dataframe['rsi'] = ta.RSI(dataframe['Close'])
    dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = ta.MACD(dataframe['Close'])
    dataframe['engulfing'] = ta.CDLENGULFING(dataframe['Open'], dataframe['High'], dataframe['Low'], dataframe['Close'])
    
    return dataframe

@st.cache_data
def scale_features(dataframe):
    
    scale_cols = ['Open', 'High', 'Low', 'rsi', 'macd', 'macd_signal', 'macd_hist']
    
    scaler = MinMaxScaler()
    dataframe[scale_cols] = scaler.fit_transform(dataframe[scale_cols])
    
    return dataframe
    


@st.cache_data
def prepare_data(dataframe):
    """
    Rescale the 'engulfing' column to 0 and 1 and ensures the 'Date' column is converted to
    the datetime.date format.
    
    param dataframe: a time-series with column 'Date' set as the index and an additional column
          named 'engulfing'.
    return: dataframe with 'Date' no longer set as index
    
    >>> prepare_data(dataframe=new_df)
    DataFrame
    """
    
    dataframe = dataframe
    dataframe['engulfing'] = dataframe['engulfing'] / 100
    dataframe = dataframe.reset_index()
    dataframe['Date'] = [datetime.date(i) for i in dataframe['Date']]
    dataframe = dataframe.fillna(0.0)
    
    return dataframe


@st.cache_data
def rename_columns(dataframe):
    """
    Rename column names to names that are expected by a Prophet Model.
    
    param dataframe: a dataframe with columns 'Date', 'High', 'Low', 'Close'
    return: a dataframe with the above columns renamed.
    
    >>> rename_columns(dataframe)
    DataFrame
    """
    
    dataframe = dataframe.rename(columns={'Date': 'ds',
                                                                  'High': 'cap',
                                                                  'Low': 'floor',
                                                                  'Close': 'y'})
    
    return pd.DataFrame(dataframe)


@st.cache_data
def split_train_val(dataframe, frac=0.99):
    """
    Split the dataframe to be modeled into train and validation sets using frac as fraction split.
    
    param dataframe: dataframe with column names 'cap', 'floor' and 'y'.
          frac: float value between 0 and 1, but not including 0 or 1.
          
    return: train dataframe and test dataframe split by frac value
    """
    split = int(len(dataframe) * frac)
    
    train_data = dataframe[:split]
    val_data = dataframe[split:]
    
    train_data = pd.DataFrame(train_data)
    val_data = pd.DataFrame(val_data)
    
    train_data['cap'] = train_data['y'] * 1.2
    train_data['floor'] = train_data['y'] * 0.2
    
    val_data['cap'] = val_data['y'] * 1.2
    val_data['floor'] = val_data['y'] * 0.2
    
    return train_data, val_data

@st.cache_data
def split_sarima_data(dataframe, frac=0.75):
    
    split = int(len(dataframe) * frac)
    
    train = dataframe[:split]
    test = dataframe[split:]
    
    return train, test




@st.cache_data
def training_prophet(dataframe):
    """
    Train dataframe using the Meta's Prophet model.
    
    param dataframe: a dataframe with columns 'ds', 'cap', 
    'floor', 'y', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'engulfing'.
    
    return: a trained Prophet model object.
    
    >>> training_prophet(dataframe)
    prophet.forecaster.Prophet object
    """
    
    model = Prophet(growth='logistic')
    model.add_regressor('rsi')
    model.add_regressor('macd')
    model.add_regressor('macd_signal')
    model.add_regressor('macd_hist')
    model.add_regressor('engulfing')
    
    model.fit(dataframe)
    
    return model


@st.cache_data
def prophet_predict(train_set, val_set, _trained_model):
    """
    Predict same number of days into the future as the length of the val_set.
    
    param train_set: training set used to train the trained_model
          val_set: validation set to be used to validate the results of the trained_model
          trained_model: model trained on train_set.
          
    return: A dataframe consisting of all the columns provided by a Prophet model.
    
    >>> prophet_predict(train_set=train_data, val_set=val_data, trained_model=model)
    DataFrame
    """
    
    split = len(val_set)
    future = _trained_model.make_future_dataframe(periods=len(val_set), freq='B')
    future['cap'] = train_set['cap']
    future['floor'] = train_set['floor']
    future['rsi'] = train_set['rsi']
    future['macd'] = train_set['macd']
    future['macd_signal'] = train_set['macd_signal']
    future['macd_hist'] = train_set['macd_hist']
    future['engulfing'] = train_set['engulfing']
    future['cap'].iloc[-split:] = val_set['cap']
    future['floor'].iloc[-split:] = val_set['floor']
    future['rsi'].iloc[-split:] = val_set['rsi']
    future['macd'].iloc[-split:] = val_set['macd']
    future['macd_signal'].iloc[-split:] = val_set['macd_signal']
    future['macd_hist'].iloc[-split:] = val_set['macd_hist']
    future['engulfing'].iloc[-split:] = val_set['engulfing']
    
    prediction = _trained_model.predict(future)
    
    return prediction


@st.cache_data
def predict_col(dataframe, dataframe2, colname):
    """
    Uses the date column called 'ds' to train a RandomForestRegressor and predict 
    future values of 'colname'.
    
    params dataframe: to be used as training data.
           dataframe2: to be used as input for RandomForestRegressor.predict() method.
    return: dataframe of columns 'date' and colname with length similar to dataframe2
    
    >>> predict_col(dataframe=new_df, dataframe2=future_dates_df, colname='cap')
    DataFrame
    """
    
    dataframe['ds'] = pd.to_datetime(dataframe['ds'])
    dataframe['year'] = dataframe['ds'].dt.year
    dataframe['month'] = dataframe['ds'].dt.month
    dataframe['day'] = dataframe['ds'].dt.day
    
    rf = RandomForestRegressor(random_state=44)
    rf.fit(dataframe[['year', 'month', 'day']], dataframe[colname])
    
    future_df = pd.DataFrame()
    future_df['date'] = dataframe2.year.astype(str) + '-' + dataframe2.month.astype(str) + '-' + dataframe2.day.astype(str)
    future_df['date'] = pd.to_datetime(future_df['date'])
    future_df[colname] = rf.predict(dataframe2)
    
    return future_df


@st.cache_data
def predict_col_class(dataframe, dataframe2, colname):
    """
    Uses the date column called 'ds' to train a RandomForestClassifier predict 
    future values of 'colname'.
    
    params dataframe: to be used as training data.
           dataframe2: to be used as input for RandomForestClassifier.predict() method.
    return: dataframe of columns 'date' and colname with length similar to dataframe2
    
    >>> predict_col(dataframe=new_df, dataframe2=future_dates_df, colname='engulfing')
    DataFrame
    """
    
    dataframe['ds'] = pd.to_datetime(dataframe['ds'])
    dataframe['year'] = dataframe['ds'].dt.year
    dataframe['month'] = dataframe['ds'].dt.month
    dataframe['day'] = dataframe['ds'].dt.day
    
    rf = RandomForestClassifier(random_state=44)
    rf.fit(dataframe[['year', 'month', 'day']], dataframe[colname])
    
    future_df = pd.DataFrame()
    future_df['date'] = dataframe2.year.astype(str) + '-' + dataframe2.month.astype(str) + '-' + dataframe2.day.astype(str)
    future_df['date'] = pd.to_datetime(future_df['date'])
    future_df[colname] = rf.predict(dataframe2)
    
    return future_df

@st.cache_data
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

@st.cache_data
def training_sarima(dataframe):
    """
    
    """
    
    predictors = ['Open', 'High', 'Low', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'engulfing']    
    sarimax_model = SARIMAX(dataframe['Close'], exog=dataframe[predictors], seasonal_order=(1,1,1,52))
    
    return sarimax_model.fit()


@st.cache_data
def forecast_sarima(dataframe, trained_sarima):
    """
    
    """
    
    predictors = ['Open', 'High', 'Low', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'engulfing']
     
    return trained_sarima.forecast(steps=(len(dataframe)), exog=dataframe[predictors])