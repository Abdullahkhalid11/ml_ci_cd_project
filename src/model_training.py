import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
import time
from datetime import datetime, timedelta

def fetch_live_data(symbol, interval='1m', period='1d'):
    """Fetch live data from Yahoo Finance."""
    try:
        data = yf.download(symbol, interval=interval, period=period)
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def preprocess_data(data):
    """Preprocess the data for model training."""
    # Calculate additional features
    data['Returns'] = data['Close'].pct_change()
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Drop rows with NaN values
    data.dropna(inplace=True)
    
    # Create target variable (next minute's return)
    data['Target'] = data['Returns'].shift(-1)
    
    return data

def train_model(data):
    """Train a Random Forest model on the data."""
    # Prepare features and target
    X = data[['Returns', 'SMA_5', 'SMA_20']]
    y = data['Target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse}")
    
    return model

def main():
    symbol = "AAPL"  # Pakistan Stock Exchange 100 Index
    
    # Initialize an empty DataFrame to store historical data
    historical_data = pd.DataFrame()
    
    while True:
        # Fetch live data
        live_data = fetch_live_data(symbol)
        
        # Append new data to historical data
        historical_data = pd.concat([historical_data, live_data])
        
        if not historical_data.empty:
            # Ensure the index is a DatetimeIndex
            if not isinstance(historical_data.index, pd.DatetimeIndex):
                historical_data.index = pd.to_datetime(historical_data.index)

            # Keep only the last week of data using .loc
            last_seven_days = historical_data.loc[pd.Timestamp.now() - pd.Timedelta(days=7):]
            
            # Preprocess the data
            processed_data = preprocess_data(last_seven_days)
            
            if not processed_data.empty:
                # Train the model
                model = train_model(processed_data)
                
                # Make a prediction for the next minute
                latest_data = processed_data.iloc[-1]
                prediction = model.predict([[latest_data['Returns'], latest_data['SMA_5'], latest_data['SMA_20']]])
                print(f"Prediction for next minute's return: {prediction[0]}")
        
        # Wait for 1 minute before the next iteration
        time.sleep(60)

if __name__ == "__main__":
    main()
