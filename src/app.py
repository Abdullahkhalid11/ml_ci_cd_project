import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import threading
import time
import requests
from flask import Flask

# Step 1: Data Collection
def fetch_live_data():
    # Using yfinance to fetch data for Pakistan Stock Exchange (KSE100 index)
    ticker = "PSX"
    data = yf.Ticker(ticker).history(period="1d", interval="1m")
    return data

# Step 2: Model Training
class StockPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.latest_data = None
        self.metrics = {}

    def prepare_data(self, data):
        data['Returns'] = data['Close'].pct_change()
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['Target'] = data['Close'].shift(-1)
        data = data.dropna()

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA20']
        X = data[features]
        y = data['Target']

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, data):
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        self.metrics['mse'] = mean_squared_error(y_test, y_pred)
        self.metrics['r2'] = r2_score(y_test, y_pred)

    def predict(self, data):
        return self.model.predict(data[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA5', 'MA20']])

# Step 3: Flask Application
app = Flask(__name__)
predictor = StockPredictor()

def update_model():
    while True:
        data = fetch_live_data()
        predictor.train(data)
        predictor.latest_data = data
        time.sleep(300)  # Update every 5 minutes

@app.route('/some_route')
def some_function():
    # Example of adding timeout to requests
    response = requests.get('https://api.example.com', timeout=10)
@app.route('/')
def dashboard():
    if predictor.latest_data is None:
        return "No data available yet. Please wait for the first update."

    # Prepare data for plotting
    df = predictor.latest_data.reset_index()
    df['Prediction'] = predictor.predict(df)

    # Create candlestick chart
    candlestick = go.Candlestick(
        x=df['Datetime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Actual"
    )

    # Create line for predictions
    prediction_line = go.Scatter(
        x=df['Datetime'],
        y=df['Prediction'],
        name="Prediction",
        line=dict(color='red')
    )

    # Layout
    layout = go.Layout(
        title="Pakistan Stock Exchange (KSE100) - Live Data and Predictions",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Price")
    )

    # Combine the traces and create the figure
    fig = go.Figure(data=[candlestick, prediction_line], layout=layout)

    # Convert the plot to JSON
    plot_json = fig.to_json()

    return render_template('dashboard.html', plot_json=plot_json, metrics=predictor.metrics)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = predictor.predict(df)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    # Start the model update thread
    threading.Thread(target=update_model, daemon=True).start()
    
    # Run the Flask app
    app.run(debug=False)
    
    
   