import pytest
from src.app import app, StockPredictor
import pandas as pd

def test_basic():
    assert True

def test_import():
    from src.app import app
    assert app is not None

def test_stock_predictor():
    predictor = StockPredictor()
    assert predictor is not None

def test_prepare_data():
    predictor = StockPredictor()
    data = pd.DataFrame({
        'Close': [100, 101, 102, 103, 104],
        'Open': [99, 100, 101, 102, 103],
        'High': [101, 102, 103, 104, 105],
        'Low': [98, 99, 100, 101, 102],
        'Volume': [1000, 1100, 1200, 1300, 1400]
    })
    X_train, X_test, y_train, y_test = predictor.prepare_data(data)
    assert len(X_train) + len(X_test) == len(data) - 1  # One row lost due to pct_change

def test_flask_app():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200