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
    N = 40  # Number of data points
    data = pd.DataFrame({
        'Close': list(range(100, 100 + N)),
        'Open': list(range(99, 99 + N)),
        'High': list(range(101, 101 + N)),
        'Low': list(range(98, 98 + N)),
        'Volume': [1000 + i * 100 for i in range(N)],
    })
    assert not data.empty
    X_train, X_test, y_train, y_test = predictor.prepare_data(data)

    # Update expected rows to account for both pct_change and MA20, hence 20 rows are lost
    expected_rows = N - 20  # One row for pct_change and 19 for MA20
    actual_rows = len(X_train) + len(X_test)
    
    assert actual_rows == expected_rows, f"Expected {expected_rows} rows, got {actual_rows}"
    
def test_flask_app():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200