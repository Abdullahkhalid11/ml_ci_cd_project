import yfinance as yf

def get_live_data():
    # Fetch live data from Pakistan Stock Exchange
    data = yf.download("^KSE100", period="1d", interval="1m")
    return data.iloc[-1].to_dict()  