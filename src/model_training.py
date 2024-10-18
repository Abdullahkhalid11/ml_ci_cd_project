from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

def train_model():
    # Load historical data
    data = pd.read_csv("historical_data.csv")
    features = ['Datetime','Close']
    X = data.drop(features, axis=1)
    y = data['Close']
    
    model = RandomForestRegressor()
    #model.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    #metrics = {"accuracy": model.score(X_test, y_test)}
    
    return model

def predict(model, data):
    return model.predict([list(data.values())])[0]