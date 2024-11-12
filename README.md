# Pakistan Stock Exchange (KSE100) Price Prediction Dashboard

This project provides real-time predictions for the KSE100 index of the Pakistan Stock Exchange. By leveraging live data from the yfinance API and a trained Random Forest Regressor model, this dashboard predicts stock prices based on several key indicators and displays them alongside actual prices for easy comparison. The project is deployed on [Vercel](https://vercel.com/) to enable easy accessibility and real-time performance monitoring.

## Features

- **Real-Time Data**: The dashboard fetches live KSE100 index data from yfinance, allowing up-to-date tracking of market trends.
- **Prediction Model**: A Random Forest Regressor model trained on KSE100 historical data predicts stock prices using features such as opening price, high, low, close, volume, returns, and moving averages (5-day and 20-day).
- **Data Visualization**: Visualizes both actual and predicted values on a candlestick chart, with a trend line overlay for predictions.
- **Model Metrics**: Displays Mean Squared Error (MSE) and R-squared score for a quick assessment of model accuracy.
  
## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Model Training](#model-training)
4. [Real-Time Dashboard](#real-time-dashboard)
5. [Deployment](#deployment)
6. [Getting Started](#getting-started)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Acknowledgments](#acknowledgments)

## Project Overview

The KSE100 Price Prediction Dashboard leverages machine learning to assist investors and analysts in understanding potential future prices of the KSE100 index. By training on historical data and using recent information, the model predicts the target price, giving insights into possible market trends. 

## Tech Stack

- **Backend**: Flask for API and model serving.
- **Frontend**: Plotly for interactive data visualization.
- **Machine Learning**: Random Forest Regressor from Scikit-learn.
- **Data Source**: yfinance for live data fetching.
- **Deployment**: Vercel.

## Model Training

The Random Forest Regressor model is trained on historical KSE100 data with the following features:

- **Open, High, Low, Close** prices
- **Volume** of shares traded
- **Returns** (percentage change)
- **MA5** and **MA20** (5-day and 20-day moving averages)

These features were selected based on their correlation with the target price. The model’s performance is evaluated using Mean Squared Error and R-squared metrics.

## Real-Time Dashboard

The Flask application serves the model’s predictions via an API endpoint, and the frontend displays this data alongside live actual prices. The dashboard includes:

- A **Candlestick chart** showing real-time and predicted prices.
- A **Line graph** overlaying the predicted trend.
- **Model metrics** for accuracy evaluation.

## Deployment

The project is deployed on [Vercel](https://vercel.com/), enabling public access to the dashboard. Vercel handles the server-side rendering of the Flask application, making it accessible and scalable.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- `pip` for managing Python packages

### Installation

1. Clone the repository:
  
   git clone https://github.com/YourUsername/KSE100-Price-Prediction.git
   cd KSE100-Price-Prediction

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Flask application locally:
   python app.py

4. Access the dashboard at http://localhost:5000.


### Usage

- View Real-Time Data: The main page displays real-time KSE100 data along with model predictions.
- Monitor Performance: Observe the MSE and R-squared metrics to assess the accuracy of the model.

### Project Structure

- ├── src
- │   ├── app.py               # Main Flask application
- │   ├── model.py             # Code for training and saving the model
- │   └── utils.py             # Helper functions for data fetching and processing
- ├── tests
- │   └── test_app.py          # Unit tests for API and model
- ├── requirements.txt         # Python dependencies
- ├── Dockerfile               # For containerization (optional)
- ├── Jenkinsfile              # CI/CD pipeline configuration
- ├── vercel.json              # Vercel deployment configuration
- └── README.md                # Project documentation


