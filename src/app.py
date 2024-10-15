from flask import Flask, jsonify, request, render_template
from model_training import train_model, predict
from data_collection import get_live_data

app = Flask(__name__)

model = train_model()

@app.route('/')
def dashboard():
    live_data = get_live_data()
    prediction = predict(model, live_data)
    metrics = {"accuracy": model.score(X_test, y_test)}
    return render_template('dashboard.html', data=live_data, prediction=prediction, metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict_instance():
    data = request.json
    prediction = predict(model, data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)