from flask import Flask, render_template, request
import pandas as pd
from model import train_and_predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return "No file uploaded"

    df = pd.read_csv(file)
    prediction, fig_path = train_and_predict(df)

    return render_template('result.html', prediction=prediction, plot_url=fig_path)

if __name__ == '__main__':
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)

