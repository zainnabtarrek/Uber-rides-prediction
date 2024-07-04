from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pk', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    my_features = [np.array(features)]
    predictions = model.predict(my_features)
    result = f'Predicted result: {predictions[0]}'
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run()
