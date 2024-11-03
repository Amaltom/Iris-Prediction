from flask import Flask, request, render_template
from iris_prediction import IrisModel

app = Flask(__name__)
iris_model = IrisModel()
iris_model.preprocess()
iris_model.train()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    ### Preparing the data for prediction

    sample_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = iris_model.predict(sample_data)

    return render_template('result.html', prediction=prediction[0])

if __name__=="__main__":
    app.run(debug=True)
