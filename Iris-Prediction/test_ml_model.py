import pandas as pd
import pytest
from iris_prediction import IrisModel
import os
import numpy as np


@pytest.fixture(scope='module')
def setup_data():
    data = {
        'feature1': [5.1,4.9,4.7,4.6,5.0,5.4],
        'feature2': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9],
        'feature3': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7],
        'feature4': [0.2, 0.2, 0.2, 0.2, 0.4, 0.4],
        'target': [0,0,0,0,1,1]
    }

    df = pd.DataFrame(data)
    df.to_csv('test_data.csv', index=False)
    yield
    #### Clean up of CSV file after tests are done
    import os
    os.remove('test_data.csv')

@pytest.fixture
def model(setup_data):
    ## Initialize and train the IrisModel
    m = IrisModel()
    m.X = pd.read_csv('test_data.csv').iloc[:,:-1].values
    m.y = pd.read_csv('test_data.csv').iloc[:,-1].values
    m.preprocess()
    m.train()
    return m

def test_model_accuracy(model):
    accuracy = model.evaluate()
    assert accuracy >= 0.5, "Model accuracy should be at least 50%"


def test_predict(model):
    sample_data = np.array([[5.1,3.5, 1.4, 0.2]])
    prediction = model.predict(sample_data)
    assert prediction in [0,1,2], "Prediction should be one of the Iris classes (0,1 or 2)"


def test_predict_shape(model):
    sample_data = np.array([[5.1, 3.5,1.4,0.2], [5.9, 3.0, 5.1, 1.8]])
    predictions = model.predict(sample_data)
    assert predictions.shape == (2, ), "Prediction shape should match the number of the samples"

if __name__ == '__main__':
    pytest.main()