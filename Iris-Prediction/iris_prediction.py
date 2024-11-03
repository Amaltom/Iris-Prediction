import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import logging

### Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IrisModel:
    def __init__(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.model = RandomForestClassifier()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train(self):
        logger.debug("Training has started....")
        print(type(self.X_train), type(self.y_train))
        print(self.X_train.shape, self.y_train.shape)
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def save_model(self, filename):
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        self.model = joblib.load(filename)


if __name__ == "__main__":
    #### Creating an instance of the model
    iris_model = IrisModel()
    iris_model.preprocess()
    iris_model.train()


    #### Evaluate the model
    accuracy = iris_model.evaluate()
    print(f"Model Accuracy: {accuracy:.2f}")

    #### Example prediction on a sample data point
    sample_data = [[5.1, 3.5, 1.4, 0.2]]
    prediction = iris_model.predict(sample_data)
    print(f"Predicted class for {sample_data}:{prediction[0]}")

    #### Save the model
    iris_model.save_model('iris_model.joblib')

