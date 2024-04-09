import numpy as np
import pandas as pd
import array
from collections import Counter

class Perceptron:
    def __init__(self, feature_size):
        # initialize the weight matrix and store the learning rate
        self.W = np.zeros(feature_size) + 1
        self.pred_labels = []
        self.pred_wrong = []
        self.current_weights = []

    def stop_criteria(self, x):
        # if all instances predicted correctly 
        return True if x > 0 else False
    
    def fit(self, X, Y, epochs=10):

        # Iterate over entire instances 
        for epoch in range(epochs):
            # Check stop crieteria -------------------

            # Iterate over all instance 
            for idx in range(X.shape[0]):
                input_vectors = X.iloc[idx , :]
                # train the perceptron 
                sum_w = np.dot(input_vectors, self.W)  # calculate weights summation
                y_pred = self.step_function(sum_w)  # Activation Function 

                # calculate loss for current instance
                loss = y_pred - Y.iloc[idx] 

                # Store current ( y_pred, wrong pred ) 
                self.store_parameter(y_pred, self.W, loss if loss != 0 else None)

                # Update the weight
                self.learning_weight(sum_w, input_vectors, Y.iloc[idx], y_pred)



    def store_parameter(self, y_pred, current_weights, wrong_pred_idx= None):
        self.pred_labels.append(y_pred)
        self.current_weights = current_weights
        if wrong_pred_idx != None: self.pred_wrong.append(wrong_pred_idx)

    
    def step_function(self, x):
        return 1 if x > 0 else 0

    def learning_weight(self, sum_w, input_vectors, actual_y, pred_y):
        if actual_y == pred_y:  # No not update weight, if prediction correct 
            return
        elif sum_w < 0 and actual_y != pred_y:  # weight too hiegt
            self.w = input_vectors.values - self.W

        elif sum_w > 0 and actual_y != pred_y:  # weight too low
            self.w = input_vectors.values - self.W

def prep_data(data):
    data = pd.read_csv(data, sep=" ")
    data_df = pd.DataFrame(data, columns=data.columns)
    X = data_df.iloc[1:, :-1]
    Y = data_df.iloc[1:, -1]
    # convert target from Categorical to Numberical
    class_mapping = {'b': 0, 'g': 1}  # Mapping 'g' to 1 and 'b' to 0
    encoding_y = Y.map(class_mapping)

    return X, Y, encoding_y


def main(): 
    input_user = 'ionosphere.data'  # Read file from user

    X, Y, y_encoded = prep_data(input_user)  # Reading file and separete inout and target features
    feature_size = X.shape[1]  # Get input size

    Epochs = 20
    percept = Perceptron(feature_size)  # create object of perceptron class
    percept.fit(X, y_encoded, Epochs)   # Train the perceptron



if __name__ == '__main__':
  main()