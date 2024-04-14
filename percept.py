import numpy as np
import pandas as pd
import array
from collections import Counter

class Perceptron:
    def __init__(self, feature_size):
        # initialize the weight matrix and store the learning rate
        self.W = np.zeros(feature_size) + 1
        self.bias = 1
        self.pred_labels = []     # Store prediction label for each Epcho
        self.pred_wrong = []      # Store index of instance that has wrong prediction each Epcho 
        self.current_weights = []   # store last weights
        self.train_acc = []         # Store accuracy on each Epcho during training
        self.eval_acc = []      # Store accuracy on Evaluation data on each Epcho during training

    def stop_criteria(self, actual_y, predicted_y):
        # Check all instances predicted correctly 
        for idx, y_val in enumerate(actual_y):    # Iterate over ground-truth instance
            if len(predicted_y) == 0 or y_val != predicted_y[idx]:   # if find any wrong prediction, return FALSE
                is_correct = False
                return is_correct
        is_correct = True  # in case all predicted TRUE.
        return is_correct
    
    def fit(self, X, Y, epochs=10, X_eval= None, Y_eval= None):

        # Iterate over entire instances 
        for epoch in range(epochs):

            if self.stop_criteria(Y, self.pred_labels): break # Check stop crieteria
            self.reset_store_parameter()  # Reset store parameter each Epcoch

            # Iterate over all instance 
            for idx in range(X.shape[0]):
                input_vectors = X.iloc[idx , :]
                
                # train the perceptron 
                sum_w = np.dot(input_vectors, self.W)  # calculate weights summation
                y_pred = self.step_function(sum_w)  # Activation Function 

                # calculate loss for current instance
                loss = y_pred - Y.iloc[idx] 

                # Store current 
                self.store_parameter(y_pred, self.W, loss if loss != 0 else None)

                # Update the weight
                self.learning_weight(sum_w, input_vectors, Y.iloc[idx], y_pred)

            self.train_acc.append(accuracy_metric(Y, self.pred_wrong))  # calculate the accuracy of current Epcho
            
            '''Do prediction on Evaluation data if evaluation data (X_eval) passed'''
            if X_eval is not None : 
                # Evaluate the network performance on Evaluation data
                preds_eval, wrong_preds_eval = self.predict_evaluate(X_eval, Y_eval)
                self.eval_acc.append(accuracy_metric(Y_eval, wrong_preds_eval))
            
        # Calculate  average accuracy
        # ------ TO DO --------
        
        # Print out the final weight
        print('FINAL WEIGHTS ARE:\n')
        for idx, w in enumerate(self.W):
            print(f'Vector({idx+1}) weight: {w}\n')


    ''' Make prediction on Evaluate data '''
    def predict_evaluate(self, x_eval, y_eval):
        preds = []
        wrong_preds = []
        for idx, ins in x_eval.iterrows():
            #inputs_vetcor = x_eval.iloc[idx , :] 
            pred = self.predict_(ins)
            if y_eval[idx] != pred: wrong_preds.append(idx)
            preds.append(pred)
        return preds, wrong_preds

    
    def predict(self, X_test):
        preds = []
        # iterate over each dataset, abd make prediction on it
        for idx, ins in X_test.iterrows():
            preds.append(self.predict_(ins))
        return preds

    def predict_(self, x):
        sum_weights = np.dot(x, self.W) + self.bias
        y_pred = self.step_function(sum_weights)
        return y_pred

    def store_parameter(self, y_pred, current_weights, wrong_pred_idx= None):
        self.pred_labels.append(y_pred)
        self.current_weights = current_weights
        if wrong_pred_idx != None: self.pred_wrong.append(wrong_pred_idx)

    def reset_store_parameter(self):
        self.pred_labels = []
        self.pred_wrong = []
    
    def step_function(self, x):
        return 1 if x > 0 else 0

    def learning_weight(self, sum_w, input_vectors, actual_y, pred_y):
        if actual_y == pred_y:  # No not update weight, if prediction correct 
            return
        elif sum_w < 0 and actual_y != pred_y:  # weight too hiegt
            self.W = input_vectors.values - self.W

        elif sum_w > 0 and actual_y != pred_y:  # weight too low
            self.W = input_vectors.values - self.W

def prep_data(data):
    data = pd.read_csv(data, sep=" ")
    data_df = pd.DataFrame(data, columns=data.columns)
    X = data_df.iloc[1:, :-1]
    Y = data_df.iloc[1:, -1]
    # convert target from Categorical to Numberical
    class_mapping = {'b': 0, 'g': 1}  # Mapping 'g' to 1 and 'b' to 0
    encoding_y = Y.map(class_mapping)

    return X, Y, encoding_y

def accuracy_metric(actual_y, wrong_preds):
    correct_preds = len(actual_y) - len(wrong_preds)
    acc = correct_preds / float(len(actual_y)) * 100.0
    return acc 
'''def accuracy_metric(self, y, wrong_preds):
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if wrong_preds.ndim == 1:
        wrong_preds = wrong_preds.reshape(1, -1)
    accuracy = np.mean(np.argmax(y, axis=0) == np.argmax(wrong_preds, axis=0))
    return accuracy * 100'''

''' Split data into training and Evaluation portion '''
def split_data(x, y):
    total = x.shape[0]
    idx = int(total / 2)
    x_train = x.iloc[:idx, :]
    x_eval = x.iloc[idx:, :]
    y_train = y.iloc[:idx]
    y_eval = y.iloc[idx:]
    return x_train, y_train, x_eval, y_eval


def main(): 
    input_user = 'ionosphere.data'  # Take file name from user

    X, Y, y_encoded = prep_data(input_user)  # Reading file and separete inout and target features
    feature_size = X.shape[1]  # Get input size

    x_train, y_train, x_eval, y_eval = split_data(X, y_encoded)

    Epochs = 20
    percept = Perceptron(feature_size)  # create object of perceptron class
    
    # Train the network
    #percept.fit(X, y_encoded, Epochs)   # Train the perceptron
    # Train the network and Evaluate at the same time
    percept.fit(x_train, y_train, Epochs, x_eval, y_eval)

    # Evaluate the model on test data




if __name__ == '__main__':
  main()