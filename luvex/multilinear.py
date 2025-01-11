from sklearn.base import BaseEstimator

import numpy as np
class MultipleLinear(BaseEstimator):

    def __init__(self,learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        self.X = X
        self.y = y
        

    def predict(self):
        m = self.y.size
        theta = np.zeros((self.X.shape[1],1))
        cost_list=[]

        for i in range(self.iterations):

            y_pred = np.dot(self.X, theta)
            cost = (1/(2*m))*np.sum(np.square(y_pred-self.y))

            d_theta = (1/m)*np.dot(self.X.T, y_pred-self.y)
            theta = theta - (self.learning_rate*d_theta)
            cost_list.append(cost)

            #print cost for every 10 iterations
            if(i%(self.iterations/10)==0):
                print("Cost is :", cost)

        return theta, cost_list

    def score(self):
        pred, cost = self.predict_cost()
        return cost

    def get_params(self, deep=True):
        # Return hyperparameters as a dictionary
        return {"learning_rate": self.learning_rate, "iterations": self.iterations}

    def set_params(self, **params):
        # Set hyperparameters
        for param, value in params.items():
            setattr(self, param, value)
        return self