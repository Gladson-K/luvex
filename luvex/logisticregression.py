import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Logisticregression():

    def __init__(self,learn_rate,n_iter):                     #learn_rate and epochs are given as a input to system
        self.learn_rate = learn_rate
        self.n_iter = n_iter
        

    def fit_model(self, X, y):  
                                                                    #X for feature and y for label
        self.l,self.b = X.shape
        self.Weights = np.zeros(self.b)
        self.Bias = 0
        self.X = X
        self.y = y
        self.gradient_descent()
        for i in range(self.n_iter):
            self.gradient_descent()
        return self

        
   # Backward Propagation of Weights using Gradient Descent Algorithm
        

    def gradient_descent(self):

        z = np.dot(self.X,self.Weights)+self.Bias
        sigm = sigmoid(z)
        y_hat = (sigm- self.y.T)
        y_hat = np.reshape(y_hat,self.l)
        diff_W = (np.dot(self.X.T,y_hat))/self.l
        diff_b = np.sum(y_hat)/self.l

        self.Weights = self.Weights - (np.dot(self.learn_rate,diff_W))
        self.Bias =  self.Bias - (np.dot(self.learn_rate,diff_b))

        return self

    def predictions(self,X):
        
        z = np.dot(X,self.Weights)+self.Bias
        z_final = sigmoid(z)
        y_predict = np.where(z_final>0.5,1,0)
        self.gradient_descent()
        return y_predict

    def accu_score(self, X, y):
        acc_score = []
        self.fit_model(X, y)  
    
        for epoch in range(10):  
            y_pred = self.predictions(X)  
            acc = np.mean(y_pred == y) 
            print(f"Epoch {epoch + 1}: Accuracy = {acc:.2f}")
            acc_score.append(acc)
    
        print("Training has been completed.")
        return acc_score