import jax 
import jax.numpy as jnp
import numpy as np

class LinearRegression():

    def __init__(self):
        self.name = 'Linear Regression Model'
        self.theta = None

    def h(self,x,theta):
        return np.matmul(x, theta)

    def loss_fn(self, x,y,theta):
        return ((self.h(x, theta)-y).T@(self.h(x, theta)-y))/(2*y.shape[0])
    
    def gradient_descent(self, x, y, theta, learning_rate, num_epochs):
        m = x.shape[0]
        J_all = []
        
        for _ in range(num_epochs):
            h_x = self.h(x, theta)
            cost_ = (1/m)*(x.T@(h_x - y))
            theta = theta - (learning_rate)*cost_
            J_all.append(self.loss_fn(x, y, theta))

        return theta, J_all 
        
    def train(self, X_train, y_train, learning_rate = 0.1, num_epochs = 10):
        self.theta = np.random.normal(0,1,(X_train.shape[1],1))
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        theta, J_all = self.gradient_descent(X_train, y_train, self.theta, learning_rate, num_epochs)
        return self

    def predict(self, X_test):
        X_test = X_test.to_numpy()
        outs = np.matmul(X_test, self.theta)
        return outs.reshape(outs.shape[0])

    def regularise(self):
        pass

"""
model = LinearRegression() # include params here 
model.train(X_train,y_train) 
model.predict(X_test) # gets you a singular value
"""