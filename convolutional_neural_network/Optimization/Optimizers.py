import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate*gradient_tensor
        return updated_weights


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.momentum = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if (self.momentum is None) or (np.shape(self.momentum) != np.shape(gradient_tensor)):
            self.momentum = np.zeros(np.shape(gradient_tensor))
        self.momentum = self.momentum_rate*self.momentum - self.learning_rate*gradient_tensor
        updated_weights = weight_tensor + self.momentum
        return updated_weights


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.iter = 1
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros(np.shape(gradient_tensor))
        if self.r is None:
            self.r = np.zeros(np.shape(gradient_tensor))
        try:
            self.v = self.mu*self.v + (1-self.mu)*gradient_tensor
        except:
            self.v = np.zeros(np.shape(gradient_tensor))
            self.v = np.zeros(np.shape(gradient_tensor))
            self.v = self.mu*self.v + (1-self.mu)*gradient_tensor
        try:
            self.r = self.rho*self.r + (1-self.rho)*np.multiply(gradient_tensor, gradient_tensor)
        except:
            self.r = np.zeros(np.shape(gradient_tensor))
            self.r = self.rho*self.r + (1-self.rho)*np.multiply(gradient_tensor, gradient_tensor)
        # bias correction
        v = np.divide(self.v, (1-self.mu**self.iter))
        r = np.divide(self.r, (1-self.rho**self.iter))
        updated_weights = weight_tensor - self.learning_rate*np.divide(v, np.sqrt(r)+np.finfo(float).eps)
        self.iter += 1
        return updated_weights
