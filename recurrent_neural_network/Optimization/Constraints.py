import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
        
    def calculate_gradient(self, weights):
        return  weights*self.alpha
    
    def norm(self, weights):
        return np.linalg.norm(weights.reshape(-1), ord=2)*self.alpha
        
class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
        
    def calculate_gradient(self, weights):
        return weights*self.alpha
    
    def norm(self, weights):
        return np.linalg.norm(weights.reshape(-1), ord=1)*self.alpha
    