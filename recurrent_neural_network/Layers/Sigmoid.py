import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def forward(self, input_tensor):
        self.activation = 1.0/(1+np.exp(-input_tensor))
        return self.activation
    
    def backward(self, error_tensor):
        return np.multiply((1 - self.activation)*self.activation, error_tensor)