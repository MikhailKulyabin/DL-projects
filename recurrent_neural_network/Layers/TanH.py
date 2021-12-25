import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def forward(self, input_tensor):
        self.activation =np.tanh(input_tensor)
        return self.activation
    
    def backward(self, error_tensor):
        return np.multiply((1 - self.activation**2), error_tensor) 