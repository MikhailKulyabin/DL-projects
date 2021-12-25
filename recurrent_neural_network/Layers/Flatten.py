import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        pass
    def forward(self, input_tensor):
        self.shape = np.shape(input_tensor)
        return input_tensor.ravel().reshape(self.shape[0], -1)
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)