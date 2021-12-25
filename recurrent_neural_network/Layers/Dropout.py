import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        self.probability = probability
        
    def forward(self, input_tensor):
        if self.testing_phase == False:
            input_tensor = input_tensor*(1.0/self.probability)
            self.mask = np.where(np.random.random(input_tensor.shape) < 1-self.probability, 0, 1)
            input_tensor = input_tensor*self.mask
        return input_tensor
    
    def backward(self, error_tensor):
        if self.testing_phase == False:
            error_tensor = error_tensor*(1.0/self.probability)
            error_tensor = error_tensor*self.mask
        return error_tensor