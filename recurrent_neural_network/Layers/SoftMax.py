import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def forward(self, input_tensor):
        norm_input_tensor = input_tensor-np.max(input_tensor, axis=1).reshape(-1, 1)
        self.y_pred = np.exp(norm_input_tensor)/np.sum(np.exp(norm_input_tensor), 
                                                       axis=1).reshape(-1, 1)
        return self.y_pred
    
    def backward(self, error_tensor):
        return self.y_pred*(error_tensor-np.sum(error_tensor*self.y_pred, axis=1).reshape(-1, 1))