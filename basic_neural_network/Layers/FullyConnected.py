import numpy as np


class FullyConnected:
    
    def __init__(self, input_size, output_size):
        self._optimizer = None
        self.weights = np.random.rand(input_size+1, output_size)
        self.gradient_weights = None
        
    def forward(self, input_tensor):

        input_tensor = np.append(input_tensor, 
                                 np.ones(input_tensor.shape[0]).reshape(-1, 1), 
                                 axis=1) 
        self.input_tensor = input_tensor
        return np.dot(input_tensor, self.weights)
    
    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, 
                                                            self.gradient_weights)
        return np.dot(error_tensor, self.weights[:-1:].T)
    
    @property
    def optimizer(self): 
        return self._optimizer
      
    @optimizer.setter    
    def optimizer(self, optimizer): 
        self._optimizer = optimizer
