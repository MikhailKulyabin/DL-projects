import numpy as np
class CrossEntropyLoss:
    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(-np.log(np.sum(np.multiply(input_tensor, label_tensor), axis=1) 
                              + np.finfo(float).eps))
        
    def backward(self, label_tensor):
        return -label_tensor/self.input_tensor
        