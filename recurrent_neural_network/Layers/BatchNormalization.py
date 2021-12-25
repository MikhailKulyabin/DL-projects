import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        self.optimizer= None
        self.channels = channels
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        self.mean = None
        self.var = None
        self.change_shape = False

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
    
    
    def forward(self, input_tensor):
        if len(input_tensor.shape) == 4:    
            self.input_tensor = self.reformat(input_tensor)
            self.change_shape = True
        else:
            self.input_tensor = input_tensor


        if self.testing_phase == False:
            mean = np.mean(self.input_tensor, axis=0)
            var = np.var(self.input_tensor, axis=0)
            self.moving_average_estimation(mean, var)
            self.input_tensor_norm = (self.input_tensor-mean)/np.sqrt(var+np.finfo(float).eps)
        else:
            self.input_tensor_norm = (self.input_tensor-self.mean)/np.sqrt(self.var+np.finfo(float).eps)
        if self.change_shape:
            out = self.reformat(self.input_tensor_norm*self.weights + self.bias)
            self.change_shape = False
        else:
            out = self.input_tensor_norm*self.weights + self.bias
        return out


    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:    
            error_tensor = self.reformat(error_tensor)
            self.change_shape = True
        self.gradient_weights = np.sum(np.multiply(error_tensor, self.input_tensor_norm), axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, 
                                                            self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, 
                                                            self.gradient_bias)
        output = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, 
                             self.mean, self.var)
        if self.change_shape:
            output = self.reformat(output)
            self.change_shape = False
        return output
    
    def moving_average_estimation(self, mean, var, alpha=0.8):
        if self.mean is None:
            self.mean = mean
            self.var = var
        else:
            self.mean = alpha*self.mean + (1-alpha)*mean
            self.var = alpha*self.var + (1-alpha)*var
            
    def reformat(self, tensor):    
        if len(tensor.shape) == 4:
            self.B, self.H, self.M, self.N = tensor.shape
            tensor = tensor.reshape(self.B, self.H, self.M*self.N)
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape(self.B*self.M*self.N, self.H)
        else:
            tensor = tensor.reshape(self.B , self.M * self.N , self.H)
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape(self.B, self.H, self.M, self.N)
        return tensor

