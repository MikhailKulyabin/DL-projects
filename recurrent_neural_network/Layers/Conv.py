from  Layers.Initializers import UniformRandom
from scipy import signal
import numpy as np
from Layers.Base import BaseLayer

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        uniform = UniformRandom()
        self.weights = uniform.initialize((num_kernels, *convolution_shape), 
                                          convolution_shape[-2], convolution_shape[-1])
        self.bias = uniform.initialize((num_kernels, 1), 
                                          num_kernels, 1)
        self.gradient_weights = None
        self.gradient_bias = None
        self.optimizer = None
        
    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = (np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.weights = weights_initializer.initialize((self.num_kernels, *self.convolution_shape), 
                                          fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels, 1), 
                                          fan_in, fan_out)
        
    def upsampling(self, matrix):
        matrix_slice = np.zeros((self.shape))
        if isinstance(self.stride_shape, tuple):
            matrix_slice[np.arange(0, self.shape[0], self.stride_shape[0])[:,None],
                                        np.arange(0, self.shape[1], self.stride_shape[1])] = matrix
        else:
            matrix_slice[np.arange(0, self.shape[0], self.stride_shape[0])] = matrix
        return matrix_slice
    
    def downsampling(self, matrix):
        self.shape = matrix.shape
        if isinstance(self.stride_shape, tuple):
            matrix_slice = matrix[np.arange(0, matrix.shape[0], self.stride_shape[0])[:,None],
                                        np.arange(0, matrix.shape[1], self.stride_shape[1])]
        else:
            matrix_slice = matrix[np.arange(0, matrix.shape[0], self.stride_shape[0])]
        return matrix_slice
    
    def add_pad(self, batch, channel, channel_inp):
        if len(self.input_tensor[batch][channel_inp].shape) == 1:
            m = int((self.weights.shape[-1])/2)
            shape = [m, m]
            if self.weights.shape[-1] % 2 == 0:
                shape[1] -=1
        else:
            m = int((self.weights.shape[-2])/2)
            n = int((self.weights.shape[-1])/2)
            shape = [[m, m],
                     [n, n]]
            if self.weights.shape[-2] % 2 == 0:
                shape[0][1] -=1
            if self.weights.shape[-1] % 2 == 0:
                shape[1][1] -=1
        output = np.pad(self.input_tensor[batch][channel_inp],shape)
        return output
        
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        test = self.downsampling(input_tensor[0][0])
        output = np.zeros((input_tensor.shape[0], self.num_kernels, *test.shape))
        for batch in range(input_tensor.shape[0]):
            for kernel in range(self.num_kernels):
                for channel in range(input_tensor.shape[1]):
                    if channel == 0:
                        con = signal.correlate(input_tensor[batch][channel], 
                                      self.weights[kernel][channel], 
                                      mode='same')
                    else:
                        con += signal.correlate(input_tensor[batch][channel], 
                                      self.weights[kernel][channel], 
                                      mode='same')

                output[batch][kernel] = self.downsampling(con)
                try:
                    output[batch][kernel] += self.bias[kernel]
                except:
                    output[batch][kernel] += self.bias[0]
        return output
    
    def backward(self, error_tensor):
        # Gradient with respect to lower layers
        output = np.zeros((self.input_tensor.shape))
        for batch in range(self.input_tensor.shape[0]):
            for channel in range(self.input_tensor.shape[1]):
                for kernel in range(self.num_kernels):
                    error_tensor_up = self.upsampling(error_tensor[batch][kernel])
                    if kernel == 0:
                        con = signal.convolve(error_tensor_up, 
                                      self.weights[kernel][channel], 
                                      mode='same')
                    else:
                        con += signal.convolve(error_tensor_up, 
                                      self.weights[kernel][channel], 
                                      mode='same')
                output[batch][channel] = con
        # Gradient with respect to the weights
        gradient_weights = np.zeros((self.input_tensor.shape[0], *self.weights.shape))
        gradient_bias = np.zeros((self.input_tensor.shape[0], *self.bias.shape))
        for batch in range(self.input_tensor.shape[0]):
            for channel_inp in range(self.input_tensor.shape[1]):
                for channel in range(error_tensor.shape[1]):
                    gradient_bias[batch][channel] = np.sum(error_tensor[batch][channel])
                    error_tensor_up = self.upsampling(error_tensor[batch][channel])
                    input_tensor = self.add_pad(batch, channel, channel_inp)
                    gradient_weights[batch][channel][channel_inp] = signal.correlate(input_tensor,
                                                                                    error_tensor_up, 
                                                                                    mode='valid')
        self.gradient_weights = np.sum(gradient_weights, axis=0)
        self.gradient_bias = np.sum(gradient_bias, axis=0)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, 
                                                            self.gradient_weights)
            self.bias = self.optimizer.calculate_update(self.bias, 
                                                            self.gradient_bias)
            
        return output
