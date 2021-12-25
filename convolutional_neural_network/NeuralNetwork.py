from Layers import *
import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        
    def forward(self):
        input_tensor, label_tensor = self.data_layer.forward()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        self.output = self.loss_layer.forward(input_tensor, label_tensor)
        self.label_tensor = label_tensor
        return self.output
    
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
    
    def append_trainable_layer(self, layer):
        layer.initialize(self.weights_initializer, self.bias_initializer)
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
    
    def train(self, iterations):
        for iteration in range(iterations):
            result = self.forward()
            self.loss.append(result)
            self.backward()
            
    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
        
        
        
            
