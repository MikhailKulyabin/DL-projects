import numpy as np
from Layers.TanH import TanH 
from Layers.Sigmoid import Sigmoid
from Layers.FullyConnected import FullyConnected

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False
        self.regularizer = None
        self.optimizer = None
        self.hidden_state = np.zeros(self.hidden_size)
        self.l1 = FullyConnected(self.input_size+self.hidden_size, self.hidden_size)
        self.l2 = FullyConnected(self.hidden_size,self.output_size)
        self.weights = self.l1.weights
        
    def initialize(self, weights_initializer, bias_initializer):
        self.l1.initialize(weights_initializer, bias_initializer)
        self.l2.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        if self.memorize == False:
            self.hidden_state = np.zeros(self.hidden_size)
        self.y = []
        self.tanh = []
        self.sigmoid = []
        self.l1_input = []
        self.l2_input = []
        for time in range(input_tensor.shape[0]):
            tanh = TanH()
            sigmoid = Sigmoid()
            Xt = np.concatenate((input_tensor[time], self.hidden_state))
            self.l1_input.append(Xt.reshape(1, -1).copy())
            self.hidden_state = tanh.forward(self.l1.forward(Xt.reshape(1, -1))).reshape(-1)
            self.l2_input.append(self.hidden_state.reshape(1, -1).copy())
            y = sigmoid.forward(self.l2.forward(self.hidden_state.reshape(1, -1)))
            self.y.append(y.reshape(-1))
            self.tanh.append(tanh)
            self.sigmoid.append(sigmoid)
        return np.array(self.y)
    
    def backward(self, error_tensor):
        self.gradient_l1 = []
        self.gradient_l2 = []
        output_list = []
        if self.memorize == False:
            self.hidden_state_error = np.zeros((1, self.hidden_size))
        for i in range(error_tensor.shape[0]):
            time = error_tensor.shape[0]-i-1
            # L2
            self.l2.set_input_tensor(self.l2_input[time])
            self.hidden_state_error += self.l2.backward(self.sigmoid[time].backward(error_tensor[time]))
            # L1
            self.l1.set_input_tensor(self.l1_input[time])
            output = self.l1.backward(self.tanh[time].backward(self.hidden_state_error))
            # Append
            self.gradient_l1.append(self.l1.gradient_weights)
            self.gradient_l2.append(self.l2.gradient_weights)
            output_list.append(output.reshape(-1)[0:self.input_size])
            self.hidden_state_error = output.reshape(-1)[self.input_size:].reshape(1, -1)
        
        if self.optimizer is not None:
            self.l1.update_weights(self.optimizer, sum(self.gradient_l1))
            self.l2.update_weights(self.optimizer, sum(self.gradient_l2))
        self.weights = self.l1.weights
        self.gradient_weights = sum(self.gradient_l1)
        return np.array(output_list)[::-1]
        