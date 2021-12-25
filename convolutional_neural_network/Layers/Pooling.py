import numpy as np


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
    
    def max_pooling(self, inp):
        x_shape = inp.shape[0]-self.pooling_shape[0]+1
        y_shape = inp.shape[1]-self.pooling_shape[1]+1
        output = np.zeros((np.arange(0, x_shape, self.stride_shape[0]).shape[0],
                          np.arange(0, y_shape, self.stride_shape[1]).shape[0]))
        local_max = np.zeros((np.arange(0, x_shape, self.stride_shape[0]).shape[0],
                          np.arange(0, y_shape, self.stride_shape[1]).shape[0], 2))
        for i in range(0, x_shape, self.stride_shape[0]):
            for j in range(0, y_shape, self.stride_shape[1]):
                # init
                x1 = i
                x2 = int(x1 + self.pooling_shape[0])
                y1 = j
                y2 = int(y1 + self.pooling_shape[1])
                # calculate
                output[int(i/self.stride_shape[0]), int(j/self.stride_shape[1])] = np.max(inp[x1:x2, y1:y2])
                x, y = np.unravel_index(inp[x1:x2, y1:y2].argmax(), inp[x1:x2, y1:y2].shape)
                x += x1
                y += y1
                local_max[int(i/self.stride_shape[0]), int(j/self.stride_shape[1])] = (int(x), int(y))
        return output, local_max
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        test, test_2 = self.max_pooling(input_tensor[0][0])
        output = np.zeros((*input_tensor.shape[0:2], *test.shape))
        self.local_max = np.zeros((*input_tensor.shape[0:2], *test_2.shape))
        for batch in range(input_tensor.shape[0]):
            for channel in range(input_tensor.shape[1]):
                output[batch][channel], self.local_max[batch][channel] = self.max_pooling(input_tensor[batch][channel])
        return output
    
    def backward(self, error_tensor):
        output = np.zeros(self.input_tensor.shape)
        for batch in range(error_tensor.shape[0]):
            for channel in range(error_tensor.shape[1]):
                array = self.local_max[batch][channel]
                for i in range(array.shape[0]):
                    for j in range(array.shape[1]):
                        output[batch][channel][int(array[i][j][0])][int(array[i][j][1])] += error_tensor[batch][channel][i][j]
        return output
