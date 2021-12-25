import numpy as np


class Constant:
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value
        
    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = np.ones(weights_shape) * self.constant_value
        return initialized_tensor


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = np.random.random_sample(weights_shape)
        return initialized_tensor


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in+fan_out))
        initialized_tensor = np.random.normal(0, sigma, size=weights_shape)
        return initialized_tensor


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        initialized_tensor = np.random.normal(0, sigma, size=weights_shape)
        return initialized_tensor
