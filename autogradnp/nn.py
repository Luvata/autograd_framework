from .Tensor import Tensor
import numpy as np


class Layer(object):
    """
    Base class for layer
    """
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        weight = np.random.randn(n_inputs, n_outputs) * np.sqrt(2. / n_inputs)  # He initialization
        self.weight = Tensor(weight, autograd=True)
        self.bias   = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))  # broadcasting add bias to batch matmul


class Sequential(Layer):
    """
    A sequential layer that forward propagates a list of layers
    Each layer feeds its output to the next layer
    """
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params


class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, target):
        return ((pred - target) * (pred - target)).sum(0)


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()


