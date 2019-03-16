from .Tensor import Tensor


class Optimizer(object):
    """
    Base class for optimizer strategy
    """
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, learning_rate):
        """
        Stochastic gradient descent optimizer
        :param parameters: List of Tensor which are "weights" of network
        :param learning_rate: float
        """
        self.parameters    = parameters
        self.learning_rate = learning_rate

    def zero(self):
        for param in self.parameters:
            param.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= self.learning_rate * p.grad.data
            if zero:
                p.grad.data *= 0

