from autogradnp.nn import Linear, Sequential, Tanh, Sigmoid
from autogradnp.Tensor import Tensor
from autogradnp.optim import SGD
import numpy as np

np.random.seed(9999)
NUM_STEP = 10
LR       = 0.1


def main():
    data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
    target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

    model = Sequential([Linear(2, 3), Tanh() ,Linear(3, 1), Tanh()])
    optim = SGD(model.get_parameters(), learning_rate=0.1)

    for step in range(NUM_STEP):
        pred = model.forward(data)
        loss = ((pred - target) * (pred - target)).sum(0)
        loss.backward(Tensor(np.ones_like(loss.data)))
        optim.step()  # perform once update for all parameters
        print(loss.data)


if __name__ == '__main__':
    main()
