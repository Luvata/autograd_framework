from autogradnp.nn import Linear, Sequential, MSELoss
from autogradnp.Tensor import Tensor
from autogradnp.optim import SGD
import numpy as np

np.random.seed(9999)
NUM_STEP = 10
LR       = 0.1


def main():
    data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
    target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

    model = Sequential([Linear(2, 3), Linear(3, 1)])
    optim = SGD(model.get_parameters(), learning_rate=0.05)
    criterion = MSELoss()

    for step in range(NUM_STEP):
        pred = model.forward(data)
        loss = criterion.forward(pred, target)
        loss.backward()
        optim.step()
        print(loss.data)


if __name__ == '__main__':
    main()
