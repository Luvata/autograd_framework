from autogradnp.Tensor import Tensor
import numpy as np

np.random.seed(9999)
NUM_STEP = 10
LR       = 0.1


def main():
    """
    Didn't handle gradient explode (seed 2010)
    """
    data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
    target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

    w = list()
    w.append(Tensor(np.random.randn(2, 3), autograd=True))
    w.append(Tensor(np.random.randn(3, 1), autograd=True))

    for step in range(NUM_STEP):
        pred = data.mm(w[0]).mm(w[1])
        loss = ((pred - target) * (pred - target)).sum(0)
        loss.backward(Tensor(np.ones_like(loss.data)))

        for w_ in w:
            w_.data -= LR * w_.grad.data
            w_.grad.data *= 0  # zero out gradient
        print(loss.data)


main()
