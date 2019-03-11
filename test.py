"""
UNIT TESTS FOR TENSOR BACKPROPAGATION

Usage:
    test.py all
    test.py add
    test.py neg

Options:
    -h --help       Show this screen.
"""


from docopt import docopt
from Tensor import Tensor
import numpy as np


def test_add():
    x = Tensor([1, 2, 3, 4], autograd=True)
    y = Tensor([2, 3, 1, 3], autograd=True)
    z = Tensor([1, 2, 3, 3], autograd=True)

    a = x + y
    b = y + z
    c = a + b

    loss = Tensor([1, 1, 1, 1], autograd=True)
    c.backward(loss)
    assert np.all(y.grad.data == np.array([2, 2, 2, 2]))

    somewhere_loss = Tensor([1, 2, 1, 2], autograd=True)
    d = somewhere_loss + y.grad
    d.backward(Tensor([1, 10, 10, 1], autograd=True))
    assert np.all(y.grad.grad.data == np.array([1, 10, 10, 1]))
    print("Test addition passed")


def test_neg():
    x = Tensor([1, 2, 3, 4], autograd=True)
    y = Tensor([2, 3, 1, 3], autograd=True)
    z = Tensor([1, 2, 3, 3], autograd=True)

    a = x + (-y)
    b = (-y) + z
    c = a + b

    c.backward(Tensor([1, 1, 1, 1]))
    assert np.all(y.grad.data == np.array([-2, -2, -2, -2]))
    print("Test negative passed")


def test_all():
    test_add()
    test_neg()


if __name__ == '__main__':
    args = docopt(__doc__)
    seed = 1234
    np.random.seed(seed * 13 // 7)

    if args['all']:
        test_all()
    elif args['add']:
        test_add()
    elif args['neg']:
        test_neg()
    else:
        raise RuntimeError('Invalid run mode')
