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


def test_sub():
    x = Tensor([1, 2, 3, 4], autograd=True)
    y = Tensor([2, 3, 1, 3], autograd=True)
    z = Tensor([1, 2, 3, 3], autograd=True)

    a = x + y
    b = z - y
    c = a + b

    c.backward(Tensor([1, 1, 1, 1]))
    assert np.all(x.grad.data == np.array([1, 1, 1, 1]))
    assert np.all(y.grad.data == np.array([0, 0, 0, 0]))
    assert np.all(z.grad.data == np.array([1, 1, 1, 1]))

    print("Test subtraction passed")


def test_mul():
    x = Tensor([1, 2, 3, 4], autograd=True)
    y = Tensor([2, 3, 1, 3], autograd=True)
    z = Tensor([1, 2, 3, 3], autograd=True)

    a = x * y
    b = y * z
    c = a * b

    c.backward(Tensor([1, 1, 1, 1]))
    assert np.all(x.grad.data == (y.data ** 2 * z.data))
    assert np.all(y.grad.data == (y.data * 2 * z.data * x.data))
    assert np.all(z.grad.data == (y.data ** 2 * x.data))
    print("Test element-wise multiplication passed")


def test_transpose():
    x = Tensor([[1, 2, 3, 4]], autograd=True)
    x_trans = x.transpose()
    x_trans.backward(Tensor([[9], [2], [3], [1]]))
    assert np.all(x.grad.data == np.array([9, 2, 3, 1]))
    print("Test transpose passed")


def test_mm():
    a = Tensor([[1, 2, 3], [2, 0, 1]], autograd=True)  # 2 x 3
    b = Tensor([[1], [0], [-1]], autograd=True)        # 3 x 1

    out  = a.mm(b)
    dout = Tensor([[2], [0]])
    out.backward(dout)

    assert np.all(a.grad.data == dout.data.dot(b.data.T))
    assert np.all(b.grad.data == a.data.T.dot(dout.data))
    print("Test mat mul passed")


def test_expand():
    a = Tensor([[1, 2], [3, 4]], autograd=True)
    copies = 2
    dim    = 0

    b = a.expand(dim, copies)
    expected_b = [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
    assert np.all(b.data == np.array(expected_b))

    dloss = Tensor(np.ones_like(b.data))
    b.backward(dloss)
    expected_a_grad = [[2, 2], [2, 2]]
    assert np.all(a.grad.data == np.array(expected_a_grad))
    print("Test expand passed")


def test_sum():
    a = Tensor([[1, 2], [3, 4]], autograd=True)
    dim = 1

    b = a.sum(dim=dim)
    expected_b = [3, 7]
    assert np.all(b.data == np.array(expected_b))

    dloss = Tensor(np.ones_like(b.data))
    b.backward(dloss)
    expected_a_grad = [[1, 1], [1, 1]]
    assert np.all(a.grad.data == np.array(expected_a_grad))

    print("Test sum by axis passed")


def test_all():
    test_add()
    test_neg()
    test_sub()
    test_mul()
    test_transpose()
    test_mm()
    test_expand()
    test_sum()


if __name__ == '__main__':
    test_all()
