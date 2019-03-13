import numpy as np


class Tensor(object):
    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        """
        When a tensor is created, increment the contribution of this tensor on its parents by 1
        """
        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        # Dictionary, count number of gradients received from each child during BP
        self.children = {}
        if id is None:
            id = np.random.randint(0, 100000)
        self.id = id
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        """
        Checks weather a tensor has received correct number of gradients from each child
        Normally, whenever .backward() is called on a variable, it immediately calls .backward() on its parents

        """
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        """
        Back propagation
        - grad: Tensor shape = data.shape, contain the gradient that flow through this Tensor
        - grad_origin: Tensor
        """
        if self.autograd:
            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("Cannot backprop more than one")
                else:
                    self.children[grad_origin.id] -= 1  # when call backward, dict children decrease by 1

            if self.grad is None:
                self.grad = grad  # Accumulates gradients from childrens
            else:
                self.grad += grad

            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                elif self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                elif self.creation_op == "sub":
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(new, self)

                elif self.creation_op == "mul":
                    new = self.grad.data * self.creators[1].data
                    self.creators[0].backward(new, self)
                    new = self.grad.data * self.creators[0].data
                    self.creators[1].backward(new, self)

                elif self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())

                elif self.creation_op == "mm":
                    # nxd dot dxc -> nxc
                    new = self.grad.mm(self.creators[1].transpose())
                    self.creators[0].backward(new)
                    new = self.creators[0].transpose().mm(self.grad)
                    self.creators[1].backward(new)

                elif self.creation_op.startswith("sum"):
                    dim = int(self.creation_op.split("_")[1])
                    copies = self.creators[0].data.shape[dim]
                    new = self.grad.expand(dim, copies)
                    self.creators[0].backward(new)

                elif self.creation_op.startswith("expand"):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

    def __add__(self, other):
        """
        Add element-wise two Tensors
        :param other: Tensor
        :return: Tensor
        """
        if self.autograd and other.autograd:
            return Tensor(
                self.data + other.data,
                autograd=True,
                creators=[self, other],
                creation_op="add"
            )
        else:
            return Tensor(self.data + other.data)  # no gradient & no creators

    def __neg__(self):
        """
        :return: Negative of a Tensor
        """
        if self.autograd:
            return Tensor(
                data=self.data * -1,
                autograd=True,
                creators=[self],
                creation_op="neg"
            )
        else:
            return Tensor(data=self.data * (-1))

    def __sub__(self, other):
        """
        Element-wise subtraction of two Tensor
        :param other: Tensor
        :return: Tensor difference of A - B
        """
        if self.autograd and other.autograd:
            return Tensor(
                self.data + other.__neg__().data,
                autograd=True,
                creators=[self, other],
                creation_op="sub"
            )
        else:
            return Tensor(self.data - other.data)

    def __mul__(self, other):
        """
        Element-wise multiplication of two Tensor with same shape
        :param other: Tensor same shape with current Tensor
        :return : Tensor result
        """
        if self.autograd and other.autograd:
            return Tensor(
                self.data * other.data,
                autograd=True,
                creators=[self, other],
                creation_op="mul"
            )
        else:
            return Tensor(self.data * other.data)

    def transpose(self):
        if self.autograd:
            return Tensor(
                self.data.T,
                autograd=True,
                creators=[self],
                creation_op="transpose"
            )
        else:
            return Tensor(self.data.transpose())

    def mm(self, other):
        """
        Matrix multiplication (a.k.a dot product)
        :param other: Tensor which first dimension equal to last dimension of current Tensor
        :return: Tensor shape N x C
        """
        if self.autograd:
            return Tensor(
                self.data.dot(other.data),
                autograd=True,
                creators=[self, other],
                creation_op="mm"
            )
        else:
            return Tensor(self.data.dot(other.data))

    def expand(self, dim, copies):
        """
        Duplicate Tensor over given dimension
        :param dim: int
        :param copies: int
        :return: Tensor
        """
        shapes = self.data.shape
        trans_cmd = list(range(len(shapes)))
        trans_cmd.insert(dim, len(shapes))  # inject the dimension, eg (0, 1) -> (0, 2, 1)
        new_shape = list(shapes) + [copies]  # copy many times : (3, 5) copy 3 times -> (3, 5, 3)
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)  # re-order values

        if self.autograd:
            return Tensor(
                new_data,
                autograd=True,
                creators=[self],
                creation_op="expand_" + str(dim)
            )
        else:
            return Tensor(new_data)

    def sum(self, dim):
        """
        Sum of Tensor over given dimension
        :param dim: int
        :return: Tensor
        """
        if self.autograd:
            return Tensor(
                self.data.sum(axis=dim),
                autograd=True,
                creators=[self],
                creation_op="sum_" + str(dim)
            )
        else:
            return Tensor(self.data.sum(axis=dim))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
