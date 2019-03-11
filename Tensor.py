import numpy as np

class Tensor (object):
    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        """
        When a tensor is created, increment the contribution of this tensor on its parents by 1
        """
        self.data        = np.array(data)
        self.creators    = creators
        self.creation_op = creation_op
        self.grad        = None
        self.autograd    = autograd
        # Dictionary, count number of gradients received from each child during BP
        self.children    = {}
        if id is None:
            id = np.random.randint(0, 100000)
        self.id          = id
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
            
            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is not None):
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                elif self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())
    
    def __add__(self, other):
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
        if self.autograd:
            return Tensor(
                data=self.data * -1,
                autograd=True,
                creators=[self],
                creation_op="neg"
            )
        else:
            return Tensor(data=self.data * (-1))
    
    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

