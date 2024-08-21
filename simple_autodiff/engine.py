import numpy as np


def sigmoid(a):
    return np.where(a > 0, 1 / (1 + np.exp(-a)), np.exp(a) / (np.exp(a) + 1))


class Tensor:

    def __init__(self, data, _children=()):
        # make data at least 2D to preserve shape in matmul
        self.data = np.atleast_2d(data).astype(float)
        self.grad = np.zeros_like(self.data)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            # for dimensions that were broadcasted during forward pass
            # we need to sum up during backward pass
            sum_axes_self = tuple(
                [i for i in range(len(out.grad.shape)) if self.grad.shape[i] == 1]
            )
            sum_axes_other = tuple(
                [i for i in range(len(out.grad.shape)) if other.grad.shape[i] == 1]
            )
            self.grad += np.sum(out.grad, axis=sum_axes_self, keepdims=True)
            other.grad += np.sum(out.grad, axis=sum_axes_other, keepdims=True)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            # for dimensions that were broadcasted during forward pass
            # we need to sum up during backward pass
            sum_axes_self = tuple(
                [i for i in range(len(out.grad.shape)) if self.grad.shape[i] == 1]
            )
            sum_axes_other = tuple(
                [i for i in range(len(out.grad.shape)) if other.grad.shape[i] == 1]
            )
            self.grad += np.sum(
                other.data * out.grad, axis=sum_axes_self, keepdims=True
            )
            other.grad += np.sum(
                self.data * out.grad, axis=sum_axes_other, keepdims=True
            )

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,))

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), (self,))

        def _backward():
            self.grad += out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.where(self.data < 0, 0, self.data), (self,))

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid_cross_entropy(self, y):
        y = np.reshape(y, self.data.shape)
        out = Tensor(
            self.data * (1 - y) - np.log(sigmoid(self.data)),
            (self,),
        )

        def _backward():
            self.grad += (sigmoid(self.data) - y) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
