import numpy as np

from simple_autodiff.engine import Tensor


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class Layer(Module):

    def __init__(self, nin, nout, nonlin=True):
        self.W = Tensor(np.random.randn(nin, nout))
        self.b = Tensor(np.zeros(nout))
        self.nonlin = nonlin

    def __call__(self, x):
        act = x @ self.W + self.b
        return act.relu() if self.nonlin else act

    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'linear'} layer of width {self.W.data.shape[1]}"


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
