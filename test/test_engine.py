import numpy as np
import torch

from simple_autodiff.engine import Tensor


def test_sanity_check():

    x = Tensor([-4.0, -2.0])
    z = 2 * x + 2 + x @ np.diag([0.8, 1.8])
    q = z.relu() + z * x
    h = (z * z).relu()
    y = (h + q + q * x).sum()
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0, -2.0])
    x.requires_grad = True
    z = 2 * x + 2 + x @ torch.diag(torch.tensor([0.8, 1.8]))
    q = z.relu() + z * x
    h = (z * z).relu()
    y = (h + q + q * x).sum()
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert np.allclose(ymg.data, ypt.data.numpy())
    # backward pass went well
    assert np.allclose(xmg.grad, xpt.grad.numpy())


def test_more_ops():

    a = Tensor([-4.0, -2.0])
    b = Tensor([2.0, 2.5])
    c = a + b
    d = a * b + b**1.2
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**1.1
    g = f / 2.0
    g += 4.0 / f
    g = g / 50
    g = g.sigmoid_cross_entropy(np.ones_like(g.data))
    g = g.sum()
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0, -2.0])
    b = torch.Tensor([2.0, 2.5])
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**1.2
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**1.1
    g = f / 2.0
    g = g + 4.0 / f
    g = g / 50
    g = torch.nn.functional.binary_cross_entropy_with_logits(
        g, torch.ones_like(g), reduction="none"
    )
    g = g.sum()
    g.backward()
    apt, bpt, gpt = a, b, g

    # forward pass went well
    assert np.allclose(gmg.data, gpt.data.numpy())
    # backward pass went well
    assert np.allclose(amg.grad, apt.grad.numpy())
    assert np.allclose(bmg.grad, bpt.grad.numpy())
