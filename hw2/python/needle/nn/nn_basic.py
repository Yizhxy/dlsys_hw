"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype))) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Y = ops.matmul(X,self.weight)
        if self.bias:
          bias = ops.broadcast_to(self.bias,Y.shape)
          Y += bias
        return Y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size = logits.shape[0]
        classes = logits.shape[1]
        
        normalize_x = ops.logsumexp(logits, axes=1)
        y_one_hot = init.one_hot(classes, y)
        
        Z_y = ops.summation(logits * y_one_hot,axes=1) 
        loss = ops.summation(normalize_x - Z_y)
        
        return loss / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        weight = init.ones(dim)
        self.weight = Parameter(weight)
        bias = init.zeros(dim)
        self.bias = Parameter(bias)
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)

        if self.training:
          mean_x = ops.divide_scalar(ops.summation(x, axes=0), batch_size)
          broadcast_mean = ops.broadcast_to(ops.reshape(mean_x, (1, -1)), x.shape)
          numerator = x - broadcast_mean
          var_x = ops.power_scalar(numerator, 2)
          var_x = ops.summation(ops.divide_scalar(var_x, batch_size), axes=0)
          broadcast_var = ops.broadcast_to(ops.reshape(var_x, (1, -1)), x.shape)
          denominator = ops.power_scalar(broadcast_var+self.eps, 0.5)
          frac = numerator / denominator
          y = broadcast_weight * frac + broadcast_bias

          # update running estimates
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x
        else:
          broadcast_rm = ops.broadcast_to(ops.reshape(self.running_mean, (1, -1)), x.shape)
          broadcast_rv = ops.broadcast_to(ops.reshape(self.running_var, (1, -1)), x.shape)
          numerator = x - broadcast_rm
          denominator = ops.power_scalar(broadcast_rv+self.eps, 0.5)
          frac = numerator / denominator
          y = broadcast_weight * frac + broadcast_bias
        return y
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        weight = init.ones(dim)
        self.weight = Parameter(weight)
        bias = init.zeros(dim)
        self.bias = Parameter(bias)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        features = x.shape[1]

        mean_x = ops.divide_scalar(ops.summation(x, axes=1), features)
        broadcast_mean = ops.broadcast_to(ops.reshape(mean_x, (-1, 1)), x.shape)

        numerator = x - broadcast_mean
        
        var_x = ops.power_scalar(numerator, 2)
        var_x = ops.summation(ops.divide_scalar(var_x, features), axes=1)
        broadcast_var = ops.broadcast_to(ops.reshape(var_x, (-1, 1)), x.shape)
        
        denominator = ops.power_scalar(broadcast_var+self.eps, 0.5)
        
        frac = numerator / denominator
        
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
        y = ops.multiply(broadcast_weight, frac) + broadcast_bias
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          mask = init.randb(*x.shape, p=1-self.p)
          x = x * mask
          z = x/(1-self.p)
        else:
          z = x
        return z 
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.add(self.fn(x),x)
        ### END YOUR SOLUTION
