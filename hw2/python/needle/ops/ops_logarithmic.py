from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z = array_api.max(Z, axis=self.axes, keepdims=False)
        logsumexp = array_api.log(array_api.sum(array_api.exp(Z - max_Z),axis=self.axes)) + max_z
        return logsumexp 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        max_Z = array_api.max(Z.cached_data, axis=self.axes, keepdims=True)
        exp_val = exp(Z - Tensor(max_Z))
        sum_val = summation(exp_val, axes=self.axes)

        log_grad = out_grad / sum_val
        
        #下面就是sum那部分的导数，和之前的summation算子是一样的
        input_shape = node.inputs[0].shape
        final_shape = list(input_shape)
        if self.axes:
          if isinstance(self.axes, int):
              final_shape[self.axes] = 1
          else:
              for dim in self.axes:
                  final_shape[dim] = 1
        else:
            final_shape = [1 for _ in range(len(final_shape))]
        sum_grad = reshape(log_grad, tuple(final_shape))
        sum_grad_b = broadcast_to(sum_grad, Z.shape)
        return exp_val * sum_grad_b


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

