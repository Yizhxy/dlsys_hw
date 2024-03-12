"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        #注意np被换成了array_api方便后续的换源
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad * self.scalar * array_api.power(a, self.scalar-1),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs
        return out_grad / b, -out_grad*a/(b*b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(a.ndim))
        if self.axes is None:
            self.axes = axes[-2:]  # last two axes
        axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]

        return array_api.transpose(a, axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        out_grad_shape = out_grad.shape
        self.reduce_dim = []
        # 接下来为了找到reduce_dim我们需要有指针比较,指针指向的是dim,指向负数表示超出shape范围
        point_in = len(input_shape) - 1
        # 广播后的shape肯定更大，所以我们遍历out_grad_shape比较方便
        for point_out in range(len(out_grad_shape) - 1,-1,-1):
            # 如果超出范围则是需要reduce的dim
            if point_in < 0:
                self.reduce_dim.append(point_out)
                continue
            # 比较广播后对应dim的大小是否相等，不等也是需要reduce的dim
            if  self.shape[point_out]!= input_shape[point_in]:
                self.reduce_dim.append(point_out)
            # 左移in的指针
            point_in -= 1
        # 转换传入参数类型
        out_grad = summation(out_grad,tuple(self.reduce_dim))
        return reshape(out_grad,input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(self.axes, int): 
            self.axes = tuple([self.axes])

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        # 计算出input执行sum的轴再广播后（变成1）的shape
        # 把sum理解成压缩，除了压缩的那些轴，其他轴的元素都参与了sum运算，我们需要对其他轴的元素“求导”
        # 所以我们先造出一个避开压缩轴的shape，压缩轴变成1就可以广播，然后通过逐元素乘法和全1矩阵相乘 就能得到反向导数
        final_shape = list(input_shape)
        
        if self.axes:
        # 这里要多判断一次，因为如果传入int类型就无法迭代在后面的q5中就会失败
          if isinstance(self.axes, int):
              final_shape[self.axes] = 1
          else:
              for dim in self.axes:
                  final_shape[dim] = 1
        else:
            final_shape = [1 for _ in range(len(final_shape))]
        out_grad = reshape(out_grad,final_shape)
        return out_grad * array_api.ones(input_shape, dtype=array_api.float32)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        #考虑左右顺序
        dlhs = matmul(out_grad,transpose(rhs))
        drhs = matmul(transpose(lhs),out_grad)
        if dlhs.shape != lhs.shape:
            dlhs = summation(dlhs, tuple(range(len(dlhs.shape) - len(lhs.shape))))
        if drhs.shape != rhs.shape:
            drhs = summation(drhs, tuple(range(len(drhs.shape) - len(rhs.shape))))
        return dlhs , drhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inputs = node.inputs[0]
        return out_grad * exp(inputs)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        res = a.copy()
        res[res < 0] = 0
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad_data = out_grad.realize_cached_data()
        input_data = node.inputs[0].realize_cached_data()
        out_grad_data[input_data < 0] = 0
        return Tensor(out_grad_data)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
