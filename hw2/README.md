在本次作业中，开始在needle框架下实现神经网络库(neural network library),包括不同的网络结构模块（module），数据集处理，损失函数，优化器，最后利用这些部件，实现一个简单MLPresnet.

完成本次作业后就得到了一个自己实现的类似于cpu版本的mini-pytorch

### Q0

根据提示把hw1写过的相关文件粘贴过来

### Q1：implementing weight initialization

实现一些参数的初始化函数，主要是两种方法 **Xavier Initialization**和**Kaiming Initialization**

其中这两种方法中的参数又可以服从均匀分布或者正太分布，所以一共四个函数。

根据给出的公式，比较容易写出代码

```python
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    u = rand(fan_in, fan_out, low=-a, high=a)
    return u
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    N = randn(fan_in, fan_out, mean=0, std=std)
    return N
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3 / fan_in)
    U = rand(fan_in, fan_out, low=-bound, high=bound)
    return U
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain/math.sqrt(fan_in)
    N = randn(fan_in, fan_out, mean=0, std=std)
    return N
    ### END YOUR SOLUTION
```

### Q2:

在问题二中，开始现在神经网阔的核心模块(modules)部分，在实现这些模块的时候，需要使用q1中实现的参数初始化方法

#### Q3:

实现SGD和adam两个优化器

### Q4:

实现对数据集的操作