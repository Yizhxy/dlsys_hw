在hw1中开始实现needle(necessary elements of deep learning)库,也就是我没的小型pytorch

hw1中主要完成automatic differentiation框架，然后在mnist上训练一个二层的分类模型

needle的介绍在本课程的lecture 5，并有相应的[notebook](https://github.com/dlsys10714/notebooks/blob/main/5_automatic_differentiation_implementation.ipynb)

### Q1:implementing forward computation

### Q2:implementing backward computation

在q1，q2中需要实现许多的算子的正向反向传播，但是都很简单，通过numpy库的函数可以很好的实现，注意  默认import numpy as array_api而不是np

需要完成的算子有

<center> <img src="C:/Users/%E5%A4%8F%E8%8C%83/AppData/Roaming/Typora/typora-user-images/image-20240311202622720.png" alt="image-20240311202622720" style="zoom:50%;"/> 

其中有的算子需要一些矩阵论的知识，参考[矩阵求导术](https://zhuanlan.zhihu.com/p/24709748)

大部分的算子在数理基础~~良好~~(变态)的情况下都能比较快的解决，这里主要介绍broadcast和summation的反向传播，[参考博客](https://blog.fyz666.xyz/blog/5783/)

```python
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
```

```python
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

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
```

### Q3:Topological sort

这里我们需要填写两个函数 `find_topo_sort` 方法和 `topo_sort_dfs` 帮助方法，以执行此拓扑排序

大概就是给你最后一个输出节点，你要遍历这个图，通过dfs，把他正确的计算顺序（这里指的节点排序）找到

就是一个反向深度搜索找叶子节点，然后加入要返回的顺序（拓扑序列）就行

```python
def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    visted = set()
    topo_order = []
    for node in node_list:
        if node not in visted:
            topo_sort_dfs(node, visted, topo_order)
    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    for input_node in node.inputs:
        if input_node not in visited:
            topo_sort_dfs(input_node, visited, topo_order)

    if node not in visited:
        visited.add(node)
        topo_order.append(node)
    ### END YOUR SOLUTION
```

### Q4:implementing reverse mode differentiation

q4要求结合q3得到的拓扑序列和q1,q2解决的各个算子问题，通过backward方法，就能计算出loss关于中间节点的偏导。

实现的主要步骤

函数的主要步骤：

- 创建一个字典 `node_to_output_grads_list`，用于存储每个节点对应的输出梯度列表。
- 将 `output_tensor` 对应的输出梯度列表设置为 `[out_grad]`。
- 使用 `find_topo_sort` 函数找到以 `output_tensor` 为根的图的逆拓扑排序（reverse topological order）。
- 遍历逆拓扑排序中的每个节点 `node`：
  - 将 `node.grad` 设置为其对应输出梯度列表的和。
  - 如果节点 `node` 没有操作（`op`为`None`），则跳过。
  - 否则，使用节点 `node` 上的操作（`node.op`）计算相对于输入节点的梯度 `input_grads`。
  - 如果 `input_grads` 是单个张量，则将其转换为列表。
  - 遍历 `node` 的每个输入节点 `input_node`，将对应的输入梯度添加到 `node_to_output_grads_list` 中。

```python
def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for node in reverse_topo_order:
        node.grad = sum_node_list(node_to_output_grads_list[node])
        if node.op is None:
            continue

        input_grads = node.op.gradient(node.grad, node)
        if isinstance(input_grads, Tensor):
            input_grads = [input_grads]
        for i, input_node in enumerate(node.inputs):
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []
            node_to_output_grads_list[input_node].append(input_grads[i])
    ### END YOUR SOLUTION
```

### Q5:softmax loss

比较简单，需要注意的是这里的编码方式为one-hot格式

```python
def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    Z_y = ndl.summation(Z * y_one_hot,axes=1) 
    normalize_z = ndl.log(ndl.summation(ndl.exp(Z),axes=1))
    loss = ndl.summation(normalize_z - Z_y) 
    return loss / batch_size
    ### END YOUR SOLUTION
```

### Q6:SGD for a two-layer neural network

和hw0中的应用一样，不同的是我们现在不需要手动求导了(~~欢呼~~)，通过刚才的AD算法自动实现

(记得把hw0中的加载mnist数据集函数`paras_mnist`函数粘过来)

```python
def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    idx = 0
    num_classes = W2.shape[1]
    while idx < X.shape[0]:
        X_batch = ndl.Tensor(X[idx:idx+batch])
        Z1 = X_batch.matmul(W1)
        network_output = ndl.relu(Z1).matmul(W2)

        y_batch = y[idx:idx+batch]
        y_one_hot = np.zeros((batch, num_classes))
        y_one_hot[np.arange(batch), y_batch] = 1
        y_one_hot = ndl.Tensor(y_one_hot)

        loss = softmax_loss(network_output, y_one_hot)
        loss.backward()

        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
        idx += batch
    return W1, W2
    ### END YOUR SOLUTION
```



