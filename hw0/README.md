### Q1：A basic add function

问题一是实现加法，然后进行测试，该部分比较简单不做赘述

```python
def add(x, y):
    """ A trivial 'add' function you should implement to get used to the 
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### YOUR CODE HERE
    return x + y
    ### END YOUR CODE
```

测试通过运行脚本即可进行测试，分为本地测试和注册账号后通过mugrade测试，由于测试数据都一样，所以所有hw中均使用本地测试

### Q2:loading MNIST data

该部分是实现加载mnist这个数据集，代码在src/simple_ml.py下的paras_minist_data()函数

通过函数中的注释提醒，通过numpy的读取数据相关的接口就可以实现

```python
def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # 读取图像文件
    with gzip.open(image_filename, 'rb') as f:
        image_data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    # 读取标签文件
    with gzip.open(label_filename, 'rb') as f:
        label_data = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # 图像数据归一化
    X = image_data.reshape(-1, 784).astype(np.float32)/255.0
    return X, label_data
    ### END YOUR CODE
```

### Q3:Softmax Loss

实现在src/simple_ml.py下的softmax_loss函数

softmax loss:

$$\ell_{\mathrm{softmax}}(z,y)=\log\sum_{i=1}^k\exp z_i-z_y.$$

观察公式，通过numpy可以很好的实现只需要调用np.log np.sum np.exp三个函数即可

需要注意的是，输入中的数据有一个batchsize维度

```python
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    loss = np.mean(np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(len(y)), y])
    return loss
    ### END YOUR CODE
```

### Q4:Stochastic gradient descent(SGD) for softmax regression

#### step1:实现softmax_regression_epoch()函数

公式部分比较简单，代码中需要划分batch以及注意数据集的维度，在调用np函数时候注意参数

```python
def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    m = X.shape[0]
    num_batches = (m + (batch - 1)) // batch
    for i in range(num_batches):
        
        num = min(m, (i + 1) * batch)
        x_batch = X[i*batch:num, :]
        y_batch = y[i*batch:num]

        h = x_batch @ theta

        Z = np.exp(h) / np.sum(np.exp(h), axis=1, keepdims=True,dtype=np.float64)
        I_y = np.zeros_like(Z)
        I_y[range(len(y_batch)), y_batch] = 1

        gradient = x_batch.T @ (Z - I_y)
        theta -= lr * gradient / batch 
    ### END YOUR CODE
```

#### step2:train MNIST with softmax_regression

该部分是在mnist数据集上进行训练，所需要的函数在之前都已经写过了，这里只需要在train_softmax函数中进行正确的调用就行

```python
def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))
```

### Q5:SGD for a two-layer neural network

多了一层且有激活函数，公式比较复杂，但原理同上，一步一步慢慢写即可

#### step1:实现nn_epoch()函数

需要注意的是relu在反向传播中对梯度的影响

```python
def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    m = X.shape[0]
    d = W1.shape[1]
    k = W2.shape[1]
    num_batches = (m + (batch - 1)) // batch
    
    for i in range(num_batches):
        
        num = min(m, (i + 1) * batch)
        x_batch = X[i*batch:num, :]
        y_batch = y[i*batch:num]

        Z_1 = np.maximum(0,x_batch @ W1)
        I_y = np.zeros((batch, k))
        I_y[range(len(y_batch)), y_batch] = 1
        G_2 = np.exp(Z_1@W2)/np.sum(np.exp(Z_1@W2), axis=1, keepdims=True,dtype=np.float64) - I_y
        G_1 = ((Z_1 > 0).astype(int)) * (G_2 @ W2.T)
        gradient_w1 = x_batch.T @ G_1
        gradient_w2 = Z_1.T @ G_2

        W1 -= lr * gradient_w1 / batch
        W2 -= lr * gradient_w2 / batch
    ### END YOUR CODE
```

#### step2:Training a full neural network(实现train_nn)

与之前一样

```python
def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))
```

### Q6:softmax regression in C++

用c++重写一下之前的，不做赘述
