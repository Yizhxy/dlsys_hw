## CMU 10-414/717

[课程链接](https://dlsyscourse.org/)

### 这门课主要是讲解了 PyTorch 的整体设计，以及在作业中引导我们实现一个简版的 PyTorch（项目名叫 needle）。

内容包括：

1. 介绍深度学习基础（包括反向传播算法，以及其在两层全连接网络的推导过程）；

2. 如何通过计算图的方式实现 Automatic Differentiation；

3. PyTorch 是如何做模块化的：

4. 1. Tensor：用于操作多维数组，同时在前向计算和反向传播时自动构建计算图；
      底层通过 device 指定数据放置的位置（CPU / GPU），不同的设备对应不同的
      数组操作实现（C++ / CUDA）
   2. nn.Module：封装神经网络的一个功能模块；小的 module 例如 SoftmaxLoss、BatchNorm1d，
      大一点的例如 LSTMCell、LSTM 等；
   3. Optimizer：用于反向传播时，更新参数；例如 SGD、Adam 等；
   4. Dataset 和 DataLoader：用于加载和管理数据；

5. 前两个作业中，**底层的数组操作**都是用 numpy 实现的，后面就会讲解**怎么用 C++ 和 Cuda 分别实现**；

6. 介绍了 CNN、RNN、LSTM、Transformer、GAN 等经典网络的原理和实现方式；

7. 简单介绍了模型微调、部署、编译等方面的知识。