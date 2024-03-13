从hw3开始，开始为大多数深度学习系统的处理构建一个简单的后备库：n 维数组（又名 NDArray）。 到目前为止，我们基本上一直在使用 numpy 来实现关于array的操作，但本作业将引导您开发相当于您自己的（尽管更加有限）的 numpy 变体，它将支持 CPU 和 GPU 后端(将在python,cpp,cuda上分别完成)。

### Part1:Python array operations

完成ndarray.py中的四个函数(对ndarray的操作)，重要的是要强调，这些函数中的任何一个都不应重新分配内存，而应该返回与 self 共享相同内存的 NDArrays，并且只需巧妙地操纵形状/步幅等以获取所需的转换。

##### `reshape`

```python
def reshape(self, new_shape):
        ### BEGIN YOUR SOLUTION
        if prod(new_shape) != prod(self._shape):
            raise ValueError("Product of current shape is not equal to \
                                  the product of the new shape!")
            if not self.is_compact():
                raise ValueError("The matrix is not compact!")

                return NDArray.make(new_shape, NDArray.compact_strides(new_shape), self._device, self._handle)
        ### END YOUR SOLUTION
```

##### `permute`

```python
def permute(self, new_axes):
        ### BEGIN YOUR SOLUTION
        new_shape = tuple(np.array(self._shape)[list(new_axes)])
        new_strides = tuple(np.array(self._strides)[list(new_axes)])
        return NDArray.make(new_shape, new_strides, self._device, self._handle)
        ### END YOUR SOLUTION
```

##### `broadcast`

```python
def broadcast_to(self, new_shape):
        ### BEGIN YOUR SOLUTION
        assert(len(self._shape) == len(new_shape))
        for x, y in zip(self._shape, new_shape):
        assert(x == y or x == 1)

        new_strides = list(self._strides)
        for i in range(len(self._shape)):
        if self._shape[i] != new_shape[i]:
        new_strides[i] = 0
        return NDArray.make(new_shape, tuple(new_strides), self._device, self._handle)
        ### END YOUR SOLUTION
```

### Part2:CPU backend -compact and setitem

##### `compact`

```c++
void compactHelper(AlignedArray* out, const AlignedArray& a, std::vector<uint32_t>& indexes, 
                   std::vector<uint32_t>& shape, std::vector<uint32_t> strides, 
                   int& cnt, int& loop, size_t offset, int dim) {
  uint32_t size = shape[loop];
  if (loop == dim - 1) {
    int idx = 0;
    for (int i = 0; i < loop; ++i) {
      idx += indexes[i] * strides[i];
    }
    for (int i = 0; i < size; ++i) {
      *(out->ptr + cnt) = *(a.ptr + offset + idx + strides[loop] * i);
      cnt++;
    }
    return;
  }

  for (int i = 0; i < size; ++i) {
    indexes[loop] = i;
    loop = loop + 1;
    compactHelper(out, a, indexes, shape, strides, cnt, loop, offset, dim);
    loop = loop - 1;
  }
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  int cnt = 0;
  int loop = 0;
  int dim = shape.size();
  std::vector<uint32_t> indexes(dim, 0);
  compactHelper(out, a, indexes, shape, strides, cnt, loop, offset, dim);
  /// END SOLUTION
}
```

##### `EwiseSetitem`

```c++
void ewiseSetitemHelper(AlignedArray* out, const AlignedArray& a, std::vector<uint32_t>& indexes, 
                   std::vector<uint32_t>& shape, std::vector<uint32_t> strides, 
                   int& cnt, int& loop, size_t offset, int dim) {
  uint32_t size = shape[loop];
  if (loop == dim - 1) {
    int idx = 0;
    for (int i = 0; i < loop; ++i) {
      idx += indexes[i] * strides[i];
    }
    for (int i = 0; i < size; ++i) {
      *(out->ptr + offset + idx + strides[loop] * i) = *(a.ptr + cnt);
      cnt++;
    }
    return;
  }

  for (int i = 0; i < size; ++i) {
    indexes[loop] = i;
    loop = loop + 1;
    ewiseSetitemHelper(out, a, indexes, shape, strides, cnt, loop, offset, dim);
    loop = loop - 1;
  }
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  int cnt = 0;
  int loop = 0;
  int dim = shape.size();
  std::vector<uint32_t> indexes(dim, 0);
  ewiseSetitemHelper(out, a, indexes, shape, strides, cnt, loop, offset, dim);
  /// END SOLUTION
}
```

##### `ScalarSetitem`

```c++
void scalarSetitemHelper(AlignedArray* out, scalar_t val, std::vector<uint32_t>& indexes, 
                   std::vector<uint32_t>& shape, std::vector<uint32_t> strides, 
                   int& loop, size_t offset, int dim) {
  uint32_t size = shape[loop];
  if (loop == dim - 1) {
    int idx = 0;
    for (int i = 0; i < loop; ++i) {
      idx += indexes[i] * strides[i];
    }
    for (int i = 0; i < size; ++i) {
      *(out->ptr + offset + idx + strides[loop] * i) = val;
    }
    return;
  }

  for (int i = 0; i < size; ++i) {
    indexes[loop] = i;
    loop = loop + 1;
    scalarSetitemHelper(out, val, indexes, shape, strides, loop, offset, dim);
    loop = loop - 1;
  }
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  int cnt = 0;
  int loop = 0;
  int dim = shape.size();
  std::vector<uint32_t> indexes(dim, 0);
  scalarSetitemHelper(out, val, indexes, shape, strides, loop, offset, dim);
  /// END SOLUTION
}
```

### Part3:CPU Backend-Elementwise and scalar operations

实现的和hw1中的一样，不赘述

#### Part4:CPU Backend-Reductions

##### `ReduceMax`

```c++
void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  size_t cnt = 0;
  for (size_t i = 0; i < a.size; i += reduce_size) {
    out->ptr[cnt] = a.ptr[i];
    for (size_t j = 0; j < reduce_size; ++j) {
      out->ptr[cnt] = out->ptr[cnt] > a.ptr[i + j] ? out->ptr[cnt] : a.ptr[i + j];
    }
    cnt++;
  }
  
  /// END YOUR SOLUTION
}
```

##### `ReduceSum`

```c++
void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  size_t cnt = 0;
  for (size_t i = 0; i < a.size; i += reduce_size) {
    out->ptr[cnt] = 0;
    for (size_t j = 0; j < reduce_size; ++j) {
      out->ptr[cnt] += a.ptr[i + j];
    }
    cnt++;
  }
  /// END YOUR SOLUTION
}
```

### Part5:CPU Backend-Matrix multiplication

与hw1中一样，换为c++即可

## PART6-PART8 cuda相关

由于没有学习过cuda编程，等学习后再补上