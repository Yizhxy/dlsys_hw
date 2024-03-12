from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, 'rb') as f:
          image_data = np.frombuffer(f.read(), np.uint8, offset=16)
    
        # 读取标签文件
        with gzip.open(label_filename, 'rb') as f:
            label_data = np.frombuffer(f.read(), np.uint8, offset=8)
        
        # 图像数据归一化
        X = image_data.reshape(-1, 784).astype(np.float32)/255.0
        self.images = X
        self.labels = label_data
        self.transform = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index].reshape((28, 28, 1))
        if self.transform:
          for trans in self.transform:
            img = trans(img)
        return img, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION