import os
import random
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import device
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from functools import partial
import math
import torch
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
# from torch._six import container_abcs
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import time
# from sageattention import sageattn
# import torch.nn.functional as F
# F.scaled_dot_product_attention = sageattn
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
def calculate_itr(acc_matrix, M, t):
    acc_matrix = torch.tensor(acc_matrix, dtype=torch.float32)
    num_subjects, num_blocks = acc_matrix.shape
    itr_matrix = torch.zeros_like(acc_matrix)

    for i in range(num_subjects):
        for j in range(num_blocks):
            p = acc_matrix[i, j].item() / 100
            if p < 1 / M:
                itr_matrix[i, j] = 0
            elif p == 1:
                itr_matrix[i, j] = np.log2(M) * (60 / t)
            else:
                itr_matrix[i, j] = (np.log2(M) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (M - 1))) * (60 / t)

    return itr_matrix




class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 1,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        # x = self.SRU(x)
        x = self.CRU(x)
        return x
