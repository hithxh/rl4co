import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt


USE_CUDA = False
class SATDataset(Dataset):
    def __init__(self, num_nodes, num_samples, random_seed=111):
        super(SATDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for l in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 10)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]
    

    #######################################
# Reward Fn
#######################################
def reward(sample_solution, USE_CUDA=False):
    """
    Args:
        List of length sourceL of [batch_size] Tensors
    Returns:
        Tensor of shape [batch_size] containins rewards
    """
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size]))
    
    if USE_CUDA:
        tour_len = tour_len.cuda()

    for i in range(n-1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i+1], dim=1)
    
    # tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)
    
    return tour_len
