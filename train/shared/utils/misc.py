import json
import os
import random

import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def hashable(key):
    try:
        hash(key)
        return True
    except TypeError:
        return False


def sigmoid(n):
    return 1 / (1 + np.exp(-n))


GLOBAL_OBJECTS = {}
