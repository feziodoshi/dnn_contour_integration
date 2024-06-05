# Pytorch related
import torch
import torch.nn as nn
import torchvision
import torchvision.models as pretrained_models
import torch.optim as optim



# Numpy, Matplotlib, Pandas, Sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

 


# python utilities
from itertools import combinations
import pickle
from tqdm import tqdm_notebook as tqdm
import copy
import os
import collections
import math
import time

from PIL import Image, ImageStat
from matplotlib.pyplot import imshow


def imshow_grid(img,figsize=None): 
    '''
    Visualize the images in a 2d grid with 8 images in each row by default
    
    Parameters:
    -----------
    
        img: 4d Tensor of images (B, C, H, W)
    
    
    '''
#     img = img / 2 + 0.5     # unnormalize
    if(figsize):
        plt.figure(figsize=figsize)
    
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
