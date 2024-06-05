

import sys
import argparse
sys.path.insert(1,'../')
sys.path.insert(1,'../lib/')


# Pytorch related
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils import data as dt
from torchinfo import summary
import torchvision.models as pretrained_models
from alexnet_pytorch import AlexNet
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils import model_zoo
from torch.autograd import Variable



# Numpy, Matplotlib, Pandas, Sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import patches
# import seaborn as sns
# from sklearn import manifold
# from sklearn.decomposition import PCA
# from scipy.spatial import distance
# from scipy.stats.stats import pearsonr 
# from PIL import Image, ImageStat
# from matplotlib.pyplot import imshow
# %matplotlib inline


# python utilities
from itertools import combinations
import pickle
from tqdm import tqdm
import copy
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import networkx as nx
import os
from IPython.display import Image
from IPython.core.debugger import set_trace
import collections
from functools import partial
import math
import time
import glob



# Extra imports
# from pytorch_pretrained_vit import ViT
# from models import barlow_twins
from lib.feature_extractor import FeatureExtractor
from lib.custom_dataset import Contour_Dataset
from lib.build_fe_ft_models import *
from lib.misc_functions import imshow_grid
from lib.field_stim_functions import *













################################################################################################################################################################
parser = argparse.ArgumentParser(description='Evaluate model features on ImageNet')


parser.add_argument('--get-B', nargs='+', type=int, default=[0,15,30,45,60,75], help='path element - global curvature')

parser.add_argument('--get-D', nargs='+', type=int, default=[32], help='path element - distance between elements')

parser.add_argument('--get-A', nargs='+', type=int, default=[0], help='path element - local curvature')


parser.add_argument('--fine-tune', default='true', 
                    type=str, metavar='N', help='Fine Tuning or Frozen weights in base model')

parser.add_argument('--base-model-name', default='alexnet-pytorch_regim_categ', 
                    type=str, metavar='N', help='base model to readout from')


parser.add_argument('--layer-name', default='avgpool', 
                    type=str, metavar='N', help='layer of base model to readout from')

parser.add_argument('--batch-size', default=8, 
                    type=int, metavar='N', help='batch size')

parser.add_argument('--num-workers', default=32, 
                    type=int, metavar='N', help='number of workers')

parser.add_argument('--use-device', default='2', 
                    type=str, metavar='N', help='device - gpu or cpu')


parser.add_argument('--total-num-epochs', default=100, 
                    type=int, metavar='N', help='training epochs')



parser.add_argument('--optimizer-name', default='sgd', 
                    type=str, metavar='N', help='optimizer name')


parser.add_argument('--optimizer-lr', default=0.001, 
                    type=float, metavar='N', help='optimizer learning rate')


parser.add_argument('--optimizer-momentum', default=0.9, 
                    type=float, metavar='N', help='optimizer momentum')


parser.add_argument('--optimizer-wd', default=0.0, 
                    type=float, metavar='N', help='optimizer weight decay')


parser.add_argument("--is-scheduler", action="store_true",
                    help="one cycle scheduler")


args = parser.parse_args()
################################################################################################################################################################







################################################################################################################################################################
visual_diet_config={'root_directory':os.path.expanduser('/home/jovyan/work/Datasets/contour_integration/model-training/config_0/'),'get_B':args.get_B,'get_D':args.get_D,'get_A':args.get_A,'get_numElements':[12]}

training_config={'fine_tune':True,'base_model_name':args.base_model_name,'layer_name':args.layer_name,
                  'batch_size':args.batch_size,'num_workers':args.num_workers, 'use_device':args.use_device,'total_num_epochs':args.total_num_epochs,
                 'optimizer_name':args.optimizer_name,'optimizer_lr':args.optimizer_lr,'optimizer_momentum':args.optimizer_momentum,'optimizer_wd':args.optimizer_wd, 'is_scheduler':True}
################################################################################################################################################################  
if(args.fine_tune=='true'):
    training_config['fine_tune']=True
else:
    training_config['fine_tune']=False


    
    
################################################################################################################################################################  
## Getting all the variables
root_directory=visual_diet_config['root_directory']
get_B=visual_diet_config['get_B']
get_D=visual_diet_config['get_D']
get_A=visual_diet_config['get_A']
get_numElements=visual_diet_config['get_numElements']





batch_size=training_config['batch_size']
num_workers=training_config['num_workers']
if(training_config['use_device']=='cpu'):
    device='cpu'
else:
    if torch.cuda.is_available():
        device = torch.device('cuda:'+training_config['use_device'])
    else:
        device='cpu'


fine_tune=training_config['fine_tune']
mode='finetune' if training_config['fine_tune'] else 'frozen'
base_model_name=training_config['base_model_name']
layer_name=training_config['layer_name']
num_workers=training_config['num_workers']


if('vit' not in training_config['base_model_name']):
    temp_input_to_base=torch.randn(2,3,512,512)
    img_dim=512
else:
    temp_input_to_base=torch.randn(2,3,384,384)
    img_dim=384


# normalize images using parameters from the training image set
data_transform = transforms.Compose([       
 transforms.Resize(img_dim),                   
 transforms.CenterCrop((img_dim,img_dim)),         
 transforms.ToTensor(),                    
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])

data_transform_without_norm = transforms.Compose([       
 transforms.Resize(img_dim),                   
 transforms.CenterCrop((img_dim,img_dim)),         
 transforms.ToTensor()                    
 ])
################################################################################################################################################################ 









################################################################################################################################################################  
# Loading the dataset

train_dataset_norm = Contour_Dataset(root=root_directory,transform=data_transform,train=True,get_B=get_B,get_D=get_D,get_A=get_A,get_numElements=get_numElements,total_images=5000)
train_loader_norm = torch.utils.data.DataLoader(dataset=train_dataset_norm, batch_size=batch_size, num_workers=num_workers, shuffle=True)

train_dataset_without_norm = Contour_Dataset(root=root_directory,transform=data_transform_without_norm,train=True,get_B=get_B,get_D=get_D,get_A=get_A,get_numElements=get_numElements,total_images=5000)
train_loader_without_norm = torch.utils.data.DataLoader(dataset=train_dataset_without_norm, batch_size=batch_size, num_workers=num_workers, shuffle=True)



val_dataset_norm = Contour_Dataset(root=root_directory,transform=data_transform,train=False,get_B=get_B,get_D=get_D,get_A=get_A,get_numElements=get_numElements)
val_loader_norm = torch.utils.data.DataLoader(dataset=val_dataset_norm, batch_size=batch_size, num_workers=num_workers, shuffle=True)

val_dataset_without_norm = Contour_Dataset(root=root_directory,transform=data_transform_without_norm,train=False,get_B=get_B,get_D=get_D,get_A=get_A,get_numElements=get_numElements)
val_loader_without_norm = torch.utils.data.DataLoader(dataset=val_dataset_without_norm, batch_size=batch_size, num_workers=num_workers, shuffle=True)





dataloaders_dict={'train':train_loader_norm,'val':val_loader_norm}
training_config['len_train_dataset']=len(train_dataset_norm)
training_config['len_val_dataset']=len(val_dataset_norm)


################################################################################################################################################################  







################################################################################################################################################################
# Spliced Model -> Base model + Readout model
spliced_model=SpliceModel(base_model_name,layer_name,fine_tune=fine_tune,device=device)
################################################################################################################################################################








################################################################################################################################################################
# Optimizer, Criterion and Scheduler

# params_to_update = []
# for name,param in readout_model.named_parameters():
#     if param.requires_grad == True:
#         params_to_update.append(param)
params_to_update=list(spliced_model.parameters())


optimizer=optim.SGD(params_to_update, lr=training_config['optimizer_lr'], momentum=training_config['optimizer_momentum'], weight_decay=training_config['optimizer_wd'])

criterion = nn.CrossEntropyLoss()

if(training_config['is_scheduler']):
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=training_config['optimizer_lr'],pct_start=0.4, steps_per_epoch=len(dataloaders_dict['train']), epochs=training_config['total_num_epochs'], div_factor=100)
else:
    scheduler=None
################################################################################################################################################################




################################################################################################################################################################
# Training Loop

def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25,save_model_weights=False):

    since = time.time()  
    

    ## Recording accuracy
    val_acc_history = []
    train_acc_history=[]
    
    ## Recording Loss
    val_loss_history=[]
    train_loss_history=[]
    
    ## Recording optimizer learning rate
    optimizer_lr=[]
    
    
    for epoch in tqdm(range(num_epochs)):


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # model.train()   # Set model to train mode
                change_train_eval_mode(model,model.fine_tune,train_eval_mode='train')
            else:
                # model.eval()   # Set model to evaluate mode
                change_train_eval_mode(model,model.fine_tune,train_eval_mode='eval')

            running_loss = 0.0
            running_corrects = 0

            
            for inputs, b, d, a, nel, labels, record in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs= model.forward(inputs)


                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if(scheduler):
                            scheduler.step()
                            optimizer_lr.append(optimizer.param_groups[0]['lr'])

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                


            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_loss)
                # print(val_acc_history[-1])
            if phase == 'train':
                train_acc_history.append(epoch_acc.item())
                train_loss_history.append(epoch_loss)


    time_elapsed = time.time() - since
    # Save the model checkpoints
    if(save_model_weights):
        torch.save({
                'visual_diet_config':visual_diet_config,
                'training_config':training_config,
                'model_state_dict': get_modified_state_dict(spliced_model,get_untrainable_params(spliced_model)),
                'metrics':{'training_time':time_elapsed,'train_acc':train_acc_history,'train_loss':train_loss_history,'val_acc':val_acc_history,'val_loss':val_loss_history,'optim_lr':optimizer_lr},
                }, './saved_model/model_' + training_config['base_model_name'].replace('_','-') + '_layer_' + training_config['layer_name'].replace('.','-')+'_mode_'+mode+'.pt')
    
    
    return train_acc_history, val_acc_history, train_loss_history, val_loss_history,optimizer_lr


################################################################################################################################################################

# Start the training
train_acc_history, val_acc_history, train_loss_history, val_loss_history,optimizer_lr=train_model(spliced_model, dataloaders_dict, criterion, optimizer, scheduler=scheduler, num_epochs=training_config['total_num_epochs'], save_model_weights=True)

