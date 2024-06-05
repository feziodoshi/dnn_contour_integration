import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches


from itertools import combinations
import pickle
from tqdm.notebook import tqdm
import os
from PIL import Image
from PIL import ImageFilter

            
class Contour_Dataset(Dataset):
    
    def __init__(self, root, get_B=[30,60], get_D=[32],get_A=[0], get_numElements=[12], train=True,total_images=None,transform=None):

        if not isinstance(get_B, (list,)):
            get_B = [get_B]
            
            
        if not isinstance(get_D, (list,)):
            get_D = [get_D]
            
        
        if not isinstance(get_A, (list,)):
            get_A = [get_A]
            
        if not isinstance(get_numElements, (list,)):
            get_numElements = [get_numElements]
              
                
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        
        if(train):
            self.df = pd.read_csv(os.path.join(self.root,'train.csv'))
        else:
            self.df = pd.read_csv(os.path.join(self.root,'val.csv'))
        
        
        self.conditional_df=self.df.loc[(self.df['B'].isin(get_B)) & (self.df['D'].isin(get_D)) & (self.df['A'].isin(get_A)) & (self.df['numElements'].isin(get_numElements))]
        
        
        ## Getting a controlled set BASED ON BETA VALUES
        complete_df=[]
        for current_beta in get_B:
            beta_df=self.conditional_df.loc[self.conditional_df['B']==current_beta]
            if(total_images!=None):
                selected_id_list=np.random.choice(list(set(list(beta_df['id_num']))),(total_images//len(get_B))//2,replace=False)
            else:
                selected_id_list=np.random.choice(list(set(list(beta_df['id_num']))),len(list(set(list(beta_df['id_num'])))),replace=False)
                
            complete_df.append(beta_df.loc[beta_df['id_num'].isin(selected_id_list)])
        self.conditional_df=pd.concat(complete_df)
        
        
        self.img_path=self.conditional_df.img_path
        self.img_B=self.conditional_df.B
        self.img_D=self.conditional_df.D
        self.img_A=self.conditional_df.A
        self.img_numElements=self.conditional_df.numElements
        self.img_contour=self.conditional_df.c
        self.img_recorder_path=self.conditional_df.img_recorder_path
          
    def __getitem__(self, index):
        
        ## Get the images
        img = Image.open(self.img_path.iloc[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        
        ## get the Labels
        img_B=self.img_B.iloc[index]
        img_D=self.img_D.iloc[index]
        img_A=self.img_A.iloc[index]
        img_numElements=self.img_numElements.iloc[index]
        img_contour = self.img_contour.iloc[index]
        if(img_contour=='contour'):
            img_contour=1
        else:
            img_contour=0
        img_recorder_path=self.img_recorder_path.iloc[index]

        return img, img_B, img_D, img_A, img_numElements, img_contour, img_recorder_path
    
    def __len__(self):
        '''
        Return the length of the complete dataset
        '''
        return len(self.conditional_df)
    
    def condition_frequency(self,condition):
        print('TOTAL Datapoints: ',len(self.conditional_df))
        if(condition in list(self.conditional_df.columns)):
            for i in set(list(self.conditional_df[condition])):
                print('Condition/Factor ',condition, ':\t', i,'\t',len(np.where(self.conditional_df[condition]==i)[0]))
        else:
            raise Exception('Condition/Factor not in df')
            
            
            
            
            
            
            
# class Psychophysics_Contour_Dataset(Dataset):
    
#     def __init__(self, root, get_B=[30,60], get_alpha=[0], get_D=[32], get_bg=[0],transform=None):

#         if not isinstance(get_B, (list,)):
#             get_B = [get_B]
            
#         if not isinstance(get_bg, (list,)):
#             get_bg = [get_bg]
            
#         if not isinstance(get_bg, (list,)):
#             get_bg = [get_bg]
              
#         self.root = os.path.expanduser(root)
#         self.transform = transform
        
#         self.df = pd.read_csv(os.path.join(self.root,'psychophysics.csv'))
        
# #         if(train):
# #             self.df = pd.read_csv(os.path.join(self.root,'train.csv'))
# #         else:
# #             self.df = pd.read_csv(os.path.join(self.root,'val.csv'))
        
        
#         self.conditional_df=self.df.loc[(self.df['bg'].isin(get_bg)) & (self.df['B'].isin(get_B)) & (self.df['D'].isin(get_D))]
#         self.img_path=self.conditional_df.img_path
#         self.img_D=self.conditional_df.D
#         self.img_B=self.conditional_df.B
#         self.img_A=self.conditional_df.A
#         self.img_contour=self.conditional_df.c
#         self.img_recorder_path=self.conditional_df.img_recorder_path
          
#     def __getitem__(self, index):
        
#         ## Get the images
#         img = Image.open(self.img_path.iloc[index]).convert('RGB')
#         if self.transform:
#             img = self.transform(img)

        
#         ## get the Labels
#         img_D=self.img_D.iloc[index]
#         img_B=self.img_B.iloc[index]
#         img_A=self.img_A.iloc[index]
#         img_recorder_path=self.img_recorder_path.iloc[index]
#         img_contour = self.img_contour.iloc[index]
#         if(img_contour=='contour'):
#             img_contour=1
#         else:
#             img_contour=0

#         return img, img_B, img_contour, img_recorder_path
    
#     def __len__(self):
#         '''
#         Return the length of the complete dataset
#         '''
#         return len(self.conditional_df)
    
#     def condition_frequency(self,condition):
#         print('TOTAL Datapoints: ',len(self.conditional_df))
#         if(condition in list(self.conditional_df.columns)):
#             for i in set(list(self.conditional_df[condition])):
#                 print('Condition: ',i,'\t',len(np.where(self.conditional_df[condition]==i)[0]))
#         else:
#             raise Exception('Condition not in df')
        
        
 