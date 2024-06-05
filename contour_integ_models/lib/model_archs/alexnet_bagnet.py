from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path 
from torch.hub import load_state_dict_from_url
from pdb import set_trace
from collections import OrderedDict
    
    
######################################################################################################    
class AlexNet_Epoch(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
def alexnet_epoch(pretrained=False,filename = './bagnet.pt'):
    model = AlexNet_Epoch()
    
    if pretrained: 
        url = os.path.join(filename)        
        print(f"... loading checkpoint: "+url)
        
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint, strict=True) 

    return model 
    
    
######################################################################################################

    
######################################################################################################
## Alexnet Bagnet RF 33
class AlexNetBagnet33_137331(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=13, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=7, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 



def alexnet_bagnet33(pretrained=False,filename = './bagnet.pt'):
    model = AlexNetBagnet33_137331()
    
    if pretrained: 
        url = os.path.join(filename)        
        print(f"... loading checkpoint: "+url)
        
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint, strict=True) 

    return model 

######################################################################################################



######################################################################################################
## Alexnet Bagnet RF 31
class AlexNetBagnet31_115333(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
def alexnet_bagnet31(pretrained=False,filename = './bagnet.pt'):
    model = AlexNetBagnet31_115333()
    
    if pretrained: 
        url = os.path.join(filename)        
        print(f"... loading checkpoint: "+url)
        
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint, strict=True) 

    return model


######################################################################################################




######################################################################################################
## Alexnet Bagnet RF 17
class AlexNetBagnet17_93311(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
def alexnet_bagnet17(pretrained=False,filename = './bagnet.pt'):
    model = AlexNetBagnet17_93311()
    
    if pretrained: 
        url = os.path.join(filename)        
        print(f"... loading checkpoint: "+url)
        
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint, strict=True) 

    return model


######################################################################################################





######################################################################################################
## Alexnet Bagnet RF 11
class AlexNetBagnet11_72211(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
def alexnet_bagnet11(pretrained=False,filename = './bagnet.pt'):
    model = AlexNetBagnet11_72211()
    
    if pretrained: 
        url = os.path.join(filename)        
        print(f"... loading checkpoint: "+url)
        
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint, strict=True) 

    return model


######################################################################################################




######################################################################################################
## Alexnet Bagnet RF 9
class AlexNetBagnet9_73111(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # Two pooling steps because AdaptiveAvgPool2d on large maps throws an error
        # The initial nn.AvgPool2d cuts the map in half, then does the adaptive pool
        # Quick tests suggest these are ~ equivalent, but this is more memory efficient
        self.avgpool = nn.Sequential(OrderedDict([
            ('pool1', nn.AvgPool2d(kernel_size=2, stride=2)),
            ('pool2', nn.AdaptiveAvgPool2d((6, 6)))
        ]))
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
def alexnet_bagnet9(pretrained=False,filename = './bagnet.pt'):
    model = AlexNetBagnet9_73111()
    
    if pretrained: 
        url = os.path.join(filename)        
        print(f"... loading checkpoint: "+url)
        
        checkpoint = torch.load(url)
        model.load_state_dict(checkpoint, strict=True) 

    return model


######################################################################################################