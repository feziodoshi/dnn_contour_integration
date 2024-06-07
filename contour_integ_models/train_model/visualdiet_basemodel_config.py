import numpy as np
import os 





######################### VISUAL DIET   ##################################
## Broad Diet 
visual_diet_config={'root_directory':os.path.expanduser('/home/jovyan/work/Datasets/contour_integration/model-training/config_0/'),'get_B':[0, 15, 30, 45, 60, 75],'get_D':[32],'get_A':[0],'get_numElements':[12]}

## Narrow Diet
# visual_diet_config={'root_directory':os.path.expanduser('/home/jovyan/work/Datasets/contour_integration/model-training/config_0/'),'get_B':[20],'get_D':[32],'get_A':[0],'get_numElements':[12]}
###########################################################################












######################### BASE MODEL AND TRAINING PARAMETERS   ##################################
## set fine_tune to False for frozen backbone



## Base Model: Imagenet- Object Recognition - Alexnet (frozen)
## Layer: Avgpool
training_config={'fine_tune':False,'base_model_name':'alexnet-pytorch_regim_categ','layer_name':'avgpool',
                 'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
                 'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}



## Base Model: Imagenet- Object Recognition - Alexnet (fine-tune)
## Layer: Avgpool
# training_config={'fine_tune':True,'base_model_name':'alexnet-pytorch_regim_categ','layer_name':'avgpool',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}









## Base Model: Random Alexnet (frozen)
## Layer: Avgpool
# training_config={'fine_tune':False,'base_model_name':'alexnet-random_nodata_notask','layer_name':'avgpool',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}


## Base Model: Random Alexnet (fine-tune)
## Layer: Avgpool
# training_config={'fine_tune':True,'base_model_name':'alexnet-random_nodata_notask','layer_name':'avgpool',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}








## Stylized- Object Recognition - Alexnet (frozen)
## Layer: Avgpool
# training_config={'fine_tune':True,'base_model_name':'alexnet_styim_categ','layer_name':'avgpool',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}


## Stylized- Object Recognition - Alexnet (fine-tune)
## Layer: Avgpool
# training_config={'fine_tune':False,'base_model_name':'alexnet_styim_categ','layer_name':'avgpool',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}











# # Base Model: PinholeNet33 (fine-tune)
# # Layer: features.9
# training_config={'fine_tune':True,'base_model_name':'alexnet-bagnet33_regim_categ','layer_name':'features.9',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}


## Base Model: PinholeNet31 (fine-tune)
## Layer: features.9
# training_config={'fine_tune':True,'base_model_name':'alexnet-bagnet31_regim_categ','layer_name':'features.9',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}



## Base Model: PinholeNet17 (fine-tune)
## Layer: features.9
# training_config={'fine_tune':True,'base_model_name':'alexnet-bagnet17_regim_categ','layer_name':'features.9',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}



## Base Model: PinholeNet11 (fine-tune)
## Layer: features.9
# training_config={'fine_tune':True,'base_model_name':'alexnet-bagnet11_regim_categ','layer_name':'features.9',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}









## Base Model: Imagenet- Object Recognition - Alexnet (fine-tune) and trained till epoch 50
## Layer: Avgpool
# training_config={'fine_tune':True,'base_model_name':'alexnet-epoch50_regim_categ','layer_name':'features.12',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}


## Base Model: Imagenet- Object Recognition - Alexnet (fine-tune) and trained till epoch 100
## Layer: Avgpool
# training_config={'fine_tune':True,'base_model_name':'alexnet-epoch100_regim_categ','layer_name':'features.12',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}









# Imagenet- Barlow - Alexnet (frozen)
# Layer: backbone
# training_config={'fine_tune':False,'base_model_name':'alexnet-barlow_regim_barlow','layer_name':'backbone',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}











## Base Model: Imagenet- Object Recognition - Resnet50 (fine-tune)
## Layer: Avgpool
# training_config={'fine_tune':True,'base_model_name':'resnet50-pytorch_regim_categ','layer_name':'avgpool',
#                  'batch_size':8,'num_workers':2, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}



## Base Model: Imagenet- Object Recognition - Bagnet9 (fine-tune)
## Layer: layer4
# training_config={'fine_tune':True,'base_model_name':'resnet50-bagnet9_regim_categ','layer_name':'layer4',
#                  'batch_size':4,'num_workers':2, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}


## Base Model: Imagenet- Object Recognition - Bagnet33 (fine-tune)
## Layer: layer4
# training_config={'fine_tune':True,'base_model_name':'resnet50-bagnet33_regim_categ','layer_name':'layer4',
#                  'batch_size':4,'num_workers':2, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}



## Base Model: Gabornet (frozen)
## Layer: gabor
# training_config={'fine_tune':False,'base_model_name':'gabornet_conv','layer_name':'gabor',
#                  'batch_size':8,'num_workers':32, 'use_device':'2','total_num_epochs':100,
#                  'optimizer_name':'sgd','optimizer_lr':0.0001,'optimizer_momentum':0.9,'optimizer_wd':0.0, 'is_scheduler':True}



###########################################################################