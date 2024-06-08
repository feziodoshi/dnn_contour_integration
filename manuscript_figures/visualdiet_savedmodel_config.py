import numpy as np
import os 





######################### VISUAL DIET   ##################################
## Visual Diet 
visual_diet_config = {'root_directory':os.path.expanduser('/home/jovyan/work/Datasets/contour_integration/model-training/config_0/'),'get_B':[0,15,30,45,60,75],'get_D':[32],'get_A':[0],'get_numElements':[12]}

## Psychophysics Visual Diet
psychophysics_visual_diet_config = {'root_directory':os.path.expanduser('/home/jovyan/work/Datasets/contour_integration/model-psychophysics/experiment_1/'),'get_B':[15,30,45,60,75],'get_D':[32],'get_A':[0],'get_numElements':[12]}


## Checkpoin Diet 
# saved_model_config = {'saved_model_directory': os.path.join(os.path.dirname(os.path.realpath(__file__)), '../saved_model/')}
# saved_model_config = {'saved_model_directory_or_frozen_broad': '../../model_weights/contour_model_weights/alexnet_regimagenet_categ_frozen_broad/',
#                      'saved_model_directory_or_finetune_broad': '../../model_weights/contour_model_weights/alexnet_regimagenet_categ_finetune_broad/',
#                      'saved_model_directory_random_frozen_broad': '../../model_weights/contour_model_weights/alexnet-random-nodata-notask_frozen_broad/'}




