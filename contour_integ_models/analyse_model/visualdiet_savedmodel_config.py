import numpy as np
import os 





######################### VISUAL DIET   ##################################
## Visual Diet 
visual_diet_config = {'root_directory':os.path.expanduser('/home/jovyan/work/Datasets/contour_integration/model-training/config_0/'),'get_B':[0,15,30,45,60,75],'get_D':[32],'get_A':[0],'get_numElements':[12]}


## Checkpoin Diet 
# saved_model_config = {'saved_model_directory': os.path.join(os.path.dirname(os.path.realpath(__file__)), '../saved_model/')}
saved_model_config = {'saved_model_directory': '../saved_model/'}

