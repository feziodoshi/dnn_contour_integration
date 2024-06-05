import numpy as np
import os 

model_training_dict={
    
    'experiment_dir': '/home/jovyan/work/Datasets/contour_integration/model-training/config_0',    
    'num_images_train':2500,
    'num_images_val':50,
    
    
    'num_elements_list':[12], ## these are the number of path elements -> total elemenst are 256, rest our background
    'alpha_list':[0], ## we do 90-this quantity directly in the code, so 0 are alignede with path and 90 are perpendicular
    'beta_list':np.arange(91), ## List of the beta values
    'D_list':[32.0],
    
    'jitterB':10,
    'jitterD':0.25,
    'gridSize':(16,16),
    'imWidth':512,
    'imHeight':512,
    'startRadius':64,
    
    
    'gabor_lambda':8,
    'gabor_phase':-90,
    'gabor_stddev':4.0,
    'gabor_imSize':28,
    'gabor_elCentre':None,
    'gabor_gratingContrast':1.0
    
}


##################################################################################################################################################################
##################################################################################################################################################################



psychophysics_experiment1={
    
    'experiment_dir': '/home/jovyan/work/Datasets/contour_integration/model-psychophysics/experiment_1',    
    'num_images_psychophysics':200,
    
    
    'num_elements_list':[12], ## these are the number of path elements -> total elemenst are 256, rest our background
    'alpha_list':[0], ## we do 90-this quantity directly in the code, so 0 are alignede with path and 90 are perpendicular
    'beta_list':[15,30,45,60,75], ## List of the beta values
    'D_list':[32.0],
    
    'jitterB':10,
    'jitterD':0.25,
    'gridSize':(16,16),
    'imWidth':512,
    'imHeight':512,
    'startRadius':64,
    
    
    'gabor_lambda':8,
    'gabor_phase':-90,
    'gabor_stddev':4.0,
    'gabor_imSize':28,
    'gabor_elCentre':None,
    'gabor_gratingContrast':1.0
    
}