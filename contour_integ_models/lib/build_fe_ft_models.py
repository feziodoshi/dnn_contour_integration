import sys
sys.path.insert(1,'../')
sys.path.insert(1,'./')

import torch
import torchvision
import torch.nn as nn
from torch.nn import ReLU
import torchvision.models as pretrained_models
from pdb import set_trace
from torch.utils import model_zoo
import numpy as np
import os



# Extra imports
from lib.feature_extractor import FeatureExtractor
from lib.model_archs import barlow_twins
from lib.model_archs import alexnet_bagnet
from lib.misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


from pytorch_pretrained_vit import ViT
import bagnets.pytorchnet



print('updated june4')
################################################################################################################################################
def get_base_model(base_model_name):
    base_model=None
    if(base_model_name=='gabornet_conv'):
        base_model=torch.load('../../model_weights/base_model_weights/gabornet.pt')
        
    elif(base_model_name=='alexnet-pytorch_regim_categ'):
        base_model=pretrained_models.alexnet(pretrained=True)
        
    elif(base_model_name=='alexnet-random_nodata_notask'):
        # print(os.path.exists('../../dev/base_model_weights/alexnet_random.pth'))
        # base_model=torch.load('../../dev/base_model_weights/alexnet_random.pth')
        
        
        base_model=torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_weights/base_model_weights/alexnet_random.pth'))
        
        
    elif(base_model_name=='resnet50-pytorch_regim_categ'):
        base_model=pretrained_models.resnet50(pretrained=True)
        
    elif(base_model_name=='resnet50-bagnet9_regim_categ'):
        base_model=bagnets.pytorchnet.bagnet9(pretrained=True)
        base_model.layer4.add_module("adaptive",nn.AdaptiveAvgPool2d(output_size=(1,1)))
        # base_model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
    elif(base_model_name=='resnet50-bagnet33_regim_categ'):
        base_model=bagnets.pytorchnet.bagnet33(pretrained=True)
        base_model.layer4.add_module("adaptive",nn.AdaptiveAvgPool2d(output_size=(1,1)))
        # base_model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
    elif(base_model_name=='alexnet_styim_categ'):
        link_model='https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar'
        base_model = torchvision.models.alexnet(pretrained=False)
        checkpoint = model_zoo.load_url(link_model)
        
        transformed_keys={}
        for key in checkpoint["state_dict"].keys():
            if('module' in key):
                x=key.split('.')
                new_key='.'.join(np.array(x)[[0,2,3]])
                transformed_keys[key]=new_key

        for key in transformed_keys.keys():
            checkpoint["state_dict"][transformed_keys[key]]=checkpoint["state_dict"][key]
            checkpoint["state_dict"].pop(key)
        
        base_model.load_state_dict(checkpoint["state_dict"])
        
    elif(base_model_name=='alexnet-barlow_regim_barlow'):
        base_model,_=barlow_twins.alexnet_gn_barlow_twins(pretrained=True,filename=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_weights/base_model_weights/barlow_reserailized.pt')))
        # base_model,_=barlow_twins.alexnet_gn_barlow_twins(pretrained=True,filename='../../dev/base_model_weights/barlow_reserailized.pt')
        
    elif(base_model_name=='vit_regular'):    
        base_model = ViT('B_16_imagenet1k', pretrained=True)
        
    elif(base_model_name=='alexnet-bagnet33_regim_categ'):
        base_model=alexnet_bagnet.alexnet_bagnet33(pretrained=True,filename=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_weights/base_model_weights/alexnet_bagnet33_137331_final_weights_pytorch2-3b5d8dae71.pth')))
        
        
    elif(base_model_name=='alexnet-bagnet31_regim_categ'):
        base_model=alexnet_bagnet.alexnet_bagnet31(pretrained=True,filename=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_weights/base_model_weights/alexnet_bagnet31_115333_final_weights_pytorch2-25355f56ff.pth')))
        
    elif(base_model_name=='alexnet-bagnet17_regim_categ'):
        base_model=alexnet_bagnet.alexnet_bagnet17(pretrained=True,filename=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_weights/base_model_weights/alexnet_bagnet17_93311_final_weights_pytorch2-b051786873.pth')))
    
    elif(base_model_name=='alexnet-bagnet11_regim_categ'):
        base_model=alexnet_bagnet.alexnet_bagnet11(pretrained=True,filename=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_weights/base_model_weights/alexnet_bagnet11_72211_final_weights_pytorch2-d37f3073cd.pth')))
    
    elif(base_model_name=='alexnet-bagnet09_regim_categ'):
        base_model=alexnet_bagnet.alexnet_bagnet9(pretrained=True,filename=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_weights/base_model_weights/alexnet_bagnet9_73111_final_weights_pytorch2-26c23b5440.pth')))
    
    elif(base_model_name=='alexnet-epoch50_regim_categ'):
        base_model=alexnet_bagnet.alexnet_epoch(pretrained=True,filename=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_weights/base_model_weights/alexnet_epoch050_pytorch2weights-f8ee0e1895.pth')))
    
    elif(base_model_name=='alexnet-epoch60_regim_categ'):
        base_model=alexnet_bagnet.alexnet_epoch(pretrained=True,filename=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../dev/base_model_weights/alexnet_epoch060_pytorch2weights-7322541ea3.pth')))
        
    elif(base_model_name=='alexnet-epoch100_regim_categ'):
        base_model=alexnet_bagnet.alexnet_epoch(pretrained=True,filename=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_weights/base_model_weights/alexnet_final_weights_pytorch2-92c51abe44.pth')))
        
    return base_model
        
def get_readout_model(base_model,layer_name,temp_input_to_base=torch.randn(1,3,512,512)):   
    temp_input_to_readout=None
    
    with FeatureExtractor(base_model, layer_name) as extractor:

        features = extractor(temp_input_to_base)
        temp_input_to_readout=features[layer_name]
        if(len(temp_input_to_readout.shape)!=2):
            temp_input_to_readout=temp_input_to_readout.view((temp_input_to_readout.shape[0],-1))
            
    readout_model=nn.Sequential(*[nn.Linear(temp_input_to_readout.shape[1],2)])
    
    return readout_model
################################################################################################################################################





################################################################################################################################################
def freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = False     
           
################################################################################################################################################        
  
    
    
    
    
################################################################################################################################################
# Function works on spliced models only
def get_untrainable_params(spliced_model):
    
    # assert isinstance(spliced_model,SpliceModel)
    list_untrainable_parameters=[]
    torch.mean(spliced_model.forward(spliced_model.temp_input_to_base)).backward()

    for name,param in spliced_model.named_parameters():
        if(param.requires_grad==False):
            # All the frozen parameters
            list_untrainable_parameters.append(name)
        else:
            ## This is for the fine-tune model parameters which are after the hooked layer so, not frozen by default but are unused and not updated
            if(param.grad==None):
                list_untrainable_parameters.append(name)
    
    return list_untrainable_parameters


# Function works on spliced models only
def get_modified_state_dict(spliced_model,list_untrainable_parameters):
    temp_dict=spliced_model.state_dict()
    for key in list_untrainable_parameters:
        del temp_dict[key]
    return temp_dict
################################################################################################################################################



################################################################################################################################################
# Function works on spliced models only
def change_train_eval_mode(spliced_model,fine_tune,train_eval_mode='eval'):
    if(fine_tune):
        if(train_eval_mode=='train'):
            spliced_model.train()
        elif(train_eval_mode=='eval'):
            spliced_model.eval()
    else:
        if(train_eval_mode=='train'):
            spliced_model.train()
            spliced_model.base_model.eval()
            # This is just for safety check, already true because whole sploiced model is in training mode
            spliced_model.readout_model.train()
        elif(train_eval_mode=='eval'):
            spliced_model.eval()
################################################################################################################################################





################################################################################################################################################
class SpliceModel(nn.Module):
    def __init__(self, base_model_name, layer_name,fine_tune=True, device='cpu'):
        super().__init__()
        
        self.base_model_name=base_model_name
        self.layer_name=layer_name
        self.fine_tune=fine_tune
        self.device=device
        self.temp_input_to_base=None
        
        
        self.base_model=get_base_model(self.base_model_name).to(self.device)
        self.arrange_base_model()
        
        
        self.readout_model=get_readout_model(self.base_model,self.layer_name,temp_input_to_base=self.temp_input_to_base).to(self.device)
        
        
    
    def arrange_base_model(self):
        if(self.fine_tune):
            ## Do nothing
            pass
        else:
            ##freeze the weights of self.base model
            freeze_all_parameters(self.base_model)
            self.base_model.eval()
        
        
        if('vit' not in self.base_model_name):
            self.temp_input_to_base=torch.randn(2,3,512,512).to(self.device)

        else:
            self.temp_input_to_base=torch.randn(2,3,384,384).to(self.device)
            
            
    
    
    
    def forward(self, x):
        
        output=None
        with FeatureExtractor(self.base_model, self.layer_name, detach=False, clone=True, retain=False) as extractor:

            features = extractor(x)
            temp_input_to_readout=features[self.layer_name].to(self.device)
            if(len(temp_input_to_readout.shape)!=2):
                temp_input_to_readout=temp_input_to_readout.view((temp_input_to_readout.shape[0],-1))

            output= self.readout_model(temp_input_to_readout)
        return output
  ################################################################################################################################################






################################################################################################################################################
# Visualization Function and Script

class Spliced_GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image on a Spliced Model
    """
    def __init__(self, spliced_model,device='cpu',print_hooked_layers=False):
        self.model = spliced_model
        self.gradients = None
        
        self.base_model_name=spliced_model.base_model_name

        
        
        # Put model in evaluation mode
        self.model.eval()
        
        
        self.device=device
        self.print_hooked_layers=print_hooked_layers
        
        
        self.update_relus()
        self.hook_layers()
        
        
        

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        # Currentl;y dependent on the model architecture
        if(self.base_model_name == 'alexnet-pytorch_regim_categ' or self.base_model_name == 'alexnet_styim_categ' or self.base_model_name == 'alexnet-bagnet33_regim_categ' or self.base_model_name == 'alexnet-bagnet17_regim_categ' or self.base_model_name == 'alexnet-epoch50_regim_categ' or self.base_model_name == 'alexnet-epoch60_regim_categ' or self.base_model_name == 'alexnet-epoch100_regim_categ'):
            first_layer = list(self.model.base_model.features._modules.items())[0][1]

        elif(self.base_model_name == 'resnet50-pytorch_regim_categ' or self.base_model_name == 'resnet50-bagnet9_regim_categ' or self.base_model_name == 'resnet50-bagnet33_regim_categ'):
            first_layer = list(self.model.base_model._modules.items())[0][1]
            
        elif(self.base_model_name == 'alexnet-barlow_regim_barlow'):
            first_layer=list(self.model.base_model._modules.items())[0][1][0][0]
        
        else:
            first_layer = list(self.model.base_model._modules.items())[0][1]
            
            
            
        
        first_layer.register_backward_hook(hook_function)


    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            
            new_grad_in=torch.clamp(grad_in[0],min=0.0)
            
            return (new_grad_in,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)
            
        
        def run_function(model):
            'This should hook all the relu layers in the backbone'
            found_leaf_layer=False
            def get_leaf_nodes(model,parent_name=''):
                nonlocal found_leaf_layer

                for child_name,module in model.named_children():


                    layer=layer=parent_name + '.' + child_name
                    if(len(list(module.named_children()))):
                        ## Not a lead module, so dig deeper
                        get_leaf_nodes(module,layer)

                    else:
                        # print(layer,module)
                        # print(layer[1:])

                        if(layer[1:] == self.model.layer_name):
                            found_leaf_layer=True

                        if(not found_leaf_layer and isinstance(module, ReLU)):
                            if(self.print_hooked_layers):
                                print(layer[1:])
                            module.register_backward_hook(relu_backward_hook_function)
                
            get_leaf_nodes(model)
    
    
        run_function(self.model.base_model)
        


    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(self.device)
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.detach().cpu().numpy()[0]
        return gradients_as_arr



################################################################################################################################################
