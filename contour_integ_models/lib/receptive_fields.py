'''
    Determine a unit's receptive field analytically and/or using backprop.
    
    Analytic solution adapted from: https://github.com/Fangyh09/pytorch-receptive-field
    
    Backprop solution adapted from: https://learnopencv.com/cnn-receptive-field-computation-using-backprop/
    
    but modified to have an API that suits our needs, and to separately 
    quantify two versions of the receptive field:
    
    1) input receptive field: should be the same as what you would compute analytically
    
    2) effective receptive field: smaller, reflects that more central pixels have
       more influence, size determined by fitting a 2D gaussian on the gradients
       in pixel space.
    
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import scipy

from collections import namedtuple, OrderedDict

from pdb import set_trace

Rect = namedtuple('Rect', 'x1 y1 x2 y2')
ReceptiveField = namedtuple('ReceptiveField', 'cx cy h w')
GradInputRF = namedtuple('GradInputRF', 'cx cy h w')
GradEffectiveRF = namedtuple('GradEffectiveRF', 'cx cy h w')
AnalyticRF = namedtuple('AnalyticRF', 'cx cy h w')
Gaussian2D = namedtuple('Gaussian2D', 'height x y width_x width_y rotation')
UnitRF = namedtuple('UnitReceptiveField', 'unit grad inp_rf eff_rf')

def rect_to_rf(rect, rf_class=ReceptiveField):
    w = rect.x2 - rect.x1
    h = rect.y2 - rect.y1
    cx = rect.x1 + w//2
    cy = rect.y1 + h//2
    
    return rf_class(cx, cy, h, w)

def rf_to_rect(rf):
    x1 = rf.cx - rf.w//2
    x2 = x1 + rf.w
    
    y1 = rf.cy - rf.h//2
    y2 = x1 + rf.h    
    
    return Rect(x1, y1, x2, y2)
    
def receptive_field_for_unit(model, layer_name, unit=None, image_size=(1,3,224,224),
                             method='gaussian', stds=4, threshold=.15):
    
    h,w = image_size[2], image_size[3]
    grad,unit = backprop_receptive_field(model, layer_name, unit=unit, image_size=image_size)
    rect = find_rect(grad)
    inp_rf = rect_to_rf(rect, GradInputRF)
    if method=="gaussian":
        p = fitgaussian(grad)
        cx,cy = int(p.x), int(p.y)
        w,h = min(w, int(p.width_x*stds)), min(int(p.width_y*stds), h)
        eff_rf = GradEffectiveRF(cx,cy,w,h)
        
    elif method=="threshold":
        rect = find_rect_smooth(grad, thresh=threshold) 
        eff_rf = rect_to_rf(rect, GradEffectiveRF)
    
    return UnitRF(unit, grad, inp_rf, eff_rf)
    
def backprop_receptive_field(model, layer_name, unit=None, image_size=(1,3,224,224)):
    
    model.zero_grad()
    model.train()
    for module in model.modules():
        try:
            nn.init.constant_(module.weight, 0.05) # .05 inference overflows with ones
            nn.init.zeros_(module.bias)
            if hasattr(module, 'running_mean'): 
                nn.init.zeros_(module.running_mean)
            if hasattr(module, 'running_var'):
                nn.init.ones_(module.running_var)
        except:
            pass

        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()
    
    # get output activations for an 'all ones' input image
    input = torch.ones(image_size, requires_grad=True)
    out = get_layer_output(model, layer_name, input)
    
    # set the gradient to 1 for only the selected unit
    grad = torch.zeros_like(out, requires_grad=True)
    if unit is not None:
        row,col = unit
    else:
        # default to as close to center as possible
        rows,cols = out.shape[-2:]
        row,col = rows//2,cols//2
    grad.data[0, 0, row, col] = 1
    
    # backprop from this unit to the input
    out.backward(gradient=grad)
    
    # get the input gradient
    gradient_of_input = input.grad[0, 0].data.numpy()
    gradient_of_input = gradient_of_input / np.amax(gradient_of_input)

    return gradient_of_input, (row,col)

def get_layer_output(model, layer_name, input):
    layer = dict([*model.named_modules()])[layer_name]
    
    activations = {}
    def save_output(layer_name):
        def hook(model, input, output):
            activations[layer_name] = output
        
        return hook
    hook = layer.register_forward_hook(save_output(layer_name))
    
    _ = model(input)
    out = activations[layer_name]
    
    hook.remove()
    
    return out

def find_rect_smooth(activations, thresh=.10):
    # Dilate and erode the activations to remove grid-like artifacts
    kernel = np.ones((5, 5), np.uint8)
    activations = cv2.dilate(activations, kernel=kernel)
    activations = cv2.erode(activations, kernel=kernel)

    # Binarize the activations
    _, activations = cv2.threshold(activations, thresh, 1, type=cv2.THRESH_BINARY)
    activations = activations.astype(np.uint8).copy()

    # Find the countour of the binary blob
    contours, _ = cv2.findContours(activations, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Find bounding box around the object.
    rect = cv2.boundingRect(contours[0])

    return Rect(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])

def find_rect(activations):    

    rows,cols = np.where(activations>0)
    y1,y2 = rows.min(), rows.max()
    x1,x2 = cols.min(), cols.max()
    
    return Rect(x1, y1, x2, y2)

def normalize(activations):
    activations = activations - np.min(activations[:])
    activations = activations / np.max(activations[:])
    return activations

def overlay_activations(image, activations, show_bounding_rect=False, rect_thresh=None,
                        color=(0, 0, 255), thickness=2):
    activations = normalize(activations)

    activations_multichannel = np.stack([activations, activations, activations], axis=2)

    masked_image = (image * activations_multichannel).astype(np.uint8)

    rect = None
    if show_bounding_rect:
        if rect_thresh is not None:
            rect = find_rect_smooth(activations, thresh=rect_thresh)  
        else: 
            rect = find_rect(activations)
        cv2.rectangle(masked_image, (rect.x1, rect.y1), (rect.x2, rect.y2), 
                      color=color, thickness=thickness)

    return masked_image, rect

# ===========================================================
#  2D Gaussian Fit
# ===========================================================

def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    center_x_rot = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y_rot = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x,y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height*np.exp(
            -(((center_x_rot-xp)/width_x)**2+
              ((center_y_rot-yp)/width_y)**2)/2.)
        return g
    return rotgauss

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.0


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    (height, x, y, width_x, width_y, rotation), success = scipy.optimize.leastsq(errorfunction, params)
    assert success, "oops, fit did not converge"
    p = Gaussian2D(height, x, y, width_x, width_y, rotation)
    return p

# ===========================================================
#  Analytic Receptive Fields
# ===========================================================

def check_same(stride):
    if isinstance(stride, (list, tuple)):
        assert len(stride) == 2 and stride[0] == stride[1]
        stride = stride[0]
    return stride

def receptive_field_analytic(model, input_size, batch_size=-1, device="cuda"):
    '''
    :parameter
    'input_size': tuple of (Channel, Height, Width)
    :return  OrderedDict of `Layername`->OrderedDict of receptive field stats {'j':,'r':,'start':,'conv_stage':,'output_shape':,}
    'j' for "jump" denotes how many pixels do the receptive fields of spatially neighboring units in the feature tensor
        do not overlap in one direction.
        i.e. shift one unit in this feature map == how many pixels shift in the input image in one direction.
    'r' for "receptive_field" is the spatial range of the receptive field in one direction.
    'start' denotes the center of the receptive field for the first unit (start) in on direction of the feature tensor.
        Convention is to use half a pixel as the center for a range. center for `slice(0,5)` is 2.5.
    '''
    def register_hook(module):

        def hook(module, input, output):
            
            # must be B x C x H x W to have a receptive field
            if len(input[0].shape) < 4: 
                return
        
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(receptive_field)
            module_name = module_to_layername[module]
            
            # skip resnet downsampling blocks; they downsample the input to the residual block
            # but these ops happen within forward function, make it hard to track
            if "downsample" in module_name: return
            
            m_key = "%i" % module_idx
            p_key = "%i" % (module_idx - 1)
            receptive_field[m_key] = OrderedDict()
            receptive_field[m_key]["idx"] = m_key
            receptive_field[m_key]["name"] = module_name
            
            if not receptive_field["0"]["conv_stage"]:
                print("Enter in deconv_stage")
                receptive_field[m_key]["j"] = 0
                receptive_field[m_key]["r"] = 0
                receptive_field[m_key]["start"] = 0
            else:
                p_j = receptive_field[p_key]["j"]
                p_r = receptive_field[p_key]["r"]
                p_start = receptive_field[p_key]["start"]

                if class_name in ["Conv2d", "MaxPool2d", "AvgPool2d"]:
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    dilation = getattr(module, 'dilation', 1)                                        
                        
                    kernel_size, stride, padding, dilation = map(check_same, [kernel_size, stride, padding, dilation])
                    
                    if module_name == "layer2.0.downsample.0":
                        set_trace()
                        
                    receptive_field[m_key]["j"] = p_j * stride
                    receptive_field[m_key]["r"] = p_r + ((kernel_size - 1) * dilation) * p_j
                    receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j
                elif class_name in ["AdaptiveAvgPool2d", "WeightedAdaptiveAvgPool2d"]:
                    inH,inW = input[0].shape[-2:]
                    outH,outW = module.output_size
                    
                    stride = inH//outH, inW//outW
                    kernel_size = inH - (outH-1)*stride[0], inW - (outW-1)*stride[1]
                    padding = 0
                    dilation = 1
                    
                    kernel_size, stride, padding, dilation = map(check_same, [kernel_size, stride, padding, dilation])
                    receptive_field[m_key]["j"] = p_j * stride
                    receptive_field[m_key]["r"] = p_r + ((kernel_size - 1) * dilation) * p_j
                    receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j
                
                elif class_name in ["BatchNorm2d", "ReLU", "Bottleneck", "Dropout"]:
                    receptive_field[m_key]["j"] = p_j
                    receptive_field[m_key]["r"] = p_r
                    receptive_field[m_key]["start"] = p_start
                elif class_name == "ConvTranspose2d":
                    receptive_field["0"]["conv_stage"] = False
                    receptive_field[m_key]["j"] = 0
                    receptive_field[m_key]["r"] = 0
                    receptive_field[m_key]["start"] = 0
                else:
                    print(module)
                    raise ValueError("module not ok")
                    pass
            receptive_field[m_key]["input_shape"] = list(input[0].size()) # only one
            receptive_field[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                # list/tuple
                receptive_field[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                # tensor
                receptive_field[m_key]["output_shape"] = list(output.size())
                receptive_field[m_key]["output_shape"][0] = batch_size        
            
        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
            and not (len(list(module.children())) > 0)
        ):            
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"
    
    model.to(device)
    
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(2, *input_size)).type(dtype)

    # create properties
    layers_by_name = {k:v for k,v in dict([*model.named_modules()]).items() if k != ''}
    module_to_layername = {v:k for k,v in layers_by_name.items()}
    receptive_field = OrderedDict()
    receptive_field["0"] = OrderedDict()
    receptive_field["0"]["idx"] = "0"
    receptive_field["0"]["name"] = "input"
    receptive_field["0"]["j"] = 1.0
    receptive_field["0"]["r"] = 1.0
    receptive_field["0"]["start"] = 0.5
    receptive_field["0"]["conv_stage"] = True
    receptive_field["0"]["output_shape"] = list(x.size())
    receptive_field["0"]["output_shape"][0] = batch_size
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("------------------------------------------------------------------------------")
    line_new = "{:>25}  {:>11} {:>11} {:>11} {:>15} ".format("Layer (type)", "map size", "start", "jump", "receptive_field")
    print(line_new)
    print("==============================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in receptive_field:
        # input_shape, output_shape, trainable, nb_params
        assert "start" in receptive_field[layer], layer
        assert len(receptive_field[layer]["output_shape"]) == 4
        line_new = "{:1} {:25}  {:>10} {:>10} {:>10} {:>15} ".format(
            "",
            f"{layer}. {receptive_field[layer]['name']}",
            str(receptive_field[layer]["output_shape"][2:]),
            str(receptive_field[layer]["start"]),
            str(receptive_field[layer]["j"]),
            format(str(receptive_field[layer]["r"]))
        )
        print(line_new)

    print("==============================================================================")
    
    # enable indexing by layer_name
    for layer in [k for k in receptive_field.keys()]:
        name = receptive_field[layer]["name"]
        receptive_field[name] = receptive_field[layer]
    
    # add input_shape
    receptive_field["input_size"] = input_size
    
    return receptive_field


def analytic_receptive_field_for_unit(receptive_field_dict, layer, unit_position):
    """Utility function to calculate the receptive field for a specific unit in a layer
        using the dictionary calculated above
    :parameter
        'layer': layer name, should be a key in the result dictionary
        'unit_position': spatial coordinate of the unit (H, W)
    ```
    alexnet = models.alexnet()
    model = alexnet.features.to('cuda')
    receptive_field_dict = receptive_field_analytic(model, (3, 224, 224))
    analytic_receptive_field_for_unit(receptive_field_dict, "8", (6,6))
    ```
    Out: [(62.0, 161.0), (62.0, 161.0)]
    """
    input_shape = receptive_field_dict["input_size"]
    if layer in receptive_field_dict:
        rf_stats = receptive_field_dict[layer]
        assert len(unit_position) == 2
        feat_map_lim = rf_stats['output_shape'][2:]
        if np.any([unit_position[idx] < 0 or
                   unit_position[idx] >= feat_map_lim[idx]
                   for idx in range(2)]):
            raise Exception("Unit position outside spatial extent of the feature tensor ((H, W) = (%d, %d)) " % tuple(feat_map_lim))
        # X, Y = tuple(unit_position)
        rf_range = [(rf_stats['start'] + idx * rf_stats['j'] - rf_stats['r'] / 2,
            rf_stats['start'] + idx * rf_stats['j'] + rf_stats['r'] / 2) for idx in unit_position]
        if len(input_shape) == 2:
            limit = input_shape
        else:  # input shape is (channel, H, W)
            limit = input_shape[1:3]
        rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(2)]
        print("Receptive field size for layer %s, unit_position %s,  is \n %s" % (layer, unit_position, rf_range))
        
        ys, xs = rf_range
        h = ys[1] - ys[0]
        w = xs[1] - xs[0]
        cy = ys[0] + h/2
        cx = xs[0] + w/2
        rf = AnalyticRF(cx, cy, h, w)
        
        return rf
    else:
        raise KeyError("Layer name incorrect, or not included in the model.")
        
        
# ===========================================================
#  Visualization Helpers
# ===========================================================

def plot_receptive_field(rf, an_rf, img=None, show_inp_rect=True, show_eff_circle=True, show_eff_rect=False, show_an_rect=True,
                         show_inp_center=False, show_eff_center=False, show_an_center=True, center_size=3,
                         inp_color=(180,0,0), eff_color=(0,0,255), an_color=(0,220,0), thickness=2):
    
    activations = rf.grad
    if img==None:
        img = (np.ones((*activations.shape,3))*255).astype(np.uint8)
        
    img = overlay_activations(img, activations)
    
    if show_inp_rect:
        rect = rf_to_rect(rf.inp_rf)
        pt1, pt2 = (rect.x1, rect.y1), (rect.x2, rect.y2)
        cv2.rectangle(img, pt1, pt2, color=inp_color, thickness=thickness)
        
    if show_inp_center:
        cv2.circle(img, (rf.inp_rf.cx,rf.inp_rf.cy), center_size, inp_color, thickness=-1)
    
    if show_inp_center:
        cv2.circle(img, (rf.eff_rf.cx,rf.eff_rf.cy), center_size, eff_color, thickness=-1)
        
    if show_eff_rect:
        rect = rf_to_rect(rf.eff_rf)
        pt1, pt2 = (rect.x1, rect.y1), (rect.x2, rect.y2)
        cv2.rectangle(img, pt1, pt2, color=eff_color, thickness=thickness)
        
    if show_eff_circle:
        size = (rf.eff_rf.w, rf.eff_rf.h) # width/height = 4std
        rotated_rect = ((rf.eff_rf.cx, rf.eff_rf.cy), size, 0)
        box = cv2.boxPoints(rotated_rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,eff_color,2)
    
    if show_an_rect and an_rf is not None:
        rect = rf_to_rect(an_rf)
        pt1, pt2 = np.int0((rect.x1, rect.y1)), np.int0((rect.x2, rect.y2))
        cv2.rectangle(img, pt1, pt2, color=an_color, thickness=thickness)
    
    if show_an_center and an_rf is not None:
        cv2.circle(img, np.int0((an_rf.cx,an_rf.cy)), center_size, an_color, thickness=-1)
        
    return img
        
def normalize(activations):
    activations = activations - np.min(activations[:])
    activations = activations / np.max(activations[:])
    return activations

def overlay_activations(image, activations, show_bounding_rect=False, rect_thresh=None,
                        color=(0, 0, 255), thickness=2):
    activations = normalize(activations)

    activations_multichannel = np.stack([activations, activations, activations], axis=2)

    masked_image = (image * activations_multichannel).astype(np.uint8)

    return masked_image  