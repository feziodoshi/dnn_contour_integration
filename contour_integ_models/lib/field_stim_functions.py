# Import python libraries
import numpy as np # fundamental package for scientific computing
import matplotlib.pyplot as plt # package for plot function
import math 
import torch
from PIL import Image, ImageStat
from matplotlib.pyplot import imshow
# from IPython.html.widgets import interact, interactive, fixed
# from IPython.display import display 
from tqdm.notebook import tqdm
import os
import random
import cv2


################################################################################################################
# Helper Functions
def imscale(im, minVal=None, maxVal=None):
    # first resclae the image between 0 and 1 -> It is important to set the bounds as -1 and 1 because the data itself will be restricted between -1 and 1
    minVal = im.min() if minVal is None else minVal
    maxVal = im.max() if maxVal is None else maxVal
    tempIm = (im - minVal)/(maxVal-minVal);
    
    return tempIm



def show_gabor(im):
    im = imscale(im, -1, 1)
    im = np.array(im*255).astype(np.uint8)
    return Image.fromarray(im)

def show_im(im):
    im = imscale(im, -1, 1)
    im = np.array(im*255).astype(np.uint8)
    return Image.fromarray(im)




def blend(image_1,image_2,alpha_image_2=0.8):
    if(len(np.array(image_1).shape)!=2):
        image_1=Image.fromarray(np.array(image_1)[:,:,0])
    
    if(len(np.array(image_2).shape)!=2):
        image_2=Image.fromarray(np.array(image_2)[:,:,0])
    
    return Image.blend(image_1,image_2,alpha_image_2)



def blend_checkerboard(img, gridSize=(16,16), imSize=(512,512)):
    imH,imW = imSize
    gridH,gridW = gridSize

    image = np.zeros((gridH,gridW,3))
    grid = np.zeros((gridH,gridW))
    grid[::2,::2] = 1
    grid[1::2,1::2] = 1

    image[grid==0,0] = .75
    image[grid==1,0] = .55
    checkerboard = Image.fromarray((image*255).astype(np.uint8)).resize(imSize, resample=Image.NEAREST)

    return blend(img, checkerboard, alpha_image_2=0.5)


################################################################################################################




################################################################################################################
## Computing Brightness functions
## Convert image to greyscale, return average pixel brightness.
def brightness_gray_mean(im):
    im = im.convert('L')
    stat = ImageStat.Stat(im)
    return stat.mean[0]

## Convert image to greyscale, return RMS pixel brightness
def brightness_gray_rms(im):
    im = im.convert('L')
    stat = ImageStat.Stat(im)
    return stat.rms[0]


## Average pixels, then transform to "perceived brightness".
def brightness_original_mean_perceived(im):
    try:
        stat = ImageStat.Stat(im)
        r,g,b = stat.mean
        return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
    except:
        return -1

## RMS of pixels, then transform to "perceived brightness"
def brightness_original_rms_perceived(im):
    try:
        stat = ImageStat.Stat(im)
        r,g,b = stat.rms
        return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
    except:
        return -1

## Calculate "perceived brightness" of pixels, then return average.
def brightness_original_perceived_avg(im):
    try:
        stat = ImageStat.Stat(im)
        gs = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)) 
             for r,g,b in im.getdata())
        return sum(gs)/stat.count[0]
    except:
        return -1


def get_all_brightness(im):
    print('brightness_gray_mean: \t\t',brightness_gray_mean(im))
    print('brightness_gray_rms: \t\t',brightness_gray_rms(im))
    print('brightness_original_mean_perceived: \t\t',brightness_original_mean_perceived(im))
    print('brightness_original_rms_perceived: \t\t',brightness_original_rms_perceived(im))
    print('brightness_original_perceived_avg: \t\t',brightness_original_perceived_avg(im))
################################################################################################################
    
    
    


################################################################################################################
# GENERATING GABORS

def Gabor2D(lambda_, theta, phase, stdev, imSize, elCentre=None, gratingContrast=1.0):
    lambda_ = torch.tensor(lambda_/1.0).float()
    theta = torch.tensor(theta/1.0).float()
    phase = torch.tensor(phase/1.0).float()
    stdev = torch.tensor(stdev/1.0).float()
        
    ##########################################
    # Computing the element center
    if elCentre is None:
        if(isinstance(imSize, (int,float,torch.Tensor))): 
            elCentre = imSize//2
        else:
            elCentre = (imSize[0]//2,imSize[1]//2)
    
    ## Get the centre of the image
    if isinstance(elCentre, (int,float,torch.Tensor)):
        yCentre = elCentre
        xCentre = elCentre
    else:    
        yCentre = elCentre[0]
        xCentre = elCentre[1]
    ##########################################
    
    ##########################################
    # Get the standard deviation of the gaussian
    if isinstance(stdev, (int,float,torch.Tensor)):
        stdevY = stdev
        stdevX = stdev
    else:  
        stdevY = stdev[1]
        stdevX = stdev[2]
    ##########################################
    
    ##########################################
    # setup colors
    # determine increment (to multiply by sine wave and subtract from mean)
    black, gray, white = -1.0, 0.0, 1.0
    inc = min(abs(white-gray),abs(black-gray))*gratingContrast
    ##########################################
    
    angle = theta * math.pi / 180 # orientation deg to radians.
    phase = phase * math.pi / 180 # phase deg to radians.
    sinX = torch.sin(angle)*((2*math.pi)/lambda_)
    cosY = torch.cos(angle)*((2*math.pi)/lambda_)
    
    ##########################################
    # X and Y positions for every element on the patch -> see the coordinate system
    y, x = torch.meshgrid([torch.arange(-xCentre + 1, xCentre + 1),
                           torch.arange(-yCentre + 1, yCentre + 1)])
    ##########################################

    
    sinusoidal= torch.sin(sinX*x+cosY*y+phase)
    gauss= torch.exp(-(x**2)/(2*stdevX**2)-(y**2)/(2*stdevY**2))
           
    gabor = gauss*sinusoidal
    gabor = gabor*inc;
    
    
    return gabor, gauss, sinusoidal
################################################################################################################






################################################################################################################
# GENERATING PATHS

def get_path_start(radius=64):
    '''choose a random point 64 pixels from center'''
    startAngleDeg = random.randint(0, 359)
    startAngleRad = math.radians(startAngleDeg)
    startX = radius * math.sin(startAngleRad)
    startY = radius * math.cos(startAngleRad)
    return startAngleDeg, startAngleRad, startX, startY

def get_next_point():
    pass

def generate_path(numElements=12, gridSize=(16,16), imSize=(512,512),
                  D=32.0, jitterD=.25, B=32, jitterB=10, 
                  startRadius=64, max_attempts=6):
    '''just a wrapper around `_generate_path` to handle retries'''
    found=False
    for attempt in range(max_attempts):
        try:
            points,centers,grid = _generate_path(numElements=numElements, 
                                                 gridSize=gridSize, 
                                                 imSize=imSize,
                                                 D=D, jitterD=jitterD, 
                                                 B=B, jitterB=jitterB, 
                                                 startRadius=startRadius, 
                                                 max_attempts=max_attempts)
        except:
            pass
            if(attempt==max_attempts-1):
                raise Exception('could not generate a path')
                
        else:
            break
    
    return points,centers,grid

def _generate_path(numElements=12, gridSize=(16,16), imSize=(512,512),
                  D=32.0, jitterD=.25, B=32, jitterB=10, 
                  startRadius=64, max_attempts=6):
    '''
        Path Elements -> TODO - Update this using the comments below
        1.  The image was divided into a grid proportional to the desired 
            spacing (default 16,16 = 32x32px)
        2.  A starting point for the path was selected (P1). 
            The starting point of the path was always at a point 64 pixels
            (~1.0 deg) from the center of the image. 
        3.  A vector of distance D was projected from P1 to 
            P2 directed towards the center of the image plus or minus 
            a variable angle (+/- B, default to 32 pixels, ~0.5 deg). 
        4.  An element was placed at the halfway point between Pl and P2 
            providing that the square was not already occupied by an element. 
            If the path already contained element, then the path was extended 
            (D/4) to a square that did not contain an element. By default, 
            the orientation of the element is set to the orientation of the 
            path (alpha = 0), but can be varied (e.g., alpha = 90 deg for 
            elements oriented orthogonal to the path).
        
        Background Elements
        
        Arguments:
            numElements: number of elements!
            D: distance between elements in pixels
            B: angular offset 
            
        Returns:
            points: (x,y,angle) for points along the path
            centers: (x,y,angle) for centers of elements along path (needed for rendering)
            grid: matrix (gridSize) indicating which cells are have elements 
    '''
    
    ##########################################
    # 1. setup grid so we can make sure there's only one element per cell -> this coordinate is different from the 
    # coordinate in steps 2,3,4,5. In this step the origin is in top-right
    # In the remaining steps rhe origin is right in the middle
    imHeight, imWidth = imSize[0], imSize[1]
    grid = np.zeros(gridSize).astype(np.uint8)
    
    gridCellH = imHeight/gridSize[0]
    gridCellW = imWidth/gridSize[1]
    
    gridIndexH = np.arange(0,imHeight) // gridCellH
    gridIndexW = np.arange(0,imWidth) // gridCellW
    ##########################################
    
    
    ##########################################
    # 2. choose P1, a starting point 64 pixels from center -> Y axis of the corrdinate system is flipped and see
    # where the angle rotation starts from
    startAngleDeg, startAngleRad, startX, startY = get_path_start(radius=startRadius)
    ##########################################
    
    ####################################################################################
    
    # 3. vector between P1 + P2 is towards center of image +/- B
    theta = startAngleDeg + 180 + random.choice([-1,1]) *B
    
    # 4. Setup a list for adding the points and their centers. Centers are where we will add the gabors
    # Hence this list contains the coordinate/pixel location for the elements and 
    # the direction we will head to "next" for element 1 but will be direction from where we came for other elements. Hence thgeta for element 1 and 2 is same
    points = [(int(startX),int(startY),theta)]
    centers = []
    ####################################################################################
    
    ####################################################################################
    # 5.
    for el_num in range(numElements):
        isValid = False
        
        # 5a. set distance to next point (D+jitter)
        attempts = 0
        currD = D + random.choice([-1,0,1]) * (D*jitterD)
        
        # 5b. Looping to find all the points and locations
        while not isValid and attempts < max_attempts:  
            attempts += 1
            
            # Getting next point and center
            lastX,lastY,lastTheta = points[-1]
            nextX = int(lastX + currD * math.sin(math.radians(theta)))
            nextY = int(lastY + currD * math.cos(math.radians(theta)))        
        
            dx = nextX - lastX
            dy = nextY - lastY
            centerX = lastX + dx/2
            centerY = lastY + dy/2
            
            # Checking if we are not going outside screen
            imageCenterX = int(imWidth/2+centerX)
            imageCenterY = int(imHeight/2+centerY)
            if imageCenterX < 0 or imageCenterY < 0 or imageCenterX >= imWidth or imageCenterY >= imHeight:
                continue

                        
            # Checking if the grid that owns that location contains an element or not -> location is equivalent to the pixel
            # If the grid does not contain an element yet, awesome, that becomes a place to situate the element
            # If it contains, do not place here, extend the path by 25% (but that then remains fixed for the remaninign elements) and find a grid
            try:
                gridX = int(gridIndexW[imageCenterX])
                gridY = int(gridIndexH[imageCenterY])
            except:
                print('outside frame X: ',imageCenterX)
                print('outside frame Y: ',imageCenterY)
            if grid[gridY,gridX] == 1:
                # cell occupied, extend path segment length towards unoccupied cell
                currD = currD + D/4
            else:
                # found the cell...Yayy!
                grid[gridY,gridX] = 1
                isValid = True
        
        if attempts == max_attempts and not isValid:
            raise Exception("Sorry, no worky")
            
        # update points, centers, angles 
        points.append((nextX,nextY,theta))
        centers.append((centerX, centerY, theta))
        
        # choose new angle, no possibility of straight continuation of path
        step_dir = random.choice([-1,1])
        # offset is B plus jitter uniformly between +/- jitterB
        offset = B + random.randint(-jitterB,jitterB)
        theta = theta + offset * step_dir
  ####################################################################################
    return points, centers, grid
    
    
    
def show_path(points, centers, imHeight=512, imWidth=512):
    '''
    
    '''
    img = np.zeros((imHeight,imWidth,3))
    for i,(pointX,pointY,_) in enumerate(points):
        x,y = int(imWidth/2+pointX), int(imHeight/2+pointY)
        img = cv2.circle(img, (x,y), radius=2, color=(-1, -1, -1), thickness=-1)
        cv2.putText(img,'P'+str(i+1),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(-1,-1,-1),1)
        
    for pointX,pointY,_ in centers:
        x,y = int(imWidth/2+pointX), int(imHeight/2+pointY)
        img = cv2.circle(img, (x,y), radius=6, color=(1, 1, 1), thickness=-1)
        img = cv2.circle(img, (x,y), radius=2, color=(-1, -1, -1), thickness=-1)
        
        
    point_positions=(np.array(points)[:,:2] + [imWidth/2,imHeight/2]).astype(int)
    cv2.polylines(img, 
              [point_positions],isClosed = False,
              color = (-1,-1,-1),
              thickness = 1)
    
    return show_im(img)



def show_path_centercircle(points, centers, startRadius, imHeight=512, imWidth=512):
    img = np.zeros((imHeight,imWidth,3))
    for i,(pointX,pointY,_) in enumerate(points):
        x,y = int(imWidth/2+pointX), int(imHeight/2+pointY)
        img = cv2.circle(img, (x,y), radius=2, color=(-1, -1, -1), thickness=-1)
        cv2.putText(img,'P'+str(i+1),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(-1,-1,-1),1)
        
    for pointX,pointY,_ in centers:
        x,y = int(imWidth/2+pointX), int(imHeight/2+pointY)
        img = cv2.circle(img, (x,y), radius=6, color=(1, 1, 1), thickness=-1)
        img = cv2.circle(img, (x,y), radius=2, color=(-1, -1, -1), thickness=-1)
        
        
    point_positions=(np.array(points)[:,:2] + [imWidth/2,imHeight/2]).astype(int)
    cv2.polylines(img, 
              [point_positions],isClosed = False,
              color = (-1,-1,-1),
              thickness = 1)
    
    
    # Center Circle
    img = cv2.circle(img, (int(imWidth/2),int(imHeight/2)), radius=startRadius, color=(-1, -1, -1), thickness=1)
    img = cv2.circle(img, (int(imWidth/2),int(imHeight/2)), radius=1, color=(-1, -1, -1), thickness=2)
    # Connecting the center to the first point
    cv2.polylines(img, 
              [np.array([[int(imWidth/2),int(imHeight/2)],point_positions[0]])],isClosed = False,
              color = (-1,-1,-1),
              thickness = 1)
    
    return show_im(img)


    
def overlay_grid(points, centers, grid, imHeight=512, imWidth=512):
    '''
    
    '''
    path_img = np.array(show_path(points, centers, imHeight=imHeight, imWidth=imWidth))
    grid_img = np.array(Image.fromarray(grid*255).resize((imHeight,imWidth), Image.NEAREST))
    grid_img[np.where(grid_img==0.0)]=127.0
    path_img[:,:,0] = path_img[:,:,0]*.5 + grid_img*.5
    return blend(Image.fromarray(path_img),Image.fromarray(grid_img).convert('RGB'))
################################################################################################################


## TODO: The gabor parameters in the following functions are fixed but can be varied for subsequent experiments


################################################################################################################
# GENERATING IMAGES
def embed(img, background, startX, startY):
    
    H,W = background.shape
    maxH,maxW = H, W
    h,w = img.shape
    endX = startX + w
    endY = startY + h
    
    bg_startX = max(0,startX)
    bg_startY = max(0,startY)
    bg_endX = min(endX, maxW)
    bg_endY = min(endY, maxH)

    im_startX = 0 if startX > 0 else abs(startX)
    im_startY = 0 if startY > 0 else abs(startY)
    im_endX = w if endX < maxW else w-(endX-maxW)
    im_endY = h if endY < maxH else h-(endY-maxH)
    
    background[bg_startY:bg_endY,bg_startX:bg_endX] += img[im_startY:im_endY,im_startX:im_endX]

    return background    


# Generating the entire path image
def generate_path_image(imWidth=512,imHeight=512,numElements=12, D=32.0,
                        jitterD=.25,B=30, jitterB=10, startRadius=64,
                        randomize=False,background_present=True,alpha_offset=90,gridSize=(16,16),max_attempts=100,
                       gabor_lambda=8,gabor_phase=-90,gabor_stddev=4.0,gabor_imSize=28,gabor_elCentre=None,gabor_gratingContrast=1.0):
    
    points,centers,grid = generate_path(numElements=numElements, D=D, jitterD=jitterD,
                                        B=B, jitterB=jitterB,
                                        startRadius=startRadius,gridSize=gridSize,imSize=(imWidth,imHeight),max_attempts=max_attempts)
    
    # Here zero means the gray scale. 
    image = np.zeros((imWidth,imHeight))
    
    
    gridRows,gridCols = grid.shape
    c = 0
    for row in range(gridRows):
        for col in range(gridCols):
            if grid[row,col]:
                # cell is part of the contour path
                startX,startY,theta = centers[c]
                startX = int(imWidth/2+startX)
                startY = int(imHeight/2+startY)
                alpha = theta - alpha_offset # TODO, IMPORTAMNT- make sure our Gabor draws 0 as vertical...theta here is actually the alpha
                # theta = np.random.rand() * 180 - 90
                
                
                if randomize:
                    alpha = np.random.rand() * 180 - 90
                    
                element,_,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha, phase=gabor_phase, stdev=gabor_stddev, 
                                  imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)
                startX -= (gabor_imSize//2)
                startY -= (gabor_imSize//2)
                #element[:,:] = -1
                c+=1
                
                try:
                    image = embed(element.numpy(), image, startX, startY)
                except:
                    print("oops", startX, startY)
                    pass
                
                
            else:
                # cell is a background element and not part of the path
                if(background_present):
                    alpha = np.random.rand() * 180 - 90
                    element,_,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha, phase=gabor_phase, stdev=gabor_stddev, 
                                  imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)
                    
                    startY = row*32 + 32//2 - (gabor_imSize//2) - np.random.randint(-(gabor_imSize//3), (gabor_imSize//3))
                    startX = col*32 + 32//2 - (gabor_imSize//2) - np.random.randint(-(gabor_imSize//3), (gabor_imSize//3))

                    #startY = row*32 + 32//2 - 28//2 - np.random.randint(-1, 1)
                    #startX = col*32 + 32//2 - 28//2 - np.random.randint(-1, 1)
                    
                    # TODO- Ask GA
                    #element[:,:] *= .35

                    try:
                        image = embed(element.numpy(), image, startX, startY)
                    except:
                        print("oops", startX, startY)
                        pass
                    
                    
                    
    image = np.clip(image, -1,1)
    
    # TODO- Ask GA
    # This iscaling is important to note and is different from the kind of scaling done in show_im and show_gabor functions. Important.
    # The way we had normalized before 0.5 was mapping on to 0
    img = Image.fromarray((image*127.5+127.5).astype(np.uint8))
    
    return img,points,centers,grid





def generate_completerandom_image(imWidth=512,imHeight=512,gridRows=16,gridCols=16,
                                 gabor_lambda=8,gabor_phase=-90,gabor_stddev=4.0,gabor_imSize=28,gabor_elCentre=None,gabor_gratingContrast=1.0):
    image = np.zeros((imWidth,imHeight))
    c = 0
    for row in range(gridRows):
        for col in range(gridCols):
            alpha = np.random.rand() * 180 - 90            
            element,_,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha, phase=gabor_phase, stdev=gabor_stddev, 
                                  imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)
            
            startY = row*32 + 32//2 - (gabor_imSize//2) - np.random.randint(-(gabor_imSize//3),(gabor_imSize//3))
            startX = col*32 + 32//2 - (gabor_imSize//2) - np.random.randint(-(gabor_imSize//3),(gabor_imSize//3))

            try:
                image = embed(element.numpy(), image, startX, startY)
            except:
                print("oops", startX, startY)
                pass
    image = np.clip(image, -1,1)
    img = Image.fromarray((image*127.5+127.5).astype(np.uint8))
    
    return img
################################################################################################################








################################################################################################################
# Generating the image pair -> with contour and control (either get with or without background i.e toggle the background_present argument)

def generate_contour_control_pair_image(imWidth=512,imHeight=512,numElements=12, D=32.0,
                                        jitterD=.25,B=30, jitterB=10, startRadius=64,
                                        background_present=True,alpha_offset=90,gridSize=(16,16),max_attempts=100,
                                        gabor_lambda=8,gabor_phase=-90,gabor_stddev=4.0,gabor_imSize=28,gabor_elCentre=None,gabor_gratingContrast=1.0):
    
    points,centers,grid = generate_path(numElements=numElements, D=D, jitterD=jitterD,
                                        B=B, jitterB=jitterB,
                                        startRadius=startRadius,gridSize=gridSize,imSize=(imWidth,imHeight),max_attempts=max_attempts)
    
    image_contour = np.zeros((imWidth,imHeight))
    image_control = np.zeros((imWidth,imHeight))
    
    
    gridRows,gridCols = grid.shape
    
    c = 0
    for row in range(gridRows):
        for col in range(gridCols):
            if grid[row,col]:
                # cell is part of the contour path
                startX,startY,theta = centers[c]
                startX = int(imWidth/2+startX)
                startY = int(imHeight/2+startY)
                
                
                alpha_contour = theta - alpha_offset # TODO, make sure our Gabor draws 0 as vertical...theta here is actually the alpha
                alpha_control = np.random.rand() * 180 - 90
                
                
                element_contour,_,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha_contour, phase=gabor_phase, stdev=gabor_stddev, 
                                  imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)
                
                element_control,_,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha_control, phase=gabor_phase, stdev=gabor_stddev, 
                                  imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)
                
                
                startX -= (gabor_imSize//2)
                startY -= (gabor_imSize//2)
                #element[:,:] = -1
                c+=1
                
                try:
                    
                    image_contour = embed(element_contour.numpy(), image_contour, startX, startY)
                    image_control = embed(element_control.numpy(), image_control, startX, startY)
                    
                except:
                    print("oops", startX, startY)
                    pass
                
                
            else:
                # cell is a background element and not part of the path
                if(background_present):
                    alpha = np.random.rand() * 180 - 90
                    
                    
                    element,_,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha, phase=gabor_phase, stdev=gabor_stddev, 
                                  imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)
                    
                    startY = row*32 + 32//2 - (gabor_imSize//2) - np.random.randint(-(gabor_imSize//3),(gabor_imSize//3))
                    startX = col*32 + 32//2 - (gabor_imSize//2) - np.random.randint(-(gabor_imSize//3),(gabor_imSize//3))

                    #startY = row*32 + 32//2 - 28//2 - np.random.randint(-1, 1)
                    #startX = col*32 + 32//2 - 28//2 - np.random.randint(-1, 1)

                    #element[:,:] *= .35

                    try:
                        image_contour = embed(element.numpy(), image_contour, startX, startY)
                        image_control = embed(element.numpy(), image_control, startX, startY)
                        
                    except:
                        print("oops", startX, startY)
                        pass
                    
                    
                    
    image_contour = np.clip(image_contour, -1,1)
    img_contour = Image.fromarray((image_contour*127.5+127.5).astype(np.uint8))
    
    
    image_control = np.clip(image_control, -1,1)
    img_control = Image.fromarray((image_control*127.5+127.5).astype(np.uint8))
    
    return img_contour,img_control,points,centers,grid
################################################################################################################








################################################################################################################
# Generating the image quadruplet -> contour,control, contour-background, control-background
def generate_everything(imWidth=512,imHeight=512,numElements=12, D=32.0,
                        jitterD=.25,B=30, jitterB=10, startRadius=64,
                        alpha_offset=90,gridSize=(16,16),max_attempts=100,
                        gabor_lambda=8,gabor_phase=-90,gabor_stddev=4.0,gabor_imSize=28,gabor_elCentre=None,gabor_gratingContrast=1.0):
    
    points,centers,grid = generate_path(numElements=numElements, D=D, jitterD=jitterD,
                                        B=B, jitterB=jitterB,
                                        startRadius=startRadius,gridSize=gridSize,imSize=(imWidth,imHeight),max_attempts=max_attempts)
    
    image_contour = np.zeros((imWidth,imHeight))
    image_control = np.zeros((imWidth,imHeight))
    
    image_contour_background=np.zeros((imWidth,imHeight))
    image_control_background=np.zeros((imWidth,imHeight))
    
    ########################################################
    # IMAGE RECORDER - SUPER IMPORTANT TO RENDER THE IMAGE AGAIN
    # This might be updated later based on the information we want
    image_recorder_dict={
        
        # Basic image parameters
        'image_width':imWidth,'image_height':imHeight,
        
        # Grid and path parameters
        'path_B':B,'path_D':D,'path_A':90-alpha_offset,'path_numElement':numElements,
        'path_jitterB':jitterB,'path_jitterD':jitterD,
        'path_startRadius':startRadius,
        'path_offset':alpha_offset,
        'grid':grid,
        'path_points':np.array(points),
        'path_centers':np.array(centers),
        
        # Most important parameters - position, and alpha values
        'element_position':[],
        'element_theta_contour':[],'element_theta_control':[],
        
        # Gabor parameters
        'gabor_lambda':gabor_lambda,'gabor_phase':gabor_phase,'gabor_stdev':gabor_stddev,
        'gabor_imSize':gabor_imSize,'gabor_elCentre':gabor_elCentre,'gabor_gratingContrast':gabor_gratingContrast}
    ########################################################
    
    
    gridRows,gridCols = grid.shape
    
    c = 0
    for row in range(gridRows):
        for col in range(gridCols):
            if grid[row,col]:
                # cell is part of the contour path
                startX,startY,theta = centers[c]
                startX = int(imWidth/2+startX)
                startY = int(imHeight/2+startY)
                
                
                alpha_contour = theta - alpha_offset # TODO, make sure our Gabor draws 0 as vertical...theta here is actually the alpha
                alpha_control = np.random.rand() * 180 - 90
                
                
                
                element_contour,_,_ = Gabor2D(lambda_=image_recorder_dict['gabor_lambda'], theta=alpha_contour, phase=image_recorder_dict['gabor_phase'], 
                                              stdev=image_recorder_dict['gabor_stdev'], imSize=image_recorder_dict['gabor_imSize'], elCentre=image_recorder_dict['gabor_elCentre'], 
                                              gratingContrast=image_recorder_dict['gabor_gratingContrast'])
                
                element_control,_,_ = Gabor2D(lambda_=image_recorder_dict['gabor_lambda'], theta=alpha_control, phase=image_recorder_dict['gabor_phase'], 
                                              stdev=image_recorder_dict['gabor_stdev'], imSize=image_recorder_dict['gabor_imSize'], elCentre=image_recorder_dict['gabor_elCentre'], 
                                              gratingContrast=image_recorder_dict['gabor_gratingContrast'])
                
                
                startX -= (image_recorder_dict['gabor_imSize']//2)
                startY -= (image_recorder_dict['gabor_imSize']//2)
                #element[:,:] = -1
                c+=1
                
                try:
                    
                    image_contour = embed(element_contour.numpy(), image_contour, startX, startY)
                    image_control = embed(element_control.numpy(), image_control, startX, startY)
                    
                    image_contour_background = embed(element_contour.numpy(), image_contour_background, startX, startY)
                    image_control_background = embed(element_control.numpy(), image_control_background, startX, startY)
                    
                    ########################################################
                    # Add in IMAGE RECORDER
                    image_recorder_dict['element_position'].append([startX,startY])
                    image_recorder_dict['element_theta_contour'].append(alpha_contour)
                    image_recorder_dict['element_theta_control'].append(alpha_control)
                    ########################################################
                    
                    
                except:
                    print("oops", startX, startY)
                    pass
                
                
            else:
                # cell is a background element and not part of the path
   
                alpha = np.random.rand() * 180 - 90
    
    
    
                element,_,_ = Gabor2D(lambda_=image_recorder_dict['gabor_lambda'], theta=alpha, phase=image_recorder_dict['gabor_phase'], 
                                      stdev=image_recorder_dict['gabor_stdev'], imSize=image_recorder_dict['gabor_imSize'], elCentre=image_recorder_dict['gabor_elCentre'], 
                                      gratingContrast=image_recorder_dict['gabor_gratingContrast'])
        
                startY = row*32 + 32//2 - (image_recorder_dict['gabor_imSize']//2) - np.random.randint(-(image_recorder_dict['gabor_imSize']//3),(image_recorder_dict['gabor_imSize']//3))
                startX = col*32 + 32//2 - (image_recorder_dict['gabor_imSize']//2) - np.random.randint(-(image_recorder_dict['gabor_imSize']//3),(image_recorder_dict['gabor_imSize']//3))

                #startY = row*32 + 32//2 - 28//2 - np.random.randint(-1, 1)
                #startX = col*32 + 32//2 - 28//2 - np.random.randint(-1, 1)

                #element[:,:] *= .35

                try:
                    image_contour_background = embed(element.numpy(), image_contour_background, startX, startY)
                    image_control_background = embed(element.numpy(), image_control_background, startX, startY)
                    
                    ########################################################
                    # Add in IMAGE RECORDER
                    image_recorder_dict['element_position'].append([startX,startY])
                    image_recorder_dict['element_theta_contour'].append(alpha)
                    image_recorder_dict['element_theta_control'].append(alpha)
                    ########################################################

                except:
                    print("oops", startX, startY)
                    pass
                    
                    
                    
    image_contour = np.clip(image_contour, -1,1)
    img_contour = Image.fromarray((image_contour*127.5+127.5).astype(np.uint8))
    
    
    image_control = np.clip(image_control, -1,1)
    img_control = Image.fromarray((image_control*127.5+127.5).astype(np.uint8))
    
    image_contour_background = np.clip(image_contour_background, -1,1)
    img_contour_background = Image.fromarray((image_contour_background*127.5+127.5).astype(np.uint8))
    
    image_control_background = np.clip(image_control_background, -1,1)
    img_control_background = Image.fromarray((image_control_background*127.5+127.5).astype(np.uint8))
    
    ########################################################
    # Update the IMAGE RECORDER for consistent shapes - element gabor positions, gabor alpha values for contour, and gabor alpha values for control reshaped in the format/shape of the grid
    image_recorder_dict['element_position']=np.reshape(np.array(image_recorder_dict['element_position']),(image_recorder_dict['grid'].shape[0],image_recorder_dict['grid'].shape[1],-1))
    image_recorder_dict['element_theta_contour']=np.reshape(np.array(image_recorder_dict['element_theta_contour']),(image_recorder_dict['grid'].shape[0],image_recorder_dict['grid'].shape[1],-1))
    image_recorder_dict['element_theta_control']=np.reshape(np.array(image_recorder_dict['element_theta_control']),(image_recorder_dict['grid'].shape[0],image_recorder_dict['grid'].shape[1],-1))
    ########################################################
    
    
    return img_contour,img_control,img_contour_background,img_control_background,points,centers,grid,image_recorder_dict
################################################################################################################









##########################################################################################################################
# RENDERER
def image_renderer(image_recorder_dict):
    ## Extracting what all you need from the image recorder
    
    # Image properties
    gridRows,gridCols = image_recorder_dict['grid'].shape
    imWidth=image_recorder_dict['image_width']
    imHeight=image_recorder_dict['image_height']
    
    # Gabor properties
    gabor_lambda=image_recorder_dict['gabor_lambda']
    gabor_phase=image_recorder_dict['gabor_phase']
    gabor_stdev=image_recorder_dict['gabor_stdev']
    gabor_imSize=image_recorder_dict['gabor_imSize']
    gabor_elCentre=image_recorder_dict['gabor_elCentre']
    gabor_gratingContrast=image_recorder_dict['gabor_gratingContrast']


    ## Initialize the image
    image_contour = np.zeros((imWidth,imHeight))
    image_control = np.zeros((imWidth,imHeight))

    image_contour_background=np.zeros((imWidth,imHeight))
    image_control_background=np.zeros((imWidth,imHeight))



    c = 0
    for row in range(gridRows):
        for col in range(gridCols):
            if image_recorder_dict['grid'][row,col]:

                alpha_contour = image_recorder_dict['element_theta_contour'][row][col]
                alpha_control = image_recorder_dict['element_theta_control'][row][col]
                startX = image_recorder_dict['element_position'][row][col][0]
                startY = image_recorder_dict['element_position'][row][col][1]


                element_contour,_,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha_contour, phase=gabor_phase, stdev=gabor_stdev, 
                                      imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)

                element_control,_,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha_control, phase=gabor_phase, stdev=gabor_stdev, 
                                      imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)

                try:
                    image_contour = embed(element_contour.numpy(), image_contour, startX, startY)
                    image_control = embed(element_control.numpy(), image_control, startX, startY)

                    image_contour_background = embed(element_contour.numpy(), image_contour_background, startX, startY)
                    image_control_background = embed(element_control.numpy(), image_control_background, startX, startY)
                except:
                    print("oops", startX, startY)

            else:

                assert image_recorder_dict['element_theta_contour'][row][col] == image_recorder_dict['element_theta_contour'][row][col]
                alpha = image_recorder_dict['element_theta_contour'][row][col]
                startX = image_recorder_dict['element_position'][row][col][0]
                startY = image_recorder_dict['element_position'][row][col][1]

                element,_,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha, phase=gabor_phase, stdev=gabor_stdev, 
                                      imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)

                try:
                    image_contour_background = embed(element.numpy(), image_contour_background, startX, startY)
                    image_control_background = embed(element.numpy(), image_control_background, startX, startY)
                except:
                    print("oops", startX, startY)



    image_contour = np.clip(image_contour, -1,1)
    img_contour = Image.fromarray((image_contour*127.5+127.5).astype(np.uint8))


    image_control = np.clip(image_control, -1,1)
    img_control = Image.fromarray((image_control*127.5+127.5).astype(np.uint8))

    image_contour_background = np.clip(image_contour_background, -1,1)
    img_contour_background = Image.fromarray((image_contour_background*127.5+127.5).astype(np.uint8))

    image_control_background = np.clip(image_control_background, -1,1)
    img_control_background = Image.fromarray((image_control_background*127.5+127.5).astype(np.uint8))
    
    return img_contour, img_control, img_contour_background, img_control_background
##########################################################################################################################



    
##########################################################################################################################
## new renderer to give the images with gaussians instead of gabors
def image_renderer_gaussian(image_recorder_dict):
    
    list_gauss=[]
    
    ## Extracting what all you need from the image recorder
    gridRows,gridCols = image_recorder_dict['grid'].shape
    imWidth=image_recorder_dict['image_width']
    imHeight=image_recorder_dict['image_height']

    gabor_lambda=image_recorder_dict['gabor_lambda']
    gabor_phase=image_recorder_dict['gabor_phase']
    gabor_stdev=image_recorder_dict['gabor_stdev']
    gabor_imSize=image_recorder_dict['gabor_imSize']
    gabor_elCentre=image_recorder_dict['gabor_elCentre']
    gabor_gratingContrast=image_recorder_dict['gabor_gratingContrast']

    
    
    image_path = np.zeros((imWidth,imHeight))
    image_background = np.zeros((imWidth,imHeight))



    c = 0
    for row in range(gridRows):
        for col in range(gridCols):
            if image_recorder_dict['grid'][row,col]:

                alpha_contour = image_recorder_dict['element_theta_contour'][row][col]
                alpha_control = image_recorder_dict['element_theta_control'][row][col]
                startX = image_recorder_dict['element_position'][row][col][0]
                startY = image_recorder_dict['element_position'][row][col][1]


                _,element_contour_gauss,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha_contour, phase=gabor_phase, stdev=gabor_stdev, 
                                      imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)

                _,element_control_gauss,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha_control, phase=gabor_phase, stdev=gabor_stdev, 
                                      imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)

                
                assert(torch.equal(element_contour_gauss, element_control_gauss))
                list_gauss.append(element_contour_gauss)
                
                try:
                    
                    image_path_fg = embed(element_contour_gauss.numpy(), image_path, startX, startY)
                    
                except:
                    print("oops", startX, startY)

            else:

                assert image_recorder_dict['element_theta_contour'][row][col] == image_recorder_dict['element_theta_contour'][row][col]
                alpha = image_recorder_dict['element_theta_contour'][row][col]
                startX = image_recorder_dict['element_position'][row][col][0]
                startY = image_recorder_dict['element_position'][row][col][1]

                _,element_gauss,_ = Gabor2D(lambda_=gabor_lambda, theta=alpha, phase=gabor_phase, stdev=gabor_stdev, 
                                      imSize=gabor_imSize, elCentre=gabor_elCentre, gratingContrast=gabor_gratingContrast)

                try:
                    
                    image_path_bg = embed(element_gauss.numpy(), image_background, startX, startY)
                    
                except:
                    print("oops", startX, startY)




    
    
    image_path_fg = np.clip(image_path_fg, -1,1)
    image_path_fg = Image.fromarray((image_path_fg*127.5+127.5).astype(np.uint8))
    
    
    image_path_bg = np.clip(image_path_bg, -1,1)
    image_path_bg = Image.fromarray((image_path_bg*127.5+127.5).astype(np.uint8))

    
    
    return image_path_fg, image_path_bg, list_gauss




## Get the saliency masks
def mask_renderer(image_recorder_dict, thresholded_value=200):
    # img_contour, img_control, img_contour_background, img_control_background, image_path_fg, image_path_bg, list_gauss = image_renderer_gaussian_mask(image_recorder_dict)
    image_path_fg, image_path_bg, list_gauss = image_renderer_gaussian(image_recorder_dict)
    
    ## path-fg elements
    mask_img_path_fg=np.zeros_like(image_path_fg)
    mask_img_path_fg[np.where(np.array(image_path_fg) > thresholded_value)[0],np.where(np.array(image_path_fg) > thresholded_value)[1]]=255
    
    ## path-bg elements
    mask_img_path_bg=np.zeros_like(image_path_bg)
    mask_img_path_bg[np.where(np.array(image_path_bg) > thresholded_value)[0],np.where(np.array(image_path_bg) > thresholded_value)[1]]=255
    
    
    
    return Image.fromarray(mask_img_path_fg.astype(np.uint8)),Image.fromarray(mask_img_path_bg.astype(np.uint8)),list_gauss

##########################################################################################################################


    