"""
--------------------------------------------------
Auther:                           Sytwu(111550159)
Lastest Edit:                     2024/05/21 13:30
Functions:
read                                          Done
norm                                          Done
show                                          Done
show_imgs                                     Done
save                                          Done
save_imgs                                     Done
resize                                        Done
gray                                          Done
low_pass                                      Done
high_pass                                     Done
edge                                          Done
Hist_equal                                    Done
auto_crop                                     Done
plot_heatmap                   (Haven't test) Done

requirement.txt                               Done
README.md                                  Haven't

--------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import cv2
import os
import skimage as sk
import skimage.io as skio
from scipy import signal
from PIL import Image

def read(path):
    """ read image from path
    
    Args:
        path (str)
    
    Return:
        img (np.ndarray(np.float64))
    """
    img = sk.img_as_float(skio.imread(path))
    return img

def norm(img):
    """ normalize an image

    Args:
        img (np.ndarray(np.float64))
    
    Return:
        norm_img (np.ndarray(np.float64))
    """
    norm_img = img
    norm_img = norm_img - np.min(norm_img)
    norm_img = norm_img / np.max(norm_img)
    return norm_img

def show(img,title='',is_norm=False):
    """ read image from path
    
    Args:
        img (np.ndarray(np.float64))
        title (str)
        is_norm (bool)
    """
    plt.axis('off')
    if title!='':
        plt.title(title)
    if is_norm:
        img = norm(img)
    if len(img.shape)==2:
        plt.imshow(img, cmap='gray')
    if len(img.shape)==3:
        plt.imshow(img)
    plt.show()
    
def show_imgs(imgs,titles=[],is_norm=False):
    """ read images from path
    
    Args:
        img (list of np.ndarray(np.float64)s)
        title (list of strs)
        is_norm (bool)
    """
    N = len(imgs)
    h, w = imgs[0].shape[:2]
    figsize = (w/20, h/20)
    plt.figure(figsize=figsize)
    plt.axis('off')
    for i,img in enumerate(imgs):
        plt.subplot(1, N, i+1)
        if len(titles)==N:
            plt.title(titles[i])
        if is_norm:
            img = norm(img)
        if len(img.shape)==2:
            plt.imshow(img, cmap='gray')
        if len(img.shape)==3:
            plt.imshow(img)
    plt.show()
    
def save(img,folder='../result',filename='temp_image.jpg',is_norm=False):
    """ save image in path

    Args:
        img (np.ndarray(np.float64))
        folder (str)
        filename (str)
        is_norm (bool)
    """
    if not os.path.exists:
        os.mkdir(folder)
    
    path = folder+'/'+ filename
    if is_norm:
        img = norm(img)
    skio.imsave(path,img)

def save_imgs(imgs,folders='../result',filenames='temp_image.jpg',is_norm=False):
    """ save image in path

    Args:
        img (list of np.ndarray(np.float64)s)
        folder (list of strs)
        filename (list of strs)
        is_norm (bool)
    """
    for img,folder,filename in zip(imgs,folders,filenames):
        save(img,folder,filename,is_norm)
        
def resize(img,h=500,w=500):
    """ resize image with a brutal method

    Args:
        img (np.ndarray(np.float64))
    
    Return:
        resized_img(np.ndarray(np.float64))
    """
    resized_img = sktr.resize(img,(h,w))
    return resized_img

def gray(img):
    """ compute gray image

    Args:
        img (np.ndarray(np.float64))
    
    Return:
        gray_img(np.ndarray(np.float64))
    """
    if len(img.shape)!=2:
        gray_img = np.mean(img,axis=2)
    return gray_img

def low_pass(img,ksize=21,sigma=10):
    """ compute low-pass image with the Gaussian Filter

    Args:
        img (np.ndarray(np.float64))
    
    Return:
        blur_img(np.ndarray(np.float64))
    """
    blur_img = cv2.GaussianBlur(img,(ksize,ksize),sigma)
    return blur_img

def high_pass(img,ksize=21,sigma=10):
    """ compute low-pass image (image = high-pass + low-pass)

    Args:
        img (np.ndarray(np.float64))
    
    Return:
        gray_img(np.ndarray(np.float64))
    """
    low_pass_img  = low_pass(img,ksize,sigma)
    high_pass_img = img - low_pass_img
    return high_pass_img

def edge(img,ksize=21,sigma=10,lt=10,ht=50):
    """ compute an image after canny edge detection

    Args:
        img (np.ndarray(np.float64))
    
    Return:
        edge_img(np.ndarray(np.float64))
    """
    gray_img = gray(img)
    blur_img = low_pass(gray_img,ksize,sigma)
    if blur_img.dtype == np.float64:
        blur_img = (blur_img * 255).astype(np.uint8)
    edge_img = cv2.Canny(blur_img,lt,ht)
    edge_img = sk.img_as_float(edge_img)
    return edge_img

def Hist_equal(img):
    """ compute an image after histogram equalization

    Args:
        img (np.ndarray(np.float64))
    
    Return:
        HE_img(np.ndarray(np.float64))
    """
    gray_img = gray(img)
    if gray_img.dtype == np.float64:
        gray_img = (gray_img * 255).astype(np.uint8)
    HE_img = cv2.equalizeHist(gray_img)
    HE_img = sk.img_as_float(HE_img)
    return HE_img

def auto_crop(img,h=500,w=500):
    """ resize image and crop the most important part

    Args:
        img (np.ndarray(np.float64))
    
    Return:
        auto_crop_img(np.ndarray(np.float64))
    """
    hi, wi = img.shape[:2]
    ratio = h/w
    ratioi = hi/wi
    
    h_new, w_new = h, w
    if ratioi > ratio:
        h_new = int(w_new*ratioi)
    else:
        w_new = int(h_new/ratioi)
    resized_img = sktr.resize(img,(h_new,w_new))
    
    edge_img = edge(resized_img)
    filter1 = np.ones((h,w))
    importance = signal.convolve2d(edge_img,filter1,mode='valid')
    h0,w0 = np.unravel_index(np.argmax(importance,axis=None),importance.shape)
    auto_crop_img = resized_img[h0:h0+h,w0:w0+w]
    return auto_crop_img

def plot_heatmap(heatmap,img,title=''):
    """ show the heatmap and original image

    Args:
        heatmap (np.ndarray(np.float64))
        img (np.ndarray(np.float64))
        title (str)
    
    Return:
        auto_crop_img(np.ndarray(np.float64))
    """
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = sktr.resize(heatmap, (img.shape[1], img.shape[0]))
    
    fig, ax = plt.subplots()
    plt.axis('off')
    if title!='':
        plt.title(title)
    ax.imshow(img, alpha=0.6)
    ax.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.show()

if __name__ == '__main__':
    path = 'test.jpg'
    img = read(path)
    img1 = resize(img)
    img2 = gray(img)
    img3 = low_pass(img)
    img4 = high_pass(img)
    img5 = edge(img)
    img6 = Hist_equal(img)
    img7 = auto_crop(img)
    
    show(img)
    show(img1)
    show(img2)
    show(img3)
    show(img4,is_norm=True)
    show(img5)
    show(img6)
    show(img7)