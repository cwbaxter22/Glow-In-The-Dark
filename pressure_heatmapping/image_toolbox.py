"""Mainipulate pixel intensity arrays from images

Module contains X functions:
1) Image averaging function

"""
import pathlib
import os
import numpy as np
from PIL import Image

def collect_images(folder_of_img):
    """
    Collect images from directory where images are stored and put them in an array.

    Arguments:
    ----------
    folder_of_img (str): folder images will be pulled from
    
    Returns:
    ----------
    list_images (list of np arrays): Every image from the chosen directory 
    
    """
    repo_path = pathlib.Path(__file__).parents[1]
    allfiles=os.listdir(os.path.join(repo_path, r"images", folder_of_img))
    list_image_names = [filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]
    #list_of_images = np.array(Image.open(list_image_names),dtype=float)
    list_of_images = [np.array(Image.open(os.path.join(repo_path,
                                                       r"images/heatmap_speed",
                                                       img))) for img in list_image_names]
    return list_of_images

def img_avg(list_imgs: list):
    """
    Generate a single image where each pixel value is the average of corresponding image pixels.

    Arguments:
    ----------
    list_imgs (list): list containing all image arrays to be averaged
    
    Returns:
    ----------
    img_averaged (img): 
    
    """
    #https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil

    #All pictures are the same size, so grab the dimenions of the first image
    first_image = Image.fromarray(list_imgs[0])
    w,h=first_image.size
    N=len(list_imgs)

    #Initialize an array to hold the average, 3rd index of zeros is 1 because these are CCD
    arr=np.zeros((h,w),float)

    #Add each image to an array and average them
    for im_from_list in list_imgs:
        imarr = np.array(im_from_list)
        arr=arr+imarr/N

    #Round the value of each float and cast it to 8-bit integer
    images_averaged = np.array(np.round(arr),dtype=np.uint8)
    return images_averaged