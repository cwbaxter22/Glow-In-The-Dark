"""Mainipulate pixel intensity arrays from images

Module contains X functions:
1) Image averaging function

"""
import pathlib
import os
import numpy as np
from PIL import Image

def collect_images(folder_of_img: str):
    """
    Collect images from directory where images are stored and put them in an array as np arrays.

    Arguments:
    ----------
    folder_of_img (str): folder directory images will be pulled from
    
    Returns:
    ----------
    list_images (list of np arrays): list of images from passed directory 
    
    """
    repo_path = pathlib.Path(__file__).parents[1]
    allfiles=os.listdir(os.path.join(repo_path, r"images", folder_of_img))
    list_image_names = [filename for filename in allfiles if filename[-4:] in [".jpg",".JPG"]]
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
    img_averaged (np array): one image that is a matrix avg of the input images 
    
    """
    #https://stackoverflow.com/questions/17291455
    #/how-to-get-an-average-picture-from-100-pictures-using-pil

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

def img_crop(list_imgs: list, yMinC: int, yMaxC: int, xMinC: int, xMaxC: int):
    """
    Take a list of images and crop them to given dimensions based on vertex points

    Arguments:
    ----------
    list_imgs (list np): list containing all image arrays to be cropped
    
    Returns:
    ----------
    list_cropped (list np): list of images cropped to dimenions specified 
    
    """
    list_cropped = [im[yMinC:yMaxC, xMinC:xMaxC] for im in list_imgs]
    return list_cropped

def flat_field_correction(raw: np, flat: np, dark: np):
    """
    Apply Flat Field correction to test images

    Arguments:
    ----------
    raw (np): np from raw image
    flat (np): np from image taken of white piece of paper
    dark (np): np from image taken with lens cap on
    
    Returns:
    ----------
    corrected_matrix (np): np with correction applied  
    
    """
    FmD = flat - dark
    m = np.average(FmD)
    return ((raw - dark)*(m/FmD))