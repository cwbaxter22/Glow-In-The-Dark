"""Mainipulate pixel intensity arrays from images

Module contains X functions:
1) Image averaging function

"""
import pathlib
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
                                                       r"images",
                                                       folder_of_img,
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

def flat_field_correction(data_image, ambient_image, dark_current_image, flat_field_image):
    # Ensure all images have the same dimensions
    assert data_image.shape == ambient_image.shape == dark_current_image.shape == flat_field_image.shape

    # Convert images to float for accurate calculations
    data_image = data_image.astype(np.float32)
    ambient_image = ambient_image.astype(np.float32)
    dark_current_image = dark_current_image.astype(np.float32)
    flat_field_image = flat_field_image.astype(np.float32)

    # Correct ambient and flat field images for dark current (camera noise)
    corrected_ambient = ambient_image - dark_current_image
    corrected_flat_field = flat_field_image - dark_current_image

    # Normalize the flat field image
    normalized_flat_field = corrected_flat_field / np.mean(corrected_flat_field)

    # Correct the image (subtract dark current and ambient, then divide by normalized flat field)
    corrected_image = (data_image - dark_current_image - corrected_ambient) / normalized_flat_field

    # Clip values to the valid range for an image (0-255) and convert back to uint8
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    data_image = np.clip(data_image, 0, 255).astype(np.uint8)
    #plt.imshow(corrected_image)
    #plt.imshow(data_image)
    return corrected_image
    

def plot_heatmap(wind_on_image, wind_off_image, slope):
    # Multiply each value of corrected_image by inv_slope
    #heatmap_image = wind_on_image*(np.divide(wind_off_pd_int, wind_off_image))*torr_over_wind_off
    heatmap_image = (wind_off_image/wind_on_image)*slope

    # Plot a heatmap of the resulting image
    plt.imshow(heatmap_image, cmap='hot')
    plt.colorbar(label='Pressure (Torr)')  # Add a colorbar with label
    plt.show()  # Display the plot
