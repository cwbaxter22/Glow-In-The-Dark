"""
Simple module designed to allow for quick plot creation of samples.
Please remove any sample names before each push to avoid sharing research data.
Data itself is filtered out in the .gitignore/ should not be git added

To run in the shell use the command:
python -c 'import sv_creation;
sv_creation.plot_creation(["sample_1", "sample_2", "sample_3"], lowerb, upperb)'
"""

import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.linalg import cholesky
from pressure_heatmapping import image_toolbox
from scipy import stats

# https://stackoverflow.com/questions/66039235/how-to-subtract-baseline-from-spectrum-with-rising-tail-in-python
# Asymmetrically reweighted penalized least squares smoothing 
def arpls(y, lam=1e4, ratio=0.05, itermax=100):
    r"""
    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)
    """
    N = len(y)
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]
    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z

def create_sv(folder_of_wl, LOWER_B, UPPER_B, WL_TO_ANALYZE):
    repo_path = pathlib.Path(__file__).parents[1]
    #all_txt_files = os.listdir(os.path.join(repo_path, folder_of_wl))
    all_txt_files = os.listdir(os.path.join(repo_path, r"photo_diode_data", folder_of_wl))
    ox_conc = np.zeros(len(all_txt_files))
    peak_intensities = np.zeros(len(all_txt_files))

    for pos, file in enumerate(all_txt_files):
        file_number_str = str(file.replace('.txt', '', 1))
        ox_conc[pos] = float(file_number_str.replace('_', '.', 1)) * (20.9 / 760)

        #txt_path = os.path.join(repo_path, folder_of_wl, file)
        txt_path = os.path.join(repo_path, r"photo_diode_data", folder_of_wl, file)
        df = pd.read_csv(txt_path, header=None, sep='\t', skiprows=17)
        df.drop(df.tail(1).index, inplace=True)
        df = df.apply(pd.to_numeric)
        # Name the two columns wavelength and intensity
        df.columns = ['wavelength', 'intensity']
        # Grab all the intensity values between the lower and upper bounds
        unfiltered_int = df.loc[(df['wavelength'] > LOWER_B) & (df['wavelength'] < UPPER_B), 'intensity']
        # Apply a filter to 'smooth out' the trace of the values
        filtered_int = np.asarray(uniform_filter1d(unfiltered_int, size=20))
        # Grab the corresponding wavelength values between the lower and upper bounds
        wavelength_section = df.loc[(df['wavelength'] > LOWER_B) & (df['wavelength'] < UPPER_B), 'wavelength']
        # Find the background of the filtered intensity values (aka the noise floor)
        bg_noise = arpls(filtered_int, 1E6, 0.001)
        # Subtract the background noise from the filtered intensity values
        all_int_bg_removed = filtered_int - bg_noise
        # Find the index of the wavelength of interest (most likely the highest intensity)
        # Since the wavelength values are not integers, we cannot use a simple index (WL_TO_ANALYZE - LOWER_B could be a decimal)
        # Instead, we find the index of the wavelength that is closest to the wavelength of interest
        desired_wl_index = np.abs(wavelength_section - WL_TO_ANALYZE).argmin()
        desired_wl_int_bg_removed = all_int_bg_removed[desired_wl_index]
        peak_intensities[pos] = desired_wl_int_bg_removed

    ox_conc = np.array(ox_conc)
    peak_intensities = np.array(peak_intensities)
    ox_conc, peak_intensities = zip(*sorted(zip(ox_conc, peak_intensities)))
    #ratios = areas_under_curve[0] / areas_under_curve
    return ox_conc, peak_intensities

def plot_creation(samples, lower_bound, upper_bound, wl_to_analyze):
    plt.figure()
    for sample_num in samples:
        ox_conc_out, ratios_out = create_sv(sample_num, lower_bound, upper_bound, wl_to_analyze)
        plt.scatter(ox_conc_out, ratios_out, label=sample_num)

    plt.xlabel('Oxygen Concentration (%)')
    plt.ylabel('Intensity (I0/I)')
    plt.legend()
    plt.show()

def plot_background_subtraction(samples, LOWER_B, UPPER_B):
    for folder_of_wl in samples:
        repo_path = pathlib.Path(__file__).parents[1]
        all_txt_files = os.listdir(os.path.join(repo_path, r"photo_diode_data", folder_of_wl))
        plt.figure()
        plt.title(f'{folder_of_wl}')
        plt.xlabel('Wavelength')
        plt.ylabel('Intensity')

        for pos, file in enumerate(all_txt_files):
            file_number_str = str(file.replace('.txt', '', 1))

            txt_path = os.path.join(repo_path, r"photo_diode_data", folder_of_wl, file)
            df = pd.read_csv(txt_path, header=None, sep='\t', skiprows=17)
            df.drop(df.tail(1).index, inplace=True)
            df = df.apply(pd.to_numeric)
            df.columns = ['wavelength', 'intensity']
            unfiltered_int = df.loc[(df['wavelength'] > LOWER_B) & (df['wavelength'] < UPPER_B), 'intensity']
            filtered_int = np.asarray(uniform_filter1d(unfiltered_int, size=20))
            wavelength_section = df.loc[(df['wavelength'] > LOWER_B) & (df['wavelength'] < UPPER_B), 'wavelength']
            bg_noise = arpls(filtered_int, 1E6, 0.001)
            int_bg_removed = filtered_int - bg_noise
            plt.plot(wavelength_section, bg_noise, label = f'{file_number_str} bg noise')
            plt.plot(wavelength_section, filtered_int, label = f'{file_number_str} filtered int')
            plt.plot(wavelength_section, int_bg_removed, label = f'{file_number_str} int - bg')
            if pos == 1:
                plt.legend()
                plt.show()
                break

def convert_to_pressure(ox_conc, intensities, WO_mat):
    ox_conc = np.array(ox_conc)
    intensities = np.array(intensities)

    # Convert oxygen concentrations to pressure in Torr
    pressure = ox_conc * 760 / 20.9
    
    # Normalize intensities by the smallest intensity
    normalized_intensities = min(intensities)/intensities
    
    # Generate plot of Torr vs normalized intensity
    #plt.figure()
    #plt.plot(pressure, normalized_intensities, label = 'pressure')
    #plt.xlabel('Pressure (Torr)')
    #plt.ylabel('Normalized Intensity')
    #plt.show()
    #I0_over_I = np.zeros((len(intensities), *np.shape(WO_mat)))
    first_dim = WO_mat.shape[0] 
    second_dim = WO_mat.shape[1]
    I0_over_I = np.zeros((first_dim, second_dim, len(intensities)))
    for pos, i in enumerate(intensities):
        I0_over_I[:, :, pos] = np.divide(WO_mat, i)
        
    # Calculate the slope of the line
    largest_index = np.argmax(intensities)
    smallest_index = np.argmin(intensities)

    for i in range(len(intensities)):
        print(np.sum(I0_over_I[:, :, i])) 
        
    slope_mat = (max(pressure) - min(pressure)) / (I0_over_I[:, :, smallest_index] - I0_over_I[:, :, largest_index])
    
    return slope_mat

def generate_SV_from_camera_data(yMinC, yMaxC, xMinC, xMaxC):
    # Collect raw images
    raw_amb_imgs = image_toolbox.collect_images('camera_sv_images/ambient')
    raw_dark_imgs = image_toolbox.collect_images('camera_sv_images/dark_noise')
    raw_flat_imgs = image_toolbox.collect_images('camera_sv_images/flat_field')
    raw_WO_imgs = image_toolbox.collect_images('camera_sv_images/wind_off')

    # Crop images to area that contains the sample
    c_amb_imgs = image_toolbox.img_crop(raw_amb_imgs, yMinC, yMaxC, xMinC, xMaxC)
    c_dark_imgs = image_toolbox.img_crop(raw_dark_imgs, yMinC, yMaxC, xMinC, xMaxC)
    c_flat_imgs = image_toolbox.img_crop(raw_flat_imgs, yMinC, yMaxC, xMinC, xMaxC)
    c_WO_imgs = image_toolbox.img_crop(raw_WO_imgs, yMinC, yMaxC, xMinC, xMaxC)

    # Average the images: average = sum of each pixel int/num images for each pixel
    avg_amb_img = image_toolbox.img_avg(c_amb_imgs)
    avg_dark_img = image_toolbox.img_avg(c_dark_imgs)
    avg_flat_img = image_toolbox.img_avg(c_flat_imgs)
    avg_WO_img = image_toolbox.img_avg(c_WO_imgs)

    # Correct images that need correcting
    corr_WO_img = image_toolbox.flat_field_correction(avg_WO_img, avg_amb_img, avg_dark_img, avg_flat_img)

    # The averaged ambient, dark, and flat images are used in each flat field correction (shown later)
    
    # repo_path = Glow-In-The-Dark
    repo_path = pathlib.Path(__file__).parents[1]
    # pressure_dir = Glow-In-The-Dark/images/camera_sv_images/pressures
    pressure_dir = os.path.join(repo_path, 'images/camera_sv_images/pressures')
    # List of all folders in the pressure directory (excludes __init__.py)
    pressure_folders = [item for item in os.listdir(pressure_dir) if os.path.isdir(os.path.join(pressure_dir, item))]
    # Use number of folders to initialize arrays to hold pressure images and values
    pressure_imgs = np.zeros(len(pressure_folders))
    pressure_val = np.zeros(len(pressure_imgs))
    first_dim = avg_amb_img.shape[0] 
    second_dim = avg_amb_img.shape[1]
    # 3D matrix where each 'page' is the new pressure value
    intensities = np.zeros((first_dim, second_dim, len(pressure_imgs)))

    # Pressure of image will correspond to folder name
    for pos, pressure_folder in enumerate(pressure_folders):
        try:
            pressure_val[pos] = float(str(pressure_folder).replace('_', '.', 1))
        except:
            pressure_val[pos] = float(str(pressure_folder))

    # Sort the pressure values and images so they go from smallest to largest pressure
    # Simpler to sort now than to try to sort the 3d array later
    sorted_indices = np.argsort(pressure_val)
    pressure_val = pressure_val[sorted_indices]
    pressure_folders = np.asarray(pressure_folders)[sorted_indices]

    # Loop though each pressure folder
    for pos, pressure_folder in enumerate(pressure_folders):
        raw_pressure_img = image_toolbox.collect_images(f'camera_sv_images/pressures/{pressure_folder}')
        c_pressure = image_toolbox.img_crop(raw_pressure_img, yMinC, yMaxC, xMinC, xMaxC)
        avg_pressure_img = image_toolbox.img_avg(c_pressure)
        # Store the corrected pressure image
        pressure_img_corrected = image_toolbox.flat_field_correction(avg_pressure_img, avg_amb_img, avg_dark_img, avg_flat_img)
        intensities[:, :, pos] = np.divide(corr_WO_img, pressure_img_corrected)
        #print(intensities[:, :, pos])

    first_dim = intensities.shape[0] 
    second_dim = intensities.shape[1]
    third_dim = intensities.shape[2]
    I0_over_I = np.zeros((first_dim, second_dim, intensities.shape[2]))

    #for i in np.arange(len(pressure_val)):
    #    I0_over_I[:, :, i] = np.divide(intensities[:, :, i], corrected_WO_img)
    I0_over_I = intensities
       
    
    #for row in np.arange(first_dim):
    #    for pres in np.arange(len(pressure_val)):
    #        plt.scatter(pressure_val[pres]*np.ones(second_dim), I0_over_I[row, :, pres])
    slope = np.zeros_like(I0_over_I)
    intercept = np.zeros_like(I0_over_I)

    pressure_3d = np.ones_like(I0_over_I)
    for i in np.arange(third_dim):
        pressure_3d[:, :, i] = pressure_3d[:, :, i] * pressure_val[i]

    slope_mat = np.zeros((I0_over_I.shape[0], I0_over_I.shape[1]))
    intercept_mat = np.zeros((I0_over_I.shape[0], I0_over_I.shape[1]))

    for i in range(I0_over_I.shape[0]):
        for j in range(I0_over_I.shape[1]):
            slope, intercept, _, _, _ = stats.linregress(pressure_3d[i, j, :], I0_over_I[i, j, :])
            slope_mat[i, j] = slope
            intercept_mat[i, j] = intercept  

    #correction_factor = 1.2
    #slope_mat = correction_factor*slope_mat

    #slope_mat = np.ones_like(slope_mat)*np.median(slope_mat)

    # A is intercept
    # B is slope
    return intercept_mat, slope_mat


def get_A_B(n_cal, p_p0_array, I0_I_array):
    """From Owen Brown's dissertation. Trying to perform Least Squares Regression with photo diode data."""
    # L[A; B]' = C
    # [A:B]' = L^-1 * C
    L = np.zeros((2, 2))
    C = np.zeros((2, 1))

    L[0, 0] = n_cal
    L[0, 1] = np.sum(p_p0_array)
    L[1, 0] = np.sum(p_p0_array)
    L[1, 1] = np.sum(p_p0_array**2)

    C[0, 0] = np.sum(I0_I_array)
    C[1, 0] = np.sum(p_p0_array*I0_I_array)
    
    L_inv = np.linalg.inv(L)
    A_B = np.dot(L_inv, C)
    A_sv = A_B[0]
    B_sv = A_B[1]
    return A_sv, B_sv

