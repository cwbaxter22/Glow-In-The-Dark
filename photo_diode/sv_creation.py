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
import peakutils
from scipy.linalg import cholesky

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

def convert_to_pressure(ox_conc, intensities):
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
    
    # Calculate the slope of the line
    slope = (max(normalized_intensities) - min(normalized_intensities)) / (max(pressure) - min(pressure))
    
    #Either this or min(normalized_intensities)
    return 1/slope, min(normalized_intensities)
