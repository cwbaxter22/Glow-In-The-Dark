"""
Simple module designed to allow for quick plot creation of samples.
Please remove any sample names before each push to avoid sharing research data.
Data itself is filtered out in the .gitignore/ should not be git added
"""

import os
import pathlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

def create_sv(folder_of_wl, LOWER_B, UPPER_B):
    repo_path = pathlib.Path(__file__).parents[1]
    all_txt_files = os.listdir(os.path.join(repo_path, r"photo_diode_data", folder_of_wl))
    
    TRAPSPACING = 0.215

    ox_conc = np.zeros(len(all_txt_files))
    areas_under_curve = np.zeros(len(all_txt_files))

    for pos, file in enumerate(all_txt_files):
        file_number_str = str(file.replace('.txt', '', 1))
        ox_conc[pos] = float(file_number_str.replace('_', '.', 1)) * (20.9 / 760)

        txt_path = os.path.join(repo_path, r"photo_diode_data", folder_of_wl, file)
        df = pd.read_csv(txt_path, header=None, sep='\t', skiprows=17)
        df.drop(df.tail(1).index, inplace=True)
        df = df.apply(pd.to_numeric)
        df.columns = ['wavelength', 'intensity']
        unfiltered_int = df.loc[(df['wavelength'] > LOWER_B) & (df['wavelength'] < UPPER_B), 'intensity']
        filtered_int = uniform_filter1d(unfiltered_int, size=5)
        area = np.trapz(filtered_int, dx=TRAPSPACING)
        areas_under_curve[pos] = area

    ox_conc, areas_under_curve = zip(*sorted(zip(ox_conc, areas_under_curve)))
    ratios = areas_under_curve[0] / areas_under_curve
    return ox_conc, ratios

def plot_creation(samples):
    plt.figure()
    for sample_num in samples:
        ox_conc_out, ratios_out = create_sv(sample_num, 645, 670)
        plt.scatter(ox_conc_out, ratios_out, label=sample_num)

    plt.xlabel('Oxygen Concentration (%)')
    plt.ylabel('Intensity (I0/I)')
    plt.legend()
    plt.show()
