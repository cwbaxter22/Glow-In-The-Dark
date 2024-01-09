"""Perform operations on matrices based on SV relation

Module contains X functions:
1) 

"""

import numpy as np
import os
import pathlib
import pandas as pd

def get_int_function():
    """
    Generate a class that acts as a funciton to convert intensity values to pressure

    Arguments:
    ----------
    
    Returns:
    ----------
    ratio_class (np class): class that converts int to pressure 
    
    """
    repo_path = pathlib.Path(__file__).parents[1]
    pd_folder = os.path.join(repo_path, 'pd_calibration_curve')
    files_in_pd_folder=os.listdir(pd_folder)
    csv_filename = [filename for filename in files_in_pd_folder if filename[-4:] in [".csv",".CSV"]]
    pviSpreadsheetPath = os.path.join(repo_path, pd_folder, csv_filename[0])
    pviDF = pd.read_csv(pviSpreadsheetPath)

    pressures_torr = np.array(pviDF["Pressure"].tolist())
    pressures_pascal = pressures_torr*133.3 #Convert to Pa
    ratio = np.array(pviDF["ratio"].tolist())

    # polyfit credit: https://stackoverflow.com/questions/19165259/python-numpy-scipy-curve-fitting
    # calculate polynomial
    # Fourth order fit seems to accomodate data well
    # Function being calculated is pressure as a function of I/I0 where I0 is the lowest intensity

    # Caclulate the polynomials coefficients
    z = np.polyfit(ratio, pressures_pascal, 5)

    # Create class that is the polynomial
    # Values can just be plugged into it like a function to convert
    # intensity to pressure
    ratio_class = np.poly1d(z)
    return ratio_class
