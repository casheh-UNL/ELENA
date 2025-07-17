from pathlib import Path

# Absolute file path .py
script_dir = Path(__file__).parent


# This dataset was downloaded from https://member.ipmu.jp/satoshi.shirai/DM2020/
# Load data from arXiv:2005.03544 [hep-ph]
# Data format: Temperature [GeV], g_s, g_rho

import numpy as np
data = np.loadtxt(str(script_dir.parent) + '/data/raw/eos2020.dat')

# Extract columns
temperature = data[:, 0]  # Temperature [GeV]
g_s_data = data[:, 1]     # g_s values
g_rho_data = data[:, 2]   # g_rho values

from scipy.interpolate import interp1d

# Create interpolation functions
# Using linear interpolation within data range
g_s_interp = interp1d(temperature, g_s_data, kind='linear', 
                      bounds_error=False, fill_value=(g_s_data[0], g_s_data[-1]))

g_rho_interp = interp1d(temperature, g_rho_data, kind='linear', 
                        bounds_error=False, fill_value=(g_rho_data[0], g_rho_data[-1]))

# Define the final functions with proper boundary handling
def g_s(T):
    """
    Returns g_s as a function of temperature T [GeV]
    For T > T_max, returns g_s_data[-1]
    For T < T_min, returns g_s_data[0]
    """
    T = np.asarray(T)
    result = np.where(T > temperature[-1], g_s_data[-1], 
                     np.where(T < temperature[0], g_s_data[0], g_s_interp(T)))
    return result

def g_rho(T):
    """
    Returns g_rho as a function of temperature T [GeV]
    For T > T_max, returns g_rho_data[-1]
    For T < T_min, returns g_rho_data[0]
    """
    T = np.asarray(T)
    result = np.where(T > temperature[-1], g_rho_data[-1], 
                     np.where(T < temperature[0], g_rho_data[0], g_rho_interp(T)))
    return result