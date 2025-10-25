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

from scipy.interpolate import interp1d, CubicSpline

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


# Using cubic spline interpolation (C^2 differentiable)
g_rho_spline_interp = CubicSpline(temperature, g_rho_data, bc_type='natural', extrapolate=True)
 
def g_rho_spline(T):
    """
    Smooth g_rho interpolation using a cubic spline.
    For T outside data range, extrapolates smoothly.
    """
    return g_rho_spline_interp(T)

def dg_rho_spline(T):
    """
    First derivative of g_rho with respect to T.
    Computed analytically from the cubic spline.
    """
    return g_rho_spline_interp(T, 1)  # 1 → first derivative
    # # The spline derivative isn't particularly smooth:
    # g = lambda T : g_rho_spline(T)
    # T_eps = 1e-6 * T
    
    # dg = g(T+T_eps)
    # dg -= g(T-T_eps)
    # dg *= 1./(2*T_eps)

    # return dg

def d2g_rho_spline(T):
    """
    Second derivative of g_rho with respect to T.
    Computed analytically from the cubic spline.
    """
    return g_rho_spline_interp(T, 2)  # 2 → second derivative
    # # The spline second derivative isn't particularly smooth:
    # dg = lambda T : dg_rho_spline(T)
    # T_eps = 1e-6 * T
    
    # d2g = dg(T+T_eps)
    # d2g -= dg(T-T_eps)
    # d2g *= 1./(2*T_eps)

    # return d2g