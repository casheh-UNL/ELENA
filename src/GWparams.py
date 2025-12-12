# Adapts expressions from https://arxiv.org/abs/1903.09642. While this paper focuses on supercooled
# phase transitions, they "note that the expressions given below for the GW spectra are more general."
# Generalizing to slower wall speeds can be done by neglecting kappa_col and using kappa_sw given in
# https://arxiv.org/abs/1004.4187.

import numpy as np
from numpy.linalg import lstsq
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar

from dof_interpolation import g_rho

def cs2(T, m, true_vev, units = 'GeV'):
        speed2 = m.dVdT(true_vev[T], T, include_radiation=True, include_SM = True, units = units) / (T * m.d2VdT2(true_vev[T], T, include_radiation=True, include_SM = True, units = units))
        return min(1/3, speed2.flatten())


def alpha_th_bar(T, m, V_min_value, false_vev, true_vev, units = 'GeV'):
    delta_rho = - V_min_value[T] -  T * (m.dVdT(false_vev[T], T, include_radiation=True, include_SM = False,  units = units) - m.dVdT(true_vev[T], T, include_radiation=True, include_SM = False,  units = units))
    delta_p = V_min_value[T] / cs2(T, m, true_vev, units)
    wf = - T * m.dVdT(false_vev[T], T, include_radiation=True, include_SM = True,  units = units)
    wf_DS = - T * m.dVdT(false_vev[T], T, include_radiation=True, include_SM = False, units = units)

    return (delta_rho - delta_p) / (3 * wf), (delta_rho - delta_p) / (3 * wf_DS)
    

def beta(Temps, ratio_V, Gamma, H, T_nuc, T_perc, verbose = False):
    idx_nuc = np.argmin(np.abs(Temps - T_nuc))
    idx_perc = np.argmin(np.abs(Temps - T_perc))

    Gamma_n = Gamma[idx_nuc]
    H_n = H[idx_nuc]

    times = cumulative_trapezoid(-np.flip((ratio_V[idx_perc:idx_nuc+1] / (3 * H[idx_perc:idx_nuc+1]))), np.flip(Temps[idx_perc:idx_nuc+1]), initial=0)
    times = np.flip(times)

    t = np.flip(H_n*times)
    ft = np.flip(Gamma[idx_perc:idx_nuc+1]/Gamma_n)
    ln_ft = np.log(ft)

    # Create the design matrix
    X = np.vstack((t, t**2)).T  # Stack t and t^2 as columns to create a design matrix

    # Fit
    coefs, _, _, _ = lstsq(X, ln_ft, rcond=None)  # Fit 'ln_ft' against 't' and 't^2'

    # Extract coefficients
    a_fit = coefs[0]  # Coefficient for the linear term (t)
    b_fit = coefs[1]  # Coefficient for the quadratic term (t^2)

    beta_Hn = a_fit
    gamma_Hn = np.sqrt(- 2 * b_fit)

    if verbose:
        return beta_Hn, gamma_Hn, t, np.flip(Gamma[idx_perc:idx_nuc+1]), np.flip(Temps[idx_perc:idx_nuc+1]), np.flip(H[idx_perc:idx_nuc+1])
    else:
        return beta_Hn, gamma_Hn
    

class GW_SuperCooled:

    def __init__(self, T_star, alpha, alpha_inf, alpha_eq, R_star, gamma_star, H_star, c_s = 1/np.sqrt(3), v_w = 1, units = 'GeV', dark_dof = 4):
        if units == 'MeV':
            self.GeV = 1e3
        elif units == 'GeV':
            self.GeV = 1
        elif units == 'TeV':
            self.GeV = 1e-3

        self.dark_dof = dark_dof

        self.alpha = alpha
        self.alpha_inf = alpha_inf
        self.alpha_eq = alpha_eq

        self.T_star = T_star
        self.T_reh = (1 + self.alpha)**(1/4) * self.T_star
        self.g_rad_T_reh = g_rho(self.T_reh / self.GeV) + self.dark_dof

        self.gamma_eq = (self.alpha - self.alpha_inf) / self.alpha_eq
        self.R_star = R_star
        self.H_star = H_star
        self.gamma_star = gamma_star

        self.c_s = c_s
        self.v_w = v_w

        self.kappa_col = self.E_wall_over_E_V()
        self.alpha_eff = self.alpha * (1 - self.kappa_col)
        self.kappa_sw = (self.alpha_eff / self.alpha) * (self.alpha_eff / (0.73 + 0.083 * np.sqrt(self.alpha_eff) + self.alpha_eff))

        self.U_f = np.sqrt((3/4) * (self.alpha_eff / (1 + self.alpha_eff)) * self.kappa_sw)
        self.tau_sw = np.min([1/H_star, self.R_star / self.U_f])

        self.amplitude_redshift = 1.67e-5 * (100 / self.g_rad_T_reh)**(1/3)
        self.freq_redshift = 1.65e-5 * (self.T_reh / (100 * self.GeV)) * (self.g_rad_T_reh / 100)**(1/6)  / H_star

        self.f_col = self.freq_redshift * 0.51 / R_star
        # self.f_sw = self.freq_redshift * 3.4 / ((self.v_w - self.c_s) * self.R_star)
        self.f_sw = 2.6e-5 * (self.T_reh / (100 * self.GeV)) * (self.g_rad_T_reh / 100)**(1/6) / (self.R_star * H_star) # eq. (31) of https://arxiv.org/pdf/1910.13125
        self.f_turb = self.freq_redshift * 3.9 / ((self.v_w - self.c_s) * self.R_star)

    def E_wall_over_E_V(self):
        if self.gamma_star > self.gamma_eq:
            return (self.gamma_eq / self.gamma_star) * (1 - (self.alpha_inf / self.alpha) * (self.gamma_eq / self.gamma_star)**2)
        else:
            return 1 - self.alpha_inf / self.alpha

    def Omegah2coll(self, f_star): # (3.2) in 1903.09642
        f_star = np.asarray(f_star)
        spectrum_star = 0.024 * (self.H_star * self.R_star)**2 * (self.kappa_col * self.alpha / (1 + self.alpha))**2 * (f_star / self.f_col)**3 * (1 + 2 * (f_star / self.f_col)**2.07)**(-2.18)
        return self.amplitude_redshift * spectrum_star
    
    def Omegah2sw(self, f_star): # (3.5) in 1903.09642
        f_star = np.asarray(f_star)
        return self.amplitude_redshift * 0.38 * (self.H_star * self.R_star) * (self.H_star * self.tau_sw) * (self.kappa_sw * self.alpha / (1 + self.alpha))**2 * (f_star / self.f_sw)**3 * (1 +  (3/4) * (f_star / self.f_sw)**2)**(-7/2)

    def Omegah2turb(self, f_star): # (3.9) in 1903.09642
        f_star = np.asarray(f_star)
        return self.amplitude_redshift * 6.8 * (self.H_star * self.R_star) * (1 - self.H_star * self.tau_sw) * (self.kappa_sw * self.alpha / (1 + self.alpha))**(3/2) * (f_star / self.f_turb)**3 * (1 +  (f_star / self.f_turb))**(-11/3) / (1 + 8*np.pi * f_star / self.H_star)

    def Omegah2(self, f_star):
        f_star = np.asarray(f_star)
        return self.Omegah2coll(f_star) + self.Omegah2sw(f_star) + self.Omegah2turb(f_star)
    
    

    def find_peak(self, f_min=None, f_max=None, verbose=False):
        '''
        Find the peak frequency and value of the total GW spectrum.

        Parameters
        ----------
        f_min : float, optional
            Minimum frequency (in the same units as f_star).
            Defaults to 1e-3 * min(self.f_col, self.f_sw, self.f_turb).
        f_max : float, optional
            Maximum frequency (in the same units as f_star).
            Defaults to 1e3 * max(self.f_col, self.f_sw, self.f_turb).
        verbose : bool, optional
            If True, also returns the peaks of each contribution
            (collisions, sound waves, turbulence).

        Returns
        -------
        tuple
            (f_peak_total, Omega_peak_total)
            If verbose=True, also returns:
            (f_peak_col, Omega_peak_col),
            (f_peak_sw, Omega_peak_sw),
            (f_peak_turb, Omega_peak_turb)
        '''
        # Define search range if not provided
        if f_min is None:
            f_min = 1e-3 * min(self.f_col, self.f_sw, self.f_turb)
        if f_max is None:
            f_max = 1e3 * max(self.f_col, self.f_sw, self.f_turb)

        # Helper to maximize a positive function by minimizing its negative
        def neg_total(f):
            return -self.Omegah2(f)

        res_total = minimize_scalar(neg_total, bounds=(f_min, f_max), method='bounded')
        f_peak_total = res_total.x
        Omega_peak_total = -res_total.fun

        if not verbose:
            return f_peak_total, Omega_peak_total

        # # Compute individual peaks
        # def neg_col(f): return -self.Omegah2coll(f)
        # def neg_sw(f): return -self.Omegah2sw(f)
        # def neg_turb(f): return -self.Omegah2turb(f)

        # # Use tighter bounds centered on each characteristic frequency
        # res_col = minimize_scalar(neg_col, bounds=(1e-2*self.f_col, 1e2*self.f_col), method='bounded')
        # res_sw = minimize_scalar(neg_sw, bounds=(1e-2*self.f_sw, 1e2*self.f_sw), method='bounded')
        # res_turb = minimize_scalar(neg_turb, bounds=(1e-2*self.f_turb, 1e2*self.f_turb), method='bounded')

        # peaks = {
        #     "total": (f_peak_total, Omega_peak_total),
        #     "collision": (res_col.x, -res_col.fun),
        #     "sound_wave": (res_sw.x, -res_sw.fun),
        #     "turbulence": (res_turb.x, -res_turb.fun)
        # }
        peaks = {
            "total":      (f_peak_total, Omega_peak_total),
            "collision":  (self.f_col, self.Omegah2coll(self.f_col)),
            "sound_wave": (self.f_sw, self.Omegah2sw(self.f_sw)),
            "turbulence": (self.f_turb, self.Omegah2turb(self.f_turb))
        }
        return peaks["total"], peaks["collision"], peaks["sound_wave"], peaks["turbulence"]