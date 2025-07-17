import numpy as np

from dof_interpolation import g_rho

def cs2(T, m, true_vev, units = 'GeV'):
        speed2 = m.dVdT(true_vev[T], T, include_radiation=True, include_SM = True, units = units) / (T * m.d2VdT2(true_vev[T], T, include_radiation=True, include_SM = True, units = units))
        return min(1/3, speed2)


def alpha_th_bar(T, m, V_min_value, false_vev, true_vev, units = 'GeV'):
    delta_rho = - V_min_value[T] -  T * (m.dVdT(false_vev[T], T, include_radiation=True, include_SM = False,  units = units) - m.dVdT(true_vev[T], T, include_radiation=True, include_SM = False,  units = units))
    delta_p = V_min_value[T] / cs2(T, m, true_vev, units)
    wf = - T * m.dVdT(false_vev[T], T, include_radiation=True, include_SM = True,  units = units)
    wf_DS = - T * m.dVdT(false_vev[T], T, include_radiation=True, include_SM = False, units = units)

    return (delta_rho - delta_p) / (3 * wf), (delta_rho - delta_p) / (3 * wf_DS)
    

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
        self.f_sw = self.freq_redshift * 3.4 / ((self.v_w - self.c_s) * self.R_star)
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