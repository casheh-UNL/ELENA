""" 
espinosa.py

This module provides the class Vt_vec for the computation of the action using the Espinosa method [1805.03680].
"""
import numpy as np
from scipy.special import gamma

Tbarrier = None
Tsmooth = None

def get_Tsmooth():
    global Tsmooth
    return Tsmooth

def set_Tsmooth(value):
    global Tsmooth
    Tsmooth = value


def find_extrema(V, dV, T, Phimax = 150, step = 1):
        phi = np.arange(step, Phimax + step, step)
        phi = phi.reshape(-1,1)
        
        v = V(phi, T)
        argmaxV = np.argmax(v)
        while (argmaxV + 1) != len(v):
            if np.isnan(v[argmaxV]):
                return [], [], []
            else:
                Phimax *= 2
                phi = np.arange(step, Phimax, step)
                phi = phi.reshape(-1,1)
                
                v = V(phi, T)
                argmaxV = np.argmax(v)

        dv = dV(phi, T)
        signs = np.sign(dv).reshape(-1)
        signs = np.round(signs).astype(int)
        sign_changes = np.diff(signs) # negative: max, positive: min
        change_indices = np.nonzero(abs(sign_changes) == 2)[0]

        filtered_change_indices = []
        previous = 0
        for idx in np.flip(change_indices):
            if idx != previous - 1:
                filtered_change_indices.append(idx)
            previous = idx
        filtered_change_indices = np.array(filtered_change_indices)
        filtered_change_indices = np.flip(filtered_change_indices)

        loc = []
        val = []
        kind = []
        for idx in filtered_change_indices:
            this_one = "max" if sign_changes[idx] < 0 else "min"
            if this_one == "max":
                extreme_location = np.interp(0, np.flip(dv[idx:idx+2]).flatten(), np.flip(phi[idx:idx+2]).flatten())
            else:
                extreme_location = np.interp(0, dv[idx:idx+2].flatten(), phi[idx:idx+2].flatten())
            potential_value = np.interp(extreme_location, phi[idx:idx+2].flatten(), v[idx:idx+2].flatten())

            kind.append(this_one)
            loc.append(extreme_location)
            val.append(potential_value)

        zeroes_indices = np.where(dv==0)[0]

        real_zeroes = []
        for idx in zeroes_indices:
            if dv[idx-1] * dv[idx+1] < 0:
                real_zeroes.append(idx)

        for idx in real_zeroes:
            kind.append("max" if dv[idx-1] > dv[idx+1] else "min")
            loc.append(phi[idx][0])
            val.append(v[idx])


        # Deals with cases where consecutive zeroes are present
        diff = np.diff(zeroes_indices)

        starts = np.where(diff > 1)[0] + 1
        ends = np.where(diff > 1)[0]

        starts = np.insert(starts, 0, 0)
        ends = np.append(ends, len(zeroes_indices) - 1)

        if len(zeroes_indices) > 1:
            sequences = [(zeroes_indices[start], zeroes_indices[end]) for start, end in zip(starts, ends)]

            filtered_sequences = [seq for seq in sequences if seq[0] != seq[1]]

            for pair in filtered_sequences:
                if dv[pair[0]-1] * dv[pair[1]+1] < 0:
                    this_one = "max" if dv[pair[0]-1] > dv[pair[1]+1] else "min"
                    if this_one == "max":
                        extreme_location = np.interp(0, dv[[pair[1]+1, pair[0]-1]].flatten(), phi[[pair[1]+1,pair[0]-1]].flatten())
                    else:
                        extreme_location = np.interp(0, dv[[pair[0]-1,pair[1]+1]].flatten(), phi[[pair[0]-1,pair[1]+1]].flatten())
                    potential_value = np.interp(extreme_location, phi[[pair[0]-1,pair[1]+1]].flatten(), v[[pair[0]-1,pair[1]+1]].flatten())
                    kind.append(this_one)
                    loc.append(extreme_location)
                    val.append(potential_value)

        kind, loc, val = np.array(kind), np.array(loc), np.array(val)

        kind = kind[np.argsort(loc)].tolist()
        val = val[np.argsort(loc)].tolist()
        loc = loc[np.argsort(loc)].tolist()

        if len(kind) > 2:
            global_min_idx = np.argmin(val)
            #print("Cutting at:", kind[global_min_idx], loc[global_min_idx], val[global_min_idx])
            kind = kind[:global_min_idx+1]
            loc = loc[:global_min_idx+1]
            val = val[:global_min_idx+1]
            #print(kind[:global_min_idx+1], loc[:global_min_idx+1], val[:global_min_idx+1])

        return kind, loc, val


class Vt_vec:
    """
    Class for the vectorised computation of the action using the Espinosa method [1805.03680].
    It takes advantage of a vectorised potential definition to compute both V(phi) and S_E,d(phi_0) [eq. 37] for a given temperature T in a single step.
    It is then immediate to identify the value of phi_0 that minimises S_E,d(phi_0).

    Attributes:
        - T: temperature
        - V: vectorised potential in the form of the one returned by CosmoTransitions
        - dV: vectorised derivative of the potential in the form of the one returned by CosmoTransitions
        - step_phi: step size for the phi grid
        - step_phi0: step size for the phi0 grid
        - d: number of dimensions
        - vev0: vev at T = 0

    Methods:
        - Vt1: computes the first order approximation [eq. 17] of the potential V_t(phi, phi_0) [eq. 7]
        - Vt2: computes the second order approximation [eq. 18] of the potential V_t(phi, phi_0) [eq. 7]
        - Vt3: computes the third order approximation [eq. 19] of the potential V_t(phi, phi_0) [eq. 7]
        - d1Vt3: computes the first derivative of Vt3 with respect to phi
        - d2Vt3: computes the second derivative of Vt3 with respect to phi
        - Vt4: computes the fourth order approximation [eq. 20] of the potential V_t(phi, phi_0) [eq. 7]
        - dVt4: computes the derivative of Vt4 with respect to phi
        - compute_sums: computes a discretized approximation of the integrand in eq. 37 for all phi_0 values
    """

    def __init__(self, T, V, dV, step_phi = 1e-3, precision = 1e-3, d = 3, vev0 = 100, int_threshold = 2e-1, ratio_vev_step0 = 50, save_all = False):
        global Tbarrier, Tsmooth
        self.save_all = save_all

        self.all_sums = {}
        self.all_vt4 = {}

        self.d = d
        self.T = T
        self.vev0 = vev0

        self.ratio_vev_step0 = ratio_vev_step0
        self.step_phi = step_phi * self.vev0
        self.step_phi0 = self.vev0 / self.ratio_vev_step0

        self.int_threshold = int_threshold

        self.Phimax = 1.5 * self.vev0
        self.phi = np.arange(self.step_phi, self.Phimax + self.step_phi, self.step_phi)
        self.V = V(self.phi.reshape(-1,1), self.T)
        self.dV = dV(self.phi.reshape(-1,1), self.T)
        self.argmaxV = np.argmax(self.V)

        kind, loc, val  = find_extrema(V, dV, self.T, Phimax = 2 * self.vev0, step = self.vev0 * 1e-3)
        
        if len(kind) < 2:
            self.barrier = False
            Tsmooth = self.T
        else:
            self.false_min = 0
            true_min_idx = np.argmin(val)
            self.true_min = loc[true_min_idx]

            if len(kind) >=2 and kind[true_min_idx] == 'min' and kind[true_min_idx -1] == 'max' and true_min_idx >= 1:
                self.true_min = loc[true_min_idx]
                self.phi_max_V = loc[true_min_idx - 1]
                self.barrier = True
                if len(kind) > 2 and kind[true_min_idx - 2] == 'min' and val[true_min_idx - 2] < 0 and ( (loc[true_min_idx] - loc[true_min_idx - 2])/loc[true_min_idx] > 0.01 ):
                    self.false_min = loc[true_min_idx - 2]
                else:
                    self.false_min = 0

            self.phi_original = self.phi
            self.false_vev_argument = self.find_closest_index(self.phi, self.false_min)
            self.phi_shift = self.phi[self.false_vev_argument]
            self.phi_original_false_vev = self.phi_shift - self.step_phi
            self.V_shift = self.V[self.false_vev_argument]
            self.argminV = self.find_closest_index(self.phi, self.true_min)
            self.vevT_original = self.phi[self.argminV]
            if val[true_min_idx] < 0:
                while self.V[self.argminV-1] > 0:
                    self.argminV += 1
            self.phi = self.phi[self.false_vev_argument+1:self.argminV]
            self.phi_cut = self.phi
            self.V = self.V[self.false_vev_argument+1:self.argminV]
            self.dV = self.dV[self.false_vev_argument+1:self.argminV]
            self.phi = self.phi - self.phi_shift
            self.phi_final = self.phi
            self.V = self.V - self.V_shift
            self.false_vev_argument = 0
            self.argminV = len(self.phi) - 1
            self.min_V = self.V[self.argminV]
            self.vevT = self.phi[self.argminV]
            self.argbarrier = np.argmax(self.V)
            self.phiT = self.phi[self.argbarrier]

            if self.V[self.argbarrier] > 0 and self.min_V < 0 and self.argminV > self.argbarrier:
                # The integral 37 in 1805.03680 is zero at phi+ and phi0, but if computed numerically is undetermined. We exclude these points from the beginning
                self.barrier = True
                Tbarrier = self.T
                self.VT = np.max(self.V) 
                self.min_phi0 = (self.phi[self.argbarrier:])[self.V[self.argbarrier:] < 0][0]
                self.phi0 = np.arange(self.min_phi0, self.vevT + self.step_phi0, self.step_phi0)
                counter = 0
                diff = np.inf
                action_over_T_old = np.inf
                while diff > precision or counter == 0:                    
                    self.V0 = V((self.phi0 + self.phi_shift).reshape(-1,1), self.T) - self.V_shift
                    self.dV0 = dV((self.phi0 + self.phi_shift).reshape(-1,1), self.T).reshape(1, -1)
                    self.vt1 = self.Vt1(self.phi, self.phi0)
                    self.vt2 = self.Vt2(self.phi, self.phi0)
                    self.vt3 = self.Vt3(self.phi, self.phi0)
                    self.phi0T = self.phi0 - self.phiT
                    self.c = 4 * self.phiT**2 * self.phi0T**2 * (self.phi0**2 + 2 * self.phi0T * self.phiT)              
                    self.Vt3T = (self.Vt3(np.array([self.phiT]), self.phi0)).reshape(1,-1)
                    self.d1Vt3T = (self.d1Vt3(np.array([self.phiT]), self.phi0))
                    self.d2Vt3T = (self.d2Vt3(np.array([self.phiT]), self.phi0))[0]
                    self.Ut3T = 4 * self.d1Vt3T**2 + 6*(self.VT - self.Vt3T) * self.d2Vt3T
                    self.a0T = -6*(self.VT - self.Vt3T)*(self.phi0**2 - 6*self.phi0T*self.phiT) - 8*self.phiT*(self.phi0T-self.phiT)*self.phi0T*self.d1Vt3T + 3 * self.phiT**2 * self.phi0T**2 * self.d2Vt3T
                    self.a4 = ((self.a0T - np.sqrt(self.a0T**2 - self.c*self.Ut3T) ) / self.c)[0]
                    self.vt4 = self.Vt4(self.phi, self.phi0)
                    self.dvt4 = self.dVt4(self.phi, self.phi0)
                    self.sums = self.compute_sums()
                    if self.save_all:
                        for el in self.phi0:
                            self.all_sums[el] = self.sums[self.phi0 == el][0]
                            self.all_vt4[el] = self.vt4[self.phi0 == el][0]
                    self.action = ( ((self.d-1)**(self.d-1) * (2*np.pi)**(self.d/2))/(gamma(1 + self.d/2)) ) * np.trapz((self.integrand[self.phi0==self.phi0_min])[0], self.phi)
                    self.action_over_T = self.action / self.T if self.T > 0 else np.inf

                    diff = np.abs(action_over_T_old - self.action_over_T) / action_over_T_old if counter > 0 else np.inf
                    action_over_T_old = self.action_over_T
                    
                    self.phi0_old = self.phi0
                    self.phi0 = np.arange(np.max([self.phi0_min - self.step_phi0, self.min_phi0]), self.phi0_min + 1.1*self.step_phi0, self.step_phi0 / 10)
                    self.step_phi0 /= 10
                    counter += 1
            else:
                self.barrier = False
                Tsmooth = self.T

    def Vt1(self, phi, phi0):
        phi_reshaped = phi[:, np.newaxis] 
        result = (self.V0 / phi0) * phi_reshaped
        return result.T

    def Vt2(self, phi, phi0):
        phi_reshaped = phi[:, np.newaxis] 
        result = phi_reshaped * (3 * phi0 * self.dV0 - 4 * self.V0) * (phi_reshaped - phi0) / (4 * phi0**2)
        return self.Vt1(phi, phi0) + result.T

    def Vt3(self, phi, phi0):
        phi_reshaped = phi[:, np.newaxis] 
        result = phi_reshaped * (3 * phi0 * self.dV0 - 8 * self.V0) * (phi_reshaped - phi0)**2 / (4 * phi0**3)
        return self.Vt2(phi, phi0) + result.T

    def d1Vt3(self, phi, phi0):
        phi_reshaped = phi[:, np.newaxis] 
        result = 3*phi_reshaped * (self.dV0*(3*phi_reshaped - 2*phi0)*phi0 + 8*(phi0-phi_reshaped)*self.V0) / (4*phi0**3)
        return result

    def d2Vt3(self, phi, phi0):
        phi_reshaped = phi[:, np.newaxis] 
        result = 3*(self.dV0*(3*phi_reshaped - phi0)*phi0 + 4*(phi0-2*phi_reshaped)*self.V0) / (2*phi0**3)
        return result

    def Vt4(self, phi, phi0):
        phi_reshaped = phi[:, np.newaxis]
        result = phi_reshaped**2 * (phi_reshaped - phi0)**2 * self.a4
        return self.Vt3(phi, phi0) + result.T

    def dVt4(self, phi, phi0):
        phi_reshaped = phi[:, np.newaxis] 
        result = 2 * self.a4 * phi_reshaped * (2*phi_reshaped**2 - 3*phi_reshaped*phi0 + phi0**2)
        return self.d1Vt3(phi, phi0).T + result.T

    def compute_sums(self):
        # Broadcast phi and phi0 for comparison
        phi_broadcasted = self.phi[np.newaxis, :]  
        phi0_broadcasted = self.phi0[:, np.newaxis]  

        # Create a mask where phi < phi0[j] (the integrand is zero at the endpoint phi0)
        mask = phi_broadcasted < phi0_broadcasted
        self.mask = mask
        
        # Compute V - vt4
        difference = self.V - self.vt4

        self.check_min_diff = np.min(difference[mask])
        # Apply the mask to keep only the entries where phi <= phi0[j]
        masked_difference = np.where(mask, difference, 0)
        masked_difference[masked_difference < 0] = 0
        self.integrand = ( masked_difference**(self.d/2) ) / self.dvt4**(self.d - 1)
        self.non_zero_in_integrand = np.array([np.sum(sub_array > 0) for sub_array in self.integrand])
        sums = np.sum(self.integrand, axis=1)

        sums_mask = (sums > 0)
        positive_sums = sums[sums_mask]

        # print(self.T)
        # print(self.phi0[0], self.phi0[-1], self.step_phi0)
        # #min_sum_index = np.nanargmin(sums)
        # for i in range(len(self.phi0)):
        #     #marker = " <-- MIN" if i == min_sum_index else ""
        #     marker = ""
        #     print(f"{self.phi0[i]} -> {sums[i]}{marker}")
        #     print(f"{self.phi0[i]} -> {self.a0T**2 - self.c*self.Ut3T}")
        # print("\n")

        min_index_in_positive = np.nanargmin(positive_sums)
        original_index = np.where(sums_mask)[0][min_index_in_positive]
        self.phi0_min = self.phi0[original_index]
        self.V_exit = self.V0[self.phi0 == self.phi0_min]
        return sums
        
    def find_closest_index(self, arr, target):
        # Calculate the absolute differences between each element and the target value
        differences = np.abs(arr - target)
        
        # Find the index of the minimum difference
        closest_index = np.argmin(differences)
        
        return closest_index