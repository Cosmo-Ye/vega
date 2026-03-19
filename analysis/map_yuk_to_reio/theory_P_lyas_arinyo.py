import numpy as np
from scipy import integrate
from scipy.interpolate import interp2d
from scipy.misc import derivative
from scipy import interpolate
import Pm_DM as pm
import pickle



"""
    calculate the 3D lyman alpha forest power spectrum:
    P_F^3D(z,k_Mpc,mu) = LyaLya_base_Mpc_norm(z, k_Mpc, mu) + LyaLya_reio_Mpc_norm(z, k_Mpc, mu),
    where LyaLya_base_Mpc_norm() is the first term on the RHS of Equation (6),
    and LyaLya_reio_Mpc_norm() is the second term on the RHS of Equation (6), which is the leading-order term of the impact of reionization on the 3D lyman alpha forest power spectrum.
    
    Now using Andreu Arinyo-i-Prats et al. (arXiv:1506.04519) results to compute flux bias b_F and RSD parameter beta_F in Equation (6)
"""

class theory_P_lyas(object):
    # need to massage Arinyo results first

    def __init__(self,params):
        # need to define cosmology for hubble parameter, this is the same cosmology for everyone
        self.h = params['h']
        self.Omega_b = params['Obh2'] / self.h**2
        self.Omega_m = params['Och2'] / self.h**2
        """ need to deal with sigma8, for the moment let's just say it is fiducial """
        self.gadget_realization = params['gadget-realization']
        self.gadget_model = params['gadget-model']
        self.fast_realization = params['fast-realization']
        self.fast_model = params['fast-model']
        self.our_sigma8 = params['sigma8']
        # so, the redshifts for arinyo are the following
        self.z_arinyo = [2.2, 2.4, 2.6, 2.8, 3.0]
        # the sigma8 arrays
        self.sigma8 = [0.64, 0.76, 0.8778] #note that our sigma 8 is 0.8159
        self.beta_rsd_zs = [1.163, 1.284, 1.405, 1.137, 1.257, 1.385, 1.094, 1.219, 1.343, 1.035, 1.161, 1.284, 0.965, 1.086, 1.205]
        self._beta_rsd = interp2d(self.z_arinyo, self.sigma8, self.beta_rsd_zs)
        # bias, this one needs to be converted into a flux bias
        self.b_tau_delta_zs = [0.7490, 0.6401, 0.5536, 0.7478, 0.6428, 0.5574, 0.7429, 0.6409, 0.5588, 0.7366, 0.6373, 0.5577, 0.7287, 0.6319, 0.5546]
        self.b_tau_delta = interp2d(self.z_arinyo, self.sigma8, self.b_tau_delta_zs)
        # reference
        self.z_ref = 3.0
        # the observed array
        self.z_obs_array = [2.0, 2.5, 3.0, 3.5, 4.0]
        # for memory of reionization
        bg = np.loadtxt('../data/gadget/transp_gadget_'+self.gadget_realization+'_'+self.gadget_model+'_f.txt', usecols=[7])
        self.b_gamma = interpolate.interp1d(self.z_obs_array, bg, fill_value="extrapolate")
        # let's unpickle the pickles
        f_pat = open('../pickles/pat_'+self.fast_realization+'_'+self.fast_model+'.pkl', 'rb')
        # this is actually a power and not a density!
        self.PmxH = pickle.load(f_pat)
        f_pat.close()
        f_psi = open('../pickles/psi_'+self.gadget_realization+'_'+self.gadget_model+'.pkl', 'rb')
        self.psi = pickle.load(f_psi)
        f_psi.close()
        # some extra redshifts for help
        self.z_res = np.arange(6, 12+0.1, 0.01)
        """ Please comment out the cosmo and the self-pickle if running new pickles """
        # need to unpickle my own result here
        if params['pickle'] == True:
            self._crosspower_psi = 1.
            self.cosmo = 1.
        else:
            self.cosmo = pm.P_matter(params)
            flya = open('../pickles/p_mpsi_'+self.fast_realization+'_'+self.fast_model+'_'+self.gadget_realization+'_'+self.gadget_model+'.pkl', 'rb')
            self._crosspower_psi = pickle.load(flya)
            flya.close()
        
        
        
        
    # Required functions, which need to depend on model that is being called
    def log_observed_Flux(self,z):
        """ Mean observed flux, useful to get the flux bias """
        return -0.0023 * pow((1 + z),3.65)
        
    def flux_bias(self,z,sigma_8):
        """ Obtains the flux bias from arinyo tables """
        if z <= 3.0:
            return self.b_tau_delta(z,sigma_8) * self.log_observed_Flux(z)
        else:
            return self.b_tau_delta(3.0,sigma_8) * self.log_observed_Flux(3.0)
            
    def beta_rsd(self, z, sigma8):
        """ forces rsd to not evolve after z larger than 3 """
        if z <= 3.:
            return self._beta_rsd(z, sigma8)
        else:
            return self._beta_rsd(3.0, sigma8)
    
    
    def dpsi_dz(self, z):
        dpsi_dzre = np.gradient(self.psi(self.z_res, z), self.z_res)
        dpsi_dzre[np.isnan(dpsi_dzre)] = 0.
        return dpsi_dzre
    
    
    def P_m_psi(self, z, k):
        # z is the zobs!
        dpsi_dz = self.dpsi_dz(z)
        # because Heyang did it, I am also using D prop to a to be consistent here
        Pmpsi = -1. * integrate.simps(dpsi_dz * self.PmxH(self.z_res, k) * (np.ones(self.z_res.shape) + self.z_res) / (1. + z), self.z_res)
        return Pmpsi
        
    def pickle_this(self):
        # let's pickle the memory term
        file = open('../pickles/p_mpsi_'+self.fast_realization+'_'+self.fast_model+'_'+self.gadget_realization+'_'+self.gadget_model+'.pkl', 'wb')
        pickle.dump(self.P_m_psi, file)
        file.close()
        print('PickleD!')
        return
        
    def P1D_z_kms_PD2013(self,z,k_kms):
        """ Fitting formula from Palanque-Delabrouille et al. (2013).
            Note the funny units for both k and P """
            # this function is mainly for computing the aliasing term of the noise.
        A_F = 0.064
        n_F = -2.55
        alpha_F = -0.1
        B_F = 3.55
        beta_F = -0.28
        k0 = 0.009
        z0 = 3.0
        n_F_z = n_F + beta_F * np.log((1.0 + z) / (1.0 + z0))
        # correction to cut low k behaviour
        k_min = k0*np.exp((-0.5 * n_F_z - 1.0) / alpha_F)
        k_kms = np.fmax(k_kms,k_min)
        exp1 = 3.0 + n_F_z + alpha_F * np.log(k_kms / k0)
        toret = np.pi * A_F / k0 * pow(k_kms / k0, exp1 - 1.0) * pow((1.0 + z) / (1.0 + z0), B_F)
        return toret
        
    def LyaLya_base_Mpc_norm(self, z, k_Mpc, mu):
        """ 3D base Lya flux power spectrum with redshift evolution (after z=3) with a trick, k must be in Mpc^-1 """
        bias = self.flux_bias(z, self.our_sigma8)
        beta = self.beta_rsd(z, self.our_sigma8)
        if z<=3.0:
            Kaiser = pow(bias * (1. + beta * mu**2),2)
            PF3D = self.cosmo.P_m_Mpc(k_Mpc, z) * Kaiser
        else:
            Kaiser = pow(bias * (1. + beta * mu**2),2)
            z_evol = pow((1.0 + z) / (1.0 + self.z_ref), 3.55) # got this fr
            PF3D = self.cosmo.P_m_Mpc(k_Mpc, 3.0) * Kaiser * z_evol
        return PF3D

    def memory_bias(self, z, k_Mpc, mu):
        bias = self.flux_bias(z, self.our_sigma8)
        beta = self.beta_rsd(z, self.our_sigma8)
        if z<=3.0:
            bias_fac =  bias * (1. + beta * mu**2)
        else:
            bias_fac = bias * (1. + beta * mu**2)
            z_evol = pow((1.0 + z) / (1.0 + self.z_ref), 3.55)
            bias_fac = bias_fac * np.sqrt(z_evol) * np.sqrt(self.cosmo.P_m_Mpc(k_Mpc, 3.0) / self.cosmo.P_m_Mpc(k_Mpc, z))
        return bias_fac * self.b_gamma(z)

    def LyaLya_reio_Mpc_norm(self, z, k_Mpc, mu):
        """ reio term in LyaLya auto-power spectrum """
        """ I should probably move this out of here to get things easier when generating pickles,
        instead of calling my own pickle...
        The reason this is here is because it is called in the observed class.
        """
        bias_mem = self.memory_bias(z, k_Mpc, mu)
        return 2. * bias_mem * self._crosspower_psi(z, k_Mpc)

    def LyaLya_monopole(self, z, k_Mpc):
        """ Monopole of the 3D Lya power spectrum """
        # need to integrate over mu
        mu = np.linspace(-1, 1, 100)
        integrand = np.zeros(mu.shape)
        for i in range(mu.shape[0]):
            integrand[i] = self.LyaLya_base_Mpc_norm(z, k_Mpc, mu[i]) + self.LyaLya_reio_Mpc_norm(z, k_Mpc, mu[i])
        return 0.5*integrate.simps(integrand, mu)
    
    def LyaLya_quadrupole(self, z, k_Mpc):
        """ Quadropole of the 3D Lya power spectrum """
        # need to integrate over mu
        mu = np.linspace(-1, 1, 100)
        integrand = np.zeros(mu.shape)
        for i in range(mu.shape[0]):
            integrand[i] = (self.LyaLya_base_Mpc_norm(z, k_Mpc, mu[i]) + self.LyaLya_reio_Mpc_norm(z, k_Mpc, mu[i])) * 0.5* (3. * mu[i]**2 - 1.)
        return 2.5*integrate.simps(integrand, mu)
    
    def LyaLya_hexadecapole(self, z, k_Mpc):
        """ Hexadecapole of the 3D Lya power spectrum """
        # need to integrate over mu
        mu = np.linspace(-1, 1, 100)
        integrand = np.zeros(mu.shape) 
        for i in range(mu.shape[0]):
            integrand[i] = (self.LyaLya_base_Mpc_norm(z, k_Mpc, mu[i]) + self.LyaLya_reio_Mpc_norm(z, k_Mpc, mu[i])) * 0.125 * (35. * mu[i]**4 - 30. * mu[i]**2 + 3.)
        return 4.5*integrate.simps(integrand, mu)
            

