import numpy as np
from classy import Class
from scipy.interpolate import interp1d

"""
    Build the matter power spectrum using classy
"""


class P_matter(object):
    
    def __init__(self, params):
        # now time to build class model
        # Define your cosmology (what is not specified will be set to CLASS default parameters, see explanatory.ini for specific information regarding all parameters)
        # getting wdm stuff
        self.m_WDM_keV = params['m_wdm']
        
        
        self.params = {
            'output': 'mPk',
            'sigma8': params['sigma8'],
            'n_s': 0.9667,
            'h': 0.6774,
            'Omega_b': 0.0486,
            'Omega_cdm': 0.2602,
            'z_reio': 7.93,
            'format': 'CLASS',
            'P_k_max_1/Mpc': params['P_k_max_1/Mpc'],
            'z_max_pk': params['z_max_pk'],

        } # values chosen consistent with planck cosmology, #P_k_max_1/Mpc was 5 before
        
        # Create an instance of the CLASS wrapper, this class we call for solving Boltzmann equations
        self.cosmo = Class()
        # Set the parameters to the cosmological code, i.e. tells the wrapper to pass this dictionary to the c code
        self.cosmo.set(self.params)
        # Run the whole code, i.e. computes P_m(k,z)
        self.cosmo.compute()
        # Let's make a function to get us our new computed P_m, but first let's get little h
        self.h = self.cosmo.h()
        #print('Finished prepping the model for P_m')


    def alpha_WDM(self):
        """
        Returns the suppression scale due to WDM models free streaming more than CDM, in Mpc.

        Inputs (not really since they're class attributes : m_WDM [keV] (mass of the WDM candidate),
                g_WDM (degrees of freedom, i.e. 1.5) but for thermal relics it does nothing, so let's get rid off it
                Omega_DM (dark matter density of the Universe)
                
        Outputs: alpha [Mpc]
        """
        return 0.049 / self.params['h'] * pow(self.m_WDM_keV,-1.11) * pow(self.params['Omega_cdm'] / 0.25,0.11) * pow(self.params['h'] / 0.7,1.22)

    def T_WDM(self, k_Mpc):
        """
        Transfer function of WDM, remember that it is dimensionless

        Inputs: k [Mpc^-1], note that no redshift nor little h in the input

        Outputs: T_WDM
        """
        mu = 1.12
        alpha = self.alpha_WDM()
        pa = 1.0 + pow(alpha * k_Mpc, 2 * mu)
        return pow(pa, -5. / mu)

    def P_m_Mpc(self,k_Mpc,z):
        """
        Returns the 3D matter power spectrum obtained from CLASS in units of Mpc^3 (no little h!). Note that this is a function of redshift too.
        
        Inputs: k [Mpc^-1], z
        
        Outputs: P_m [Mpc^3]
        """
        # so actually pk needs k_Mpc as input and throws P_Mpc, so [Mpc^3] units, no h.
        # note that in case of cdm m_wdm -> infinity, thus transfer function -> 1
        return self.cosmo.pk_lin(k_Mpc, z) * self.T_WDM(k_Mpc) * self.T_WDM(k_Mpc)

    
    def kill_model(self):
        # Clean CLASS
        self.cosmo.struct_cleanup()
        #if you want to completely kill everything then
        self.cosmo.empty()
