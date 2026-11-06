import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from numpy import exp, log
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker
from scipy.integrate import cumtrapz
import sys


"""
    Code for making histories
"""


    



# checking high tau
z = np.linspace(5., 30, 100, endpoint=True)



def x_HI(z_vals):
    b = 0.7
    plt.style.use('mcmc/perf.mplstyle')
#    print(z_vals)
    # z_min = max(float(np.min(early[:, 0])), float(np.min(mid[:, 0])), float(np.min(late[:, 0])), float(np.min(band[:, 0])))
    # z_max = min(float(np.max(early[:, 0])), float(np.max(mid[:, 0])), float(np.max(late[:, 0])), float(np.max(band[:, 0])))
    z_min = 5.3
    z_max = 15.0
    z_plot = np.asarray(z_vals, dtype=float)
    z_plot = z_plot[(z_plot >= z_min) & (z_plot <= z_max)]
    if z_plot.size == 0:
        raise ValueError(f"No z values left after applying overlap cut: {z_min} <= z <= {z_max}")
    # checking high tau options
    fig = plt.figure(figsize=(6.5, 4.7))
    ax1 = fig.add_subplot(111)
#    ax1.axhline(y=0.5, color='darkgray', linestyle='--', linewidth=2)
    # ax1.hlines(y=0.5, xmin=5.3, xmax=11.5, color='darkgray', linestyle='--', linewidth=2)

    # ax1.plot(z_plot, xHI_early(z_plot), '-', lw=3, color='darkgreen', label=r'Early', zorder=7)
    # ax1.plot(z_plot, xHI_mid(z_plot), '-.', lw=3, color='darkblue', label=r'Mid', zorder=7)
    # ax1.plot(z_plot, xHI_late(z_plot), '--', lw=3, color='purple', label=r'Late', zorder=7)
    ax1.fill_between(z_plot, xHI_band_low(z_plot), xHI_band_high(z_plot), alpha=0.25, color='black', linewidth=0.0, zorder=5)
    ax1.plot(z_plot, xHI_band_center(z_plot), '-', lw=2, color='black', label=r'DESI DR1', zorder=6)
    
#    ax1.plot(z_vals, xHI_s8(z_vals), '-', lw=3, color='black', label=r's8', zorder=7)
#    ax1.plot(z_vals, xHI_splus(z_vals), '-.', lw=3, color='red', label=r'splus', zorder=7)
#    ax1.plot(z_vals, xHI_sminus(z_vals), '--', lw=3, color='orange', label=r'sminus', zorder=7)


    # dark fraction
    ax1.errorbar(6.3,0.79,0.04,uplims=True,marker='o', c='gold', markersize=7, ls='none', label=r'Dark Pixel',zorder=6, alpha=b)
    ax1.errorbar(6.5,0.87,0.03,uplims=True,marker='o', c='gold', markersize=7, ls='none', zorder=6, alpha=b)
    ax1.errorbar(6.7,0.94,0.09,uplims=True,marker='o', c='gold', markersize=7, ls='none', zorder=6, alpha=b)
    ax1.errorbar(6.1,0.69,0.06,uplims=True,marker='o', c='gold', markersize=7, ls='none', zorder=6, alpha=b)

    # dark fraction/pixel 2025
    ax1.errorbar(5.481,0.030,0.048,uplims=True,marker='o', c='gold', markersize=7, ls='none',zorder=6, alpha=b)
    ax1.errorbar(5.654,0.095,0.037,uplims=True,marker='o', c='gold', markersize=7, ls='none',zorder=6, alpha=b)
    ax1.errorbar(5.831,0.191,0.056,uplims=True,marker='o', c='gold', markersize=7, ls='none',zorder=6, alpha=b)
    ax1.errorbar(6.043,0.199,0.087,uplims=True,marker='o', c='gold', markersize=7, ls='none',zorder=6, alpha=b)

    # dark fraction/pixel 2015
    ax1.errorbar(5.58,0.04,0.05,uplims=True,marker='o', c='gold', markersize=7, ls='none',zorder=6, alpha=b)
    ax1.errorbar(5.87,0.06,0.05,uplims=True,marker='o', c='gold', markersize=7, ls='none',zorder=6, alpha=b)
    ax1.errorbar(6.07,0.38,0.20,uplims=True,marker='o', c='gold', markersize=7, ls='none',zorder=6, alpha=b)
    
    # JWST Lya Damping wings
    ax1.errorbar(6.5,0.33,yerr=np.array([[0.27,0.18]]).T,uplims=False,marker='^', c='brown', markersize=7, ls='none', capthick=2,capsize=4, label=r'Galaxy DW', zorder=6, alpha=b)
    ax1.errorbar(9.3,0.64,yerr=np.array([[0.23,0.17]]).T,uplims=False,marker='^', c='brown', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)

    ax1.errorbar(5.0, 0.00, xerr=np.array([[5.0 - 4.512, 0.446]]).T, yerr=np.array([[0.00, 0.12]]).T, uplims=False,marker='^', c='brown', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)
    ax1.errorbar(5.8,0.25, xerr=np.array([[5.8 - 5.503, 6.372 - 5.8]]).T,yerr=np.array([[0.20,0.10]]).T,uplims=False,marker='^', c='brown', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)
    ax1.errorbar(6.5,0.33, xerr=np.array([[1.0, 1.5]]).T,yerr=np.array([[0.27,0.18]]).T,uplims=False,marker='^', c='brown', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)
    ax1.errorbar(7.0,0.65, xerr=np.array([[7.0 - 6.548, 0.483]]).T, yerr=np.array([[0.35,0.27]]).T,uplims=False,marker='^', c='brown', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)
    ax1.errorbar(8.6,1.00, xerr=np.array([[8.6 - 7.509, 9.433 - 8.6]]).T, yerr=np.array([[0.40,0.]]).T,uplims=False,marker='^', c='brown', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)
    ax1.errorbar(9.3,0.64, xerr=np.array([[1.3, 1.3]]).T, yerr=np.array([[0.23,0.17]]).T,uplims=False,marker='^', c='brown', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)
    ax1.errorbar(10.4,1.00, xerr=np.array([[10.4 - 9.570, 2.07]]).T, yerr=np.array([[0.40,0.]]).T,uplims=False,marker='^', c='brown', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)
    
    # Islands, Lyb forest dark gaps
#    ax1.errorbar(6.55, 0.05, 0.04, uplims=True, lolims=False, marker='o', c='black', markersize=7, ls='none', label=r'Lyb forest dark gaps', zorder=8, alpha=b)
#    ax1.errorbar(6.75, 0.17, 0.05, uplims=True,  marker='o', c='black', markersize=7, ls='none', zorder=8, alpha=b)
#    ax1.errorbar(6.95, 0.29, 0.09, uplims=True, marker='o', c='black', markersize=7, ls='none', zorder=8, alpha=b)
    # this accounts for error bar
#    ax1.errorbar(5.55, 0.05+0.04, 0.04, uplims=True, marker='o', c='black', markersize=7, ls='none', label=r'Lyb forest dark gaps', zorder=8, alpha=b)
#    ax1.errorbar(5.75, 0.17+0.05, 0.05, uplims=True,  marker='o', c='black', markersize=7, ls='none', zorder=8, alpha=b)
#    ax1.errorbar(5.95, 0.29+0.09, 0.09, uplims=True, marker='o', c='black', markersize=7, ls='none', zorder=8, alpha=b)
    
    # Lya fraction
    ax1.errorbar(6.9,0.4,0.1,uplims=False,lolims=True,marker='s',markersize=7, ls='none', label=r'Ly$\alpha$ Fraction',zorder=6, alpha=b)
    # LAE clustering
    ax1.errorbar(6.6,0.5,0.1,uplims=True,lolims=False,marker='*',markersize=7, ls='none', label=r'LAE Clustering',zorder=6, alpha=b)

    # Damping wing constraints
    ax1.errorbar(7.29,0.49, yerr=np.array([[0.11,0.11]]).T, uplims=False,marker='p', c='green', markersize=7,capthick=2,capsize=4, ls='none', label=r'QSO DW',zorder=6, alpha=b)
    ax1.errorbar(6.15, 0.20, yerr=np.array([[0.12,0.14]]).T, uplims=False,marker='p', c='green', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)
    ax1.errorbar(6.35, 0.29, yerr=np.array([[0.13,0.14]]).T, uplims=False,marker='p', c='green', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)
    ax1.errorbar(5.60, 0.19, yerr=np.array([[0.16,0.11]]).T, uplims=False,marker='p', c='green', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)
    ax1.errorbar(6.10, 0.21, yerr=np.array([[0.07,0.17]]).T, uplims=False,marker='p', c='green', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)
    ax1.errorbar(6.46, 0.21, yerr=np.array([[0.07,0.33]]).T, uplims=False,marker='p', c='green', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)
    ax1.errorbar(6.87, 0.37, yerr=np.array([[0.17,0.17]]).T, uplims=False,marker='p', c='green', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)
    # Luminosity function constraints
    ax1.errorbar(6.60, 0.08, yerr=np.array([[0.05,0.08]]).T, uplims=False,marker='H', c='violet', markersize=7,capthick=2,capsize=4, ls='none', label=r'LF',zorder=6, alpha=b)
    ax1.errorbar(7.00, 0.28, yerr=np.array([[0.05,0.05]]).T, uplims=False,marker='H', c='violet', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)
    ax1.errorbar(7.30, 0.83, yerr=np.array([[0.07,0.06]]).T, uplims=False,marker='H', c='violet', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)

    # JWST enabled Lya LF
    ax1.errorbar(5.90, 0.17, xerr=np.array([[0.4, 0.49]]).T, yerr=np.array([[0.16, 0.23]]).T, uplims=False,marker='H', c='violet', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)
    ax1.errorbar(6.96, 0.63, xerr=np.array([[6.96 - 6.54, 7.49 - 6.96]]).T, yerr=np.array([[0.28, 0.18]]).T, uplims=False,marker='H', c='violet', markersize=7, ls='none', capthick=2, capsize=4, zorder=6, alpha=b)
    ax1.errorbar(8.41, 0.79, xerr=np.array([[0.9, 1.02]]).T, yerr=np.array([[0.21, 0.13]]).T, uplims=False,marker='H', c='violet', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)
    ax1.errorbar(11.0, 0.88, xerr=np.array([[11. - 9.62, 3.18]]).T, yerr=np.array([[0.13, 0.11]]).T, uplims=False,marker='H', c='violet', markersize=7, ls='none', capthick=2,capsize=4, zorder=6, alpha=b)

    # Equivalent width constraints (not used in astro data)
    ax1.errorbar(7.0, 0.59, yerr=np.array([[0.15,0.11]]).T,xerr=0.5, uplims=False, marker='d', c='cyan', markersize=7,capthick=2,capsize=4, ls='none', label=r'Ly$\alpha$ EW',zorder=6, alpha=b)
    ax1.errorbar(7.6, 0.88,yerr=np.array([[0.10,0.05]]).T,xerr=0.6, uplims=False,marker='d', c='cyan', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)
    ax1.errorbar(8.0, 0.76, yerr=0.22,xerr=0.6, uplims=False,lolims=True,marker='d', c='cyan', markersize=7,capthick=2,capsize=4, ls='none',zorder=6, alpha=b)
    ax1.set_ylabel(r'Neutral hydrogen fraction', fontsize=16)
    ax1.set_xlim(5.3, 15)
    ax1.set_ylim(0, 1)
    ax1.errorbar(8.0469818, 0.5, xerr=np.array([[8.0469818 - 7.4377289, 8.6358795 - 8.0469818]]).T, uplims=False,marker='X', c='red', label='Planck PR4', markersize=7,capthick=2,capsize=4, ls='none',zorder=8, alpha=b)
    # y_mask = np.linspace(0, 1.1, 200)
    # deBelsunce_low = 7.99
    # deBelsunce_high = 9.03
    # Pagano_low = 7.53
    # Pagano_high = 8.75
    # ax1.fill_betweenx(y_mask, deBelsunce_low, deBelsunce_high, color='cyan', alpha=0.1)
    # ax1.fill_betweenx(y_mask, Pagano_low, Pagano_high, color='olive', alpha=0.1)
    
#    ax1.set_xscale('log')
    ax1.set_xlabel(r'Redshift',fontsize=16)
        # checking high tau
#    ax1.legend(loc='center left', bbox_to_anchor=(1.0,0.5), fontsize=12)
    ax1.legend(loc='lower right', fontsize=14)
    plt.tight_layout()
    # checking high tau
    plt.savefig("mcmc/histories.png", dpi=200, facecolor="white", transparent=False)

    

# early = np.loadtxt('mcmc/xH_zeta2_total.dat')
# mid = np.loadtxt('mcmc/xH_fiducial_total.dat')
# late = np.loadtxt('mcmc/xH_zeta1_total.dat')
band = np.loadtxt('mcmc/xhi_extwhole.txt')



# xHI_early = interp1d(early[:,0], early[:,1], kind='linear')
# error_early = interp1d(early[:,0], early[:,2], kind='linear')

# xHI_mid = interp1d(mid[:,0], mid[:,1], kind='linear')
# error_mid = interp1d(mid[:,0], mid[:,2], kind='linear')

# xHI_late = interp1d(late[:,0], late[:,1], kind='linear')
# error_late = interp1d(late[:,0], late[:,2], kind='linear')

if band.ndim != 2 or band.shape[1] < 3:
    raise ValueError("mcmc/xhi_band.txt must have at least 3 columns: z, xHI_q16, xHI_q84")

xHI_band_low = interp1d(band[:, 0], band[:, 1], kind='linear')
xHI_band_high = interp1d(band[:, 0], band[:, 2], kind='linear')
xHI_band_center = interp1d(band[:, 0], band[:, 3], kind='linear')

# print(xHI_early(5.91))


# testing Yao's histories
#s8 = np.loadtxt('xH_21cm_ave_cdm_s8.txt')
#splus = np.loadtxt('xH_21cm_ave_cdm_splus.txt')
#sminus = np.loadtxt('xH_21cm_ave_cdm_sminus.txt')

#xHI_s8 = interp1d(s8[:,0], s8[:,1], kind='linear')
#xHI_splus = interp1d(splus[:,0], splus[:,1], kind='linear')
#xHI_sminus = interp1d(sminus[:,0], sminus[:,1], kind='linear')

# x_HI(z)

if __name__ == "__main__":
    x_HI(z)
