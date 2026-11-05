import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit
import sys
from scipy.interpolate import interp1d
from getdist import loadMCSamples
from scipy.interpolate import RectBivariateSpline

"""
    Code for displaying some P_m,xHI
"""


plt.style.use('../perf.mplstyle')


kbin = np.loadtxt('cross_21cm_ave_cdm_s8.txt', usecols=(1), unpack=True)
kbin = np.unique(kbin)
kbin = kbin[2:-2]
knum = len(kbin)


file = open('p_mpsi_r1_3keV_splus_zminus_mhigh_r1_3keV_splus.pkl', 'rb')
t3keV_mhigh = pickle.load(file)
file.close()
pmpsi_3keVmhigh = np.zeros(knum)
z_obs = 2.33
for j in range(knum):
    pmpsi_3keVmhigh[j] = t3keV_mhigh(z_obs, kbin[j])

def Yukawa(k, A, alpha, beta):
    return -1.0 * A * np.exp(-1. * alpha * k) * pow(k, -1. * beta)



# using Polychord now, but since the errors are asymmetric, we have no other option than
# to directly use the samples


dir2 = '/Users/paulo/Documents/DESI-DR1-mem/poly/lya_auto_yuk3_extendedprior/Polychord_auto_extendprior'
s2 = loadMCSamples(dir2, settings={'ignore_rows': 0.2})

samples2 = s2.samples
A2 = samples2[:, s2.index['yuk_A']]
alpha2 = samples2[:, s2.index['yuk_alpha']]
beta2 = samples2[:, s2.index['yuk_beta']]

kk = np.asarray(kbin[2:])

F2 = -A2[:, None] * np.exp(-alpha2[:, None] * kk[None, :]) * kk[None, :]**(-beta2[:, None])

F2_med = np.percentile(F2, 50, axis=0)
F2_lo = np.percentile(F2, 16, axis=0)
F2_hi = np.percentile(F2, 84, axis=0)

dir3 = '/Users/paulo/Documents/DESI-DR1-mem/poly/full_combine_yuk3_extendedprior/Polychord_fullcombined_extendprior'
s3 = loadMCSamples(dir3, settings={'ignore_rows': 0.2})

samples3 = s3.samples
A3 = samples3[:, s3.index['yuk_A']]
alpha3 = samples3[:, s3.index['yuk_alpha']]
beta3 = samples3[:, s3.index['yuk_beta']]

F3 = -A3[:, None] * np.exp(-alpha3[:, None] * kk[None, :]) * kk[None, :]**(-beta3[:, None])

F3_med = np.percentile(F3, 50, axis=0)
F3_lo = np.percentile(F3, 16, axis=0)
F3_hi = np.percentile(F3, 84, axis=0)

print('######\n')
print('Mean yuk3, yuk3 extended prior, and full combined yuk3 extended prior', F2_med[0], F3_med[0])
print('16% ', F2_lo[0], F3_lo[0])
print('84% ', F2_hi[0], F3_hi[0])
print('######\n')



"""
    Let's do the different realizations
"""

zin = np.array([2., 2.5, 3., 3.5, 4.])
kin = [6.212050e-02, 7.542991e-02, 9.031192e-02, 1.082159e-01, 1.290225e-01, 1.547179e-01, 1.857322e-01, 2.232069e-01, 2.680392e-01, 3.214029e-01, 3.854592e-01, 4.623636e-01, 5.550113e-01, 6.660092e-01, 7.991547e-01, 9.591026e-01, 1.150977e+00, 1.381152e+00]

#T1_r1 = np.loadtxt('psi_T3_r1.txt')
#T1r1 = RectBivariateSpline(zin, kin, T1_r1)
#T1_r2 = np.loadtxt('psi_T3_r2.txt')
#T1r2 = RectBivariateSpline(zin, kin, T1_r2)
#T1_r3 = np.loadtxt('psi_T3_r3.txt')
#T1r3 = RectBivariateSpline(zin, kin, T1_r3)
#T1_r4 = np.loadtxt('psi_T3_r4.txt')
#T1r4 = RectBivariateSpline(zin, kin, T1_r4)

#z1_r1 = np.loadtxt('psi_zeta1_r1.txt')
#z1r1 = RectBivariateSpline(zin, kin, z1_r1)

z15_r1 = np.loadtxt('psi_zeta15_r1.txt')
z15r1 = RectBivariateSpline(zin, kin, z15_r1)

z18_r1 = np.loadtxt('psi_zeta18_r1.txt')
z18r1 = RectBivariateSpline(zin, kin, z18_r1)



# let's help me visualize first
#fig = plt.figure(figsize=(6.5, 4.7))
fig = plt.figure(figsize=(4.7, 3))
ax = fig.add_subplot(111)
cal = 'blue'
cal2 = 'olive'
cal3 = 'purple'
cal4 = 'red'

ax.fill_between(kbin[2:], F2_hi, F2_lo, color=cal, alpha=0.25, label=r'Auto Ly$\alpha$ DESI DR1')
ax.fill_between(kbin[2:], F3_hi, F3_lo, color=cal2, alpha=0.25, label=r'Full DESI DR1')
ax.plot(kbin[2:], Yukawa(kbin[2:], 1.5050, 3.0237, 1.8354), '--', label='Late')
ax.plot(kbin[2:], Yukawa(kbin[2:], 0.4775, 1.5094, 2.1774), label='Planck')
ax.plot(kbin[2:], Yukawa(kbin[2:], 0.2088, 0.7239, 2.3656), '-.', label='Early')

#ax.plot(kbin[2:], T1r1(2.33, kbin[2:])[0], ':')
#ax.plot(kbin[2:], T1r2(2.33, kbin[2:])[0], ':')
#ax.plot(kbin[2:], T1r3(2.33, kbin[2:])[0], ':')
#ax.plot(kbin[2:], T1r4(2.33, kbin[2:])[0], ':')
#ax.plot(kbin[2:], z1r1(2.33, kbin[2:])[0], '--')


#ax.plot(kbin[2:], z15r1(2.33, kbin[2:])[0], '-.', label='Ultra late')
ax.plot(kbin[2:], pmpsi_3keVmhigh[2:], ':', label='A 3keV WDM example')
ax.plot(kbin[2:], z18r1(2.33, kbin[2:])[0], '-.', label='Ultra late')
ax.set_xlim(0.072, 0.3)
ax.set_xlabel(r'$k$    [Mpc$^{-1}$]')
ax.set_ylabel(r'$P_{m,\psi} (k, z_{\mathrm{obs}} = 2.33)$    [Mpc$^3$]')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('pmpsi_reio.pdf')

