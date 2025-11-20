import numpy as np
import matplotlib.pyplot as plt
import sys

"""
    Trying something prettier to showcase the parameter differencces

"""

plt.style.use('perf.mplstyle')

# right now I only have the reproduced DESI DR1 and the base DESI DR1 but the
# base DESI DR1 will be shown with fill between, so first on the array


alpha_perp = np.array([0.993, 0.991, 0.993])
e_alpha_perp_l = np.array([0.032, 0.028, 0.029])
e_alpha_perp_h = np.array([0.029, 0.028, 0.029])
alpha_par = np.array([1.020, 1.020, 1.021])
e_alpha_par_l = np.array([0.037, 0.034, 0.036])
e_alpha_par_h = np.array([0.036, 0.034, 0.036])
b_lya = np.array([-0.1078, -0.1080, -0.1077])
e_b_lya_l = np.array([0.0036, 0.00354, 0.0034])
e_b_lya_h = np.array([0.0036, 0.00354, 0.0034])
beta_lya = np.array([1.745, 1.736, 1.745])
e_beta_lya_l = np.array([0.088, 0.0795, 0.082])
e_beta_lya_h = np.array([0.076, 0.0795, 0.071])
b_hcd = np.array([-0.0582, -0.0579, -0.0582])
e_b_hcd_l = np.array([0.0037, 0.00361, 0.0036])
e_b_hcd_h = np.array([0.0037, 0.00361, 0.0036])
beta_hcd = np.array([0.588, 0.590, 0.587])
e_beta_hcd_l = np.array([0.081, 0.0799, 0.080])
e_beta_hcd_h = np.array([0.082, 0.0799, 0.080])
L_hcd = np.array([6, 6, 6]) # dummy
e_L_hcd_l = np.array([0.3, 0.3, 0.3]) # dummy
e_L_hcd_h = np.array([0.3, 0.3, 0.3]) # dummy
b_SiII_1190 = np.array([-5.53, -5.42, -5.47])
e_b_SiII_1190_l = np.array([0.82, 0.804, 0.81])
e_b_SiII_1190_h = np.array([0.85, 0.804, 0.81])
b_SiII_1193 = np.array([-4.16, -4.25, -4.24])
e_b_SiII_1193_l = np.array([0.79, 0.773, 0.76])
e_b_SiII_1193_h = np.array([0.81, 0.773, 0.76])
b_SiII_1260 = np.array([-4.63, -4.55, -4.59])
e_b_SiII_1260_l = np.array([0.91, 0.891, 0.88])
e_b_SiII_1260_h = np.array([0.91, 0.891, 0.88])
b_SiIII_1207 = np.array([-10.80, -10.8, -10.86])
e_b_SiIII_1207_l = np.array([0.75, 0.726, 0.72])
e_b_SiIII_1207_h = np.array([0.74, 0.726, 0.72])
b_CIV = np.array([-24.5, -24.4, -24.5])
e_b_CIV_l = np.array([1.6, 1.50, 1.50])
e_b_CIV_h = np.array([1.6, 1.50, 1.50])

# models: no reio, reio
x = [0,1,2]
markers = ['x', 'd', 'p']
labels = ['Off.', 'Fid. Minimizer', 'Fid. Polychord']
colors = ['cyan', 'green', 'indigo']

# axes labels
axlab = [r'$\alpha_\perp$', r'$\alpha_\parallel$', r'$b_{Ly\alpha}$', r'$\beta_{Ly\alpha}$', r'$10^3b_\mathrm{SiII(1190)}$', r'$10^3b_\mathrm{SiII(1193)}$', r'$10^3b_\mathrm{SiII(1260)}$', r'$10^3b_\mathrm{SiIII(1207)}$', r'$10^3b_\mathrm{CIV(eff)}$', r'$b_\mathrm{HCD}$', r'$\beta_\mathrm{HCD}$', r'$L_\mathrm{HCD}$']

fig, axes = plt.subplots(3, 4, figsize=(6.5*2, 4.7*2))


xfil = np.linspace(0,3,100)

for i in range(0,3):
    for j in range(0,4):
        axes[i,j].set_ylabel(axlab[j + i*4])
        axes[i,j].set_xticklabels([])
        axes[i,j].set_xticks([])
        axes[i,j].set_xlim(0,3)
   
for k in range(1,3):
    axes[0,0].errorbar(x[k], alpha_perp[k], xerr=0, yerr=np.array([[e_alpha_perp_l[k], e_alpha_perp_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)

    axes[0,1].errorbar(x[k], alpha_par[k], xerr=0, yerr=np.array([[e_alpha_par_l[k], e_alpha_par_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)
    
    axes[0,2].errorbar(x[k], b_lya[k], xerr=0, yerr=np.array([[e_b_lya_l[k], e_b_lya_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)

    axes[0,3].errorbar(x[k], beta_lya[k], xerr=0, yerr=np.array([[e_beta_lya_l[k], e_beta_lya_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)
    
    axes[1,0].errorbar(x[k], b_SiII_1190[k], xerr=0, yerr=np.array([[e_b_SiII_1190_l[k], e_b_SiII_1190_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)
    
    axes[1,1].errorbar(x[k], b_SiII_1193[k], xerr=0, yerr=np.array([[e_b_SiII_1193_l[k], e_b_SiII_1193_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)
    
    axes[1,2].errorbar(x[k], b_SiII_1260[k], xerr=0, yerr=np.array([[e_b_SiII_1260_l[k], e_b_SiII_1260_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)
    
    axes[1,3].errorbar(x[k], b_SiIII_1207[k], xerr=0, yerr=np.array([[e_b_SiIII_1207_l[k], e_b_SiIII_1207_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)
    
    axes[2,0].errorbar(x[k], b_CIV[k], xerr=0, yerr=np.array([[e_b_CIV_l[k], e_b_CIV_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)

    axes[2,1].errorbar(x[k], b_hcd[k], xerr=0, yerr=np.array([[e_b_hcd_l[k], e_b_hcd_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)
    
    axes[2,2].errorbar(x[k], beta_hcd[k], xerr=0, yerr=np.array([[e_beta_hcd_l[k], e_beta_hcd_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)
    
#    axes[2,3].errorbar(x[k], L_hcd[k], xerr=0, yerr=np.array([[e_L_hcd_l[k], e_L_hcd_h[k]]]).T, marker=markers[k], markersize=7, capthick=3, capsize=4, color=colors[k], label=labels[k], lw=2)
    

axes[0,0].fill_between(xfil, alpha_perp[0] - e_alpha_perp_l[0], alpha_perp[0] + e_alpha_perp_h[0], color='cyan', alpha=0.1)
axes[0,0].axhline(alpha_perp[0], color='darkgray', linestyle='--', lw=2)


axes[0,1].fill_between(xfil, alpha_par[0] - e_alpha_par_l[0], alpha_par[0] + e_alpha_par_h[0], color='cyan', alpha=0.1)
axes[0,1].axhline(alpha_par[0], color='darkgray', linestyle='--', lw=2)

axes[0,2].fill_between(xfil, b_lya[0] - e_b_lya_l[0], b_lya[0] + e_b_lya_h[0], color='cyan', alpha=0.1)
axes[0,2].axhline(b_lya[0], color='darkgray', linestyle='--', lw=2)


axes[0,3].fill_between(xfil, beta_lya[0] - e_beta_lya_l[0], beta_lya[0] + e_beta_lya_h[0], color='cyan', alpha=0.1)
axes[0,3].axhline(beta_lya[0], color='darkgray', linestyle='--', lw=2)


axes[1,0].fill_between(xfil, b_SiII_1190[0] - e_b_SiII_1190_l[0], b_SiII_1190[0] + e_b_SiII_1190_h[0], color='cyan', alpha=0.1)
axes[1,0].axhline(b_SiII_1190[0], color='darkgray', linestyle='--', lw=2)


axes[1,1].fill_between(xfil, b_SiII_1193[0] - e_b_SiII_1193_l[0], b_SiII_1193[0] + e_b_SiII_1193_h[0], color='cyan', alpha=0.1)
axes[1,1].axhline(b_SiII_1193[0], color='darkgray', linestyle='--', lw=2)


axes[1,2].fill_between(xfil, b_SiII_1260[0] - e_b_SiII_1260_l[0], b_SiII_1260[0] + e_b_SiII_1260_h[0], color='cyan', alpha=0.1)
axes[1,2].axhline(b_SiII_1260[0], color='darkgray', linestyle='--', lw=2)


axes[1,3].fill_between(xfil, b_SiIII_1207[0] - e_b_SiIII_1207_l[0], b_SiIII_1207[0] + e_b_SiIII_1207_h[0], color='cyan', alpha=0.1)
axes[1,3].axhline(b_SiIII_1207[0], color='darkgray', linestyle='--', lw=2)


axes[2,0].fill_between(xfil, b_CIV[0] - e_b_CIV_l[0], b_CIV[0] + e_b_CIV_h[0], color='cyan', alpha=0.1)
axes[2,0].axhline(b_CIV[0], color='darkgray', linestyle='--', lw=2)


axes[2,1].fill_between(xfil, b_hcd[0] - e_b_hcd_l[0], b_hcd[0] + e_b_hcd_h[0], color='cyan', alpha=0.1)
axes[2,1].axhline(b_hcd[0], color='darkgray', linestyle='--', lw=2)


axes[2,2].fill_between(xfil, beta_hcd[0] - e_beta_hcd_l[0], beta_hcd[0] + e_beta_hcd_h[0], color='cyan', alpha=0.1)
axes[2,2].axhline(beta_hcd[0], color='darkgray', linestyle='--', lw=2)
#axes[2,2].axhline(beta_hcd[0] - e_beta_hcd_l[0], color='cyan', linestyle='--', lw=2)
#axes[2,2].axhline(beta_hcd[0] + e_beta_hcd_h[0], color='cyan', linestyle='--', lw=2)


#axes[2,3].fill_between(xfil, L_hcd[0] - e_L_hcd_l[0], L_hcd[0] + e_L_hcd_h[0], color='cyan', alpha=0.1)
#axes[2,3].axhline(L_hcd[0], color='darkgray', linestyle='--', lw=2)

axes[2,3].axis('off')
handles, labels = axes[2,2].get_legend_handles_labels()
axes[2,3].legend(handles, labels, loc='center', fontsize=14)


plt.tight_layout()
plt.savefig('params.pdf')

sys.exit()

# later we will put legend somewhere

# handles
handles, p_labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax1.set_ylabel(r'$b_{Ly\alpha}$')
ax1.set_xticklabels([])
ax1.set_xticks([])
ax1.axhline(y=b_lya[0]+b_lya_e[0], color='darkgray', linestyle='--', lw=2)
ax1.text(x=2.5, y=b_lya[0]+b_lya_e[0], s=r'1 $\sigma$', va='center', ha='center', backgroundcolor='white', color='darkgray')

# later we might need to make something similar to

#ax1.text(x=2.5, y=b_lya[0]+b_lya_e[0], s=r'1$\sigma$', va='center', ha='center')
ax1.axhline(y=b_lya[0]+3*b_lya_e[0], color='darkgray', linestyle='--', lw=2, alpha= 0.75)
ax1.text(x=2.5, y=b_lya[0]+3*b_lya_e[0], s=r'3 $\sigma$', va='center', ha='center', backgroundcolor='white', color='darkgray', alpha=0.75)
ax1.axhline(y=b_lya[0]+5*b_lya_e[0], color='darkgray', linestyle='--', lw=2, alpha=0.5)
ax1.text(x=2.5, y=b_lya[0]+5*b_lya_e[0], s=r'5 $\sigma$', va='center', ha='center', backgroundcolor='white', color='darkgray', alpha=0.5)
ax1.axhline(y=b_lya[0]+7*b_lya_e[0], color='darkgray', linestyle='--', lw=2, alpha=0.25)
ax1.text(x=2.5, y=b_lya[0]+7*b_lya_e[0], s=r'7 $\sigma$', va='center', ha='center', backgroundcolor='white', color='darkgray', alpha=0.25)
ax1.legend(handles, labels, loc='center left',bbox_to_anchor=(1.,0.5))

plt.savefig('bias_test.pdf')
