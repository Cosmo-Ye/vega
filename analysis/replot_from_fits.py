from vega.postprocess.fit_results import FitResults
import numpy as np
import matplotlib.pyplot as plt
from vega.plots.wedges import Wedge
import sys
from scipy.ndimage import gaussian_filter1d

fit1 = '/Users/zhengyifan/vega_result/auto/fiducial/1209_auto_fiducial.fits'
fit2 = '/Users/zhengyifan/vega_result/auto/yuk_3free/1209_auto_3free.fits'
corr = 'lyaxlya'

fr1 = FitResults(fit1)
fr2 = FitResults(fit2)

c1 = fr1.correlations[corr]
c2 = fr2.correlations[corr]

# 动态读取 RP/RT 并匹配网格
rp = np.asarray(c1.rp)
rt = np.asarray(c1.rt)
rp_vals = np.unique(rp)
rt_vals = np.unique(rt)
N = int(len(c1.data))
n_rp = int(rp_vals.size)
n_rt = int(rt_vals.size)
if n_rp * n_rt != N:
    _, rp_counts = np.unique(rp, return_counts=True)
    n_rt_guess = int(rp_counts.max())
    n_rp_guess = int(N // n_rt_guess) if n_rt_guess > 0 else n_rp
    if n_rp_guess * n_rt_guess == N:
        n_rp, n_rt = n_rp_guess, n_rt_guess
    else:
        _, rt_counts = np.unique(rt, return_counts=True)
        n_rp_guess2 = int(rt_counts.max())
        n_rt_guess2 = int(N // n_rp_guess2) if n_rp_guess2 > 0 else n_rt
        if n_rp_guess2 * n_rt_guess2 == N:
            n_rp, n_rt = n_rp_guess2, n_rt_guess2
        else:
            raise ValueError(f'Grid size mismatch: len(data)={N}, unique RP={int(rp_vals.size)}, unique RT={int(rt_vals.size)}')

rp_min_val, rp_max_val = float(rp_vals.min()), float(rp_vals.max())
if n_rp > 1:
    drp = (rp_max_val - rp_min_val) / (n_rp - 1)
    rp_min = rp_min_val - drp / 2
    rp_max = rp_max_val + drp / 2
else:
    # Fallback if only 1 bin (unlikely)
    rp_min, rp_max = rp_min_val - 2.0, rp_max_val + 2.0

rt_min_val, rt_max_val = float(rt_vals.min()), float(rt_vals.max())
if n_rt > 1:
    drt = (rt_max_val - rt_min_val) / (n_rt - 1)
    rt_min = rt_min_val - drt / 2
    rt_max = rt_max_val + drt / 2
else:
    rt_min, rt_max = 0.0, rt_max_val + 2.0


# 构造 wedge，保证 rt_bins*rp_bins 与数据长度一致
# 定义所需的 mu ranges
mu_bins = [
    (0.95, 1.0),
    (0.8, 0.95),
    (0.5, 0.8),
    (0.0, 0.5)
]

# 准备数据函数
def get_wedge_input(corr_data, n_rp, n_rt):
    expected_size = n_rp * n_rt
    mask = corr_data.model_mask.astype(bool)
    m_valid = corr_data.model[mask]
    
    if m_valid.size != expected_size:
        # 使用 covariance 处理 mask
        m_input = corr_data.model[:expected_size]
        var = corr_data.variance
        var_input = var[:expected_size]
        mask_input = mask[:expected_size]
        # 创建 var_input 的副本以避免修改原始数据（尽管这里是切片，但在某些 numpy 版本或内存布局下可能影响）
        var_input = var_input.copy()
        var_input[~mask_input] = np.inf
        cov = np.diag(var_input)
        return m_input, cov
    else:
        return m_valid, None

m1_input, cov1 = get_wedge_input(c1, n_rp, n_rt)
m2_input, cov2 = get_wedge_input(c2, n_rp, n_rt)

def get_smooth_ratio(y1, y2, x):
    # Use weighted kernel smoothing to avoid singularity at y1=0
    # Instead of smoothing y2/y1 directly, we calculate <y2*y1> / <y1*y1>
    # This is equivalent to a local least-squares estimate of the ratio
    
    sigma = 1.5
    
    # Weighted components
    num = y2 * y1
    den = y1 * y1
    
    # Apply Gaussian smoothing to numerator and denominator
    # This ensures denominator is strictly positive (averaging positive values)
    num_smooth = gaussian_filter1d(num, sigma=sigma)
    den_smooth = gaussian_filter1d(den, sigma=sigma)
    
    # Calculate ratio
    # Add a tiny epsilon to denominator just in case of numerical underflow, 
    # though den_smooth should be positive.
    ratio_smooth = num_smooth / (den_smooth + 1e-30)
    
    return ratio_smooth

# Parse args
mode = 'wedge'
if len(sys.argv) > 1:
    if '-compare' in sys.argv:
        mode = 'compare'
    elif '-wedge' in sys.argv:
        mode = 'wedge'

# 创建 2x2 的子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (mu_min, mu_max) in enumerate(mu_bins):
    ax = axes[i]
    w = Wedge(mu=(mu_min, mu_max), rp=(rp_min, rp_max, n_rp), rt=(rt_min, rt_max, n_rt), r=(0., min(rp_max, 200.), n_rp), abs_mu=True)
    
    # Run 1
    if cov1 is not None:
        r, m1, _ = w(m1_input, covariance=cov1)
    else:
        r, m1 = w(m1_input)
        
    # Run 2
    if cov2 is not None:
        r, m2, _ = w(m2_input, covariance=cov2)
    else:
        r, m2 = w(m2_input)
        
    if mode == 'wedge':
        ax.plot(r, m1*r**2, '-', label='auto fiducial')
        ax.plot(r, m2*r**2, '-', label='auto yukawa 3free')
        ax.set_ylabel(r'$r^2 \xi(r)$')
        if i == 0:
            ax.legend()
            
    elif mode == 'compare':
        ratio = get_smooth_ratio(m1, m2, r)
        ax.plot(r, ratio, 'k-', label='fit2 / fit1 (smoothed)')
        ax.set_ylabel(r'$\xi_{fit2} / \xi_{fit1}$')
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend()
    
    ax.set_title(r'${} < \mu < {}$'.format(mu_min, mu_max))
    ax.set_xlim(10, 180)
    ax.set_xlabel('r [Mpc/h]')
    ax.grid()

plt.tight_layout()
plt.show()
