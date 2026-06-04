"""
Script to replot correlation functions from Vega FITS output files.

Usage:
    python analysis/replot_from_fits.py [mode] [options]

Modes:
    -wedge (default)
        Plot the wedge correlation function r^2 * xi(r) for two fit results.
        Generates 4 subplots for different mu ranges.

    -compare
        Plot the ratio of two correlation functions: xi_fit2 / xi_fit1.
        Sub-options for -compare:
            -four (default) : Show 4 separate subplots, one for each mu range.
            -one            : Show all 4 mu ranges on a single plot with different colors.

Examples:
    python analysis/replot_from_fits.py -wedge
    python analysis/replot_from_fits.py -compare -four
    python analysis/replot_from_fits.py -compare -one
"""

from vega.postprocess.fit_results import FitResults
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./analysis/5par.mplstyle')
from vega.plots.wedges import Wedge
import sys
from scipy.ndimage import gaussian_filter1d
import os

fit1 = None
fit2 = None
corr = 'lyaxlya'
config_path = None


from scipy.interpolate import RectBivariateSpline, griddata

# 构造 wedge，保证 rt_bins*rp_bins 与数据长度一致
# 定义所需的 mu ranges
mu_bins = [
    (0.95, 1.0),
    (0.8, 0.95),
    (0.5, 0.8),
    (0.0, 0.5)
]

def _compute_wedge_mean(vals, rp, rt, mu_range, r_edges, abs_mu=True):
    r = np.sqrt(rp**2 + rt**2)
    mu = rp / np.maximum(r, 1e-12)
    if abs_mu:
        mu = np.abs(mu)
    mu_min, mu_max = mu_range
    mask_mu = (mu >= mu_min) & (mu <= mu_max)
    w = rt
    out = np.zeros(r_edges.size - 1)
    for i in range(r_edges.size - 1):
        in_bin = mask_mu & (r >= r_edges[i]) & (r < r_edges[i+1])
        if np.any(in_bin):
            wi = w[in_bin]
            vi = vals[in_bin]
            s = wi.sum()
            out[i] = (wi * vi).sum() / s if s > 0 else 0.0
        else:
            out[i] = 0.0
    centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    return centers, out

def _compute_wedge_error_diag(var_vals, rp, rt, mu_range, r_edges, abs_mu=True):
    r = np.sqrt(rp**2 + rt**2)
    mu = rp / np.maximum(r, 1e-12)
    if abs_mu:
        mu = np.abs(mu)
    mu_min, mu_max = mu_range
    mask_mu = (mu >= mu_min) & (mu <= mu_max)
    w = rt
    err = np.zeros(r_edges.size - 1)
    for i in range(r_edges.size - 1):
        in_bin = mask_mu & (r >= r_edges[i]) & (r < r_edges[i+1])
        if np.any(in_bin):
            wi = w[in_bin]
            vi = var_vals[in_bin]
            s = wi.sum()
            err[i] = np.sqrt(np.sum((wi**2) * vi)) / (s if s > 0 else 1.0)
        else:
            err[i] = 0.0
    centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    return centers, err

# 输出 r 方向的楔分箱数与范围（和 replot_polychord 一致）
n_rp_out = 50
r_min_plot, r_max_plot = 10.0, 180.0

def _build_fine_model(rp, rt, vals, scale_factor=10):
    rp_min = float(np.min(rp)); rp_max = float(np.max(rp))
    rt_min = max(0.0, float(np.min(rt)))
    rt_max = float(np.max(rt))
    rp_vals = np.unique(rp)
    rt_vals = np.unique(rt)
    n_rp = int(rp_vals.size)
    n_rt = int(rt_vals.size)
    n_rp_f = max(n_rp * scale_factor, 20)
    n_rt_f = max(n_rt * scale_factor, 20)
    drp = (rp_max - rp_min) / n_rp_f if n_rp_f > 0 else 1.0
    drt = (rt_max - rt_min) / n_rt_f if n_rt_f > 0 else 1.0
    rp_centers_f = np.linspace(rp_min + drp/2, rp_max - drp/2, n_rp_f)
    rt_centers_f = np.linspace(rt_min + drt/2, rt_max - drt/2, n_rt_f)
    RT, RP = np.meshgrid(rt_centers_f, rp_centers_f)
    pts = np.stack([rp, rt], axis=1)
    grid = griddata(pts, vals, (RP, RT), method='linear')
    if np.isnan(grid).any():
        grid_nn = griddata(pts, vals, (RP, RT), method='nearest')
        grid = np.where(np.isnan(grid), grid_nn, grid)
    m_fine = grid.flatten()
    rp_min_edge = rp_centers_f[0] - drp/2
    rp_max_edge = rp_centers_f[-1] + drp/2
    rt_min_edge = max(0.0, rt_centers_f[0] - drt/2)
    rt_max_edge = rt_centers_f[-1] + drt/2
    vol_weights = RT.flatten()
    return m_fine, (rp_min_edge, rp_max_edge, rp_centers_f.size), (rt_min_edge, rt_max_edge, rt_centers_f.size), vol_weights

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

def interpolate_to_finer_grid(m_input, n_rp, n_rt, rp_min, rp_max, rt_min, rt_max, scale_factor=4):
    """
    Interpolate the input 2D data to a finer grid using 2D spline interpolation.
    """
    # Original grid points (midpoints)
    drp = (rp_max - rp_min) / n_rp
    drt = (rt_max - rt_min) / n_rt
    rp_centers = np.linspace(rp_min + drp/2, rp_max - drp/2, n_rp)
    rt_centers = np.linspace(rt_min + drt/2, rt_max - drt/2, n_rt)
    
    # Reshape input to 2D grid (rt varies fastest in C-order usually, but check data layout)
    # Assuming standard layout where loop order is often RP then RT or vice versa.
    # Based on vega code usually: data is flattened.
    # Let's assume shape (n_rp, n_rt) for now based on typical correlation function storage
    # If m_input is 1D array of size n_rp*n_rt
    
    try:
        grid_data = m_input.reshape((n_rp, n_rt))
    except ValueError:
        # Fallback or error if shape doesn't match
        return m_input, n_rp, n_rt, rp_min, rp_max, rt_min, rt_max

    # Create interpolator
    # RectBivariateSpline requires strictly increasing x and y arrays
    # Use linear interpolation (kx=1, ky=1) to avoid overshooting/ringing artifacts near sharp features (e.g. BAO peak)
    interp_func = RectBivariateSpline(rp_centers, rt_centers, grid_data, kx=1, ky=1)
    
    # New finer grid
    n_rp_new = n_rp * scale_factor
    n_rt_new = n_rt * scale_factor
    
    # Calculate new centers to match Wedge expectations
    # The new grid should cover the same physical range [rp_min, rp_max]
    drp_new = (rp_max - rp_min) / n_rp_new
    drt_new = (rt_max - rt_min) / n_rt_new
    
    rp_centers_new = np.linspace(rp_min + drp_new/2, rp_max - drp_new/2, n_rp_new)
    rt_centers_new = np.linspace(rt_min + drt_new/2, rt_max - drt_new/2, n_rt_new)
    
    # Evaluate on new grid
    # RectBivariateSpline will extrapolate outside the original centers range.
    # Since new centers are slightly closer to the edges than old centers (by dr/2 - dr_new/2),
    # this extrapolation is minimal and physically acceptable.
    new_grid_data = interp_func(rp_centers_new, rt_centers_new)
    
    # Flatten back to 1D
    m_input_new = new_grid_data.flatten()
    
    return m_input_new, n_rp_new, n_rt_new

# 不再依赖规则网格与 Wedge 来生成模型曲线，直接按点做 (r,μ) 分箱



def _canonize_corr_name(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum())

def _resolve_corr_key(fr: FitResults, corr_name: str) -> str:
    target = _canonize_corr_name(corr_name)
    keys = list(fr.correlations.keys())
    for k in keys:
        if _canonize_corr_name(k) == target:
            return k
    raise KeyError(f"Correlation '{corr_name}' not found in FITS. Available: {keys}")

def _resolve_key_in_dict(d: dict, corr_name: str) -> str:
    target = _canonize_corr_name(corr_name)
    for k in d.keys():
        if _canonize_corr_name(k) == target:
            return k
    raise KeyError(f"Correlation '{corr_name}' not found. Available: {list(d.keys())}")

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
compare_style = 'four'  # Default to 'four' subplots for compare mode

if len(sys.argv) > 1:
    if '-compare' in sys.argv:
        mode = 'compare'
        if '-one' in sys.argv:
            compare_style = 'one'
        elif '-four' in sys.argv:
            compare_style = 'four'
    elif '-wedge' in sys.argv:
        mode = 'wedge'
    if '--corr' in sys.argv:
        try:
            idx = sys.argv.index('--corr')
            corr = sys.argv[idx+1]
        except Exception:
            pass
    if '--config' in sys.argv:
        try:
            idx = sys.argv.index('--config')
            config_path = sys.argv[idx+1]
        except Exception:
            pass
    if '--fit1' in sys.argv:
        try:
            idx = sys.argv.index('--fit1')
            fit1 = sys.argv[idx+1]
        except Exception:
            pass
    if '--fit2' in sys.argv:
        try:
            idx = sys.argv.index('--fit2')
            fit2 = sys.argv[idx+1]
        except Exception:
            pass
    if '--plot-rmin' in sys.argv:
        try:
            idx = sys.argv.index('--plot-rmin')
            r_min_plot = float(sys.argv[idx+1])
        except Exception:
            pass
    if '--plot-rmax' in sys.argv:
        try:
            idx = sys.argv.index('--plot-rmax')
            r_max_plot = float(sys.argv[idx+1])
        except Exception:
            pass

if fit1 is None:
    raise RuntimeError("--fit1 is required. Example: --fit1 /path/to/first.fits")

fr1 = FitResults(fit1)
fr2 = FitResults(fit2) if fit2 is not None else None

# Refresh correlation selection after possible overrides, case-insensitive and format-agnostic
try:
    corr_key1 = _resolve_corr_key(fr1, corr)
    c1 = fr1.correlations[corr_key1]
    if fr2 is not None:
        corr_key2 = _resolve_corr_key(fr2, corr)
        c2 = fr2.correlations[corr_key2]
    else:
        c2 = None
except Exception as e:
    print(f"Error resolving correlation '{corr}': {e}")
    # Fallback to direct key if exists; else raise
    if fr2 is not None:
        if corr in fr1.correlations and corr in fr2.correlations:
            c1 = fr1.correlations[corr]
            c2 = fr2.correlations[corr]
        else:
            raise
    else:
        if corr in fr1.correlations:
            c1 = fr1.correlations[corr]
            c2 = None
        else:
            raise

# 提取掩膜后的向量，避免网格顺序假设（在 c1/c2 就绪之后）
rp_all = np.asarray(c1.rp)
rt_all = np.asarray(c1.rt)
model_mask1 = c1.model_mask.astype(bool)
data_mask1 = c1.data_mask.astype(bool)
model_pts1 = np.asarray(c1.model)[model_mask1]
rp_model1 = rp_all[model_mask1]
rt_model1 = rt_all[model_mask1]
data_vec1 = np.asarray(c1.data)[data_mask1]
var_vec1 = np.asarray(c1.variance)[data_mask1]
rp_data1 = rp_all[data_mask1]
rt_data1 = rt_all[data_mask1]

# 第二个拟合的模型（用同一 FITS 网格结构）
if c2 is not None:
    rp_all2 = np.asarray(c2.rp)
    rt_all2 = np.asarray(c2.rt)
    model_mask2 = c2.model_mask.astype(bool)
    model_pts2 = np.asarray(c2.model)[model_mask2]
    rp_model2 = rp_all2[model_mask2]
    rt_model2 = rt_all2[model_mask2]
else:
    model_pts2 = None

# 设置全局字体大小（必须在创建 Figure 之前设置）
plt.rcParams.update({'font.size': 14})  # 基础字体大小
plt.rcParams.update({'axes.titlesize': 18}) # 标题字体大小
plt.rcParams.update({'axes.labelsize': 16}) # 坐标轴标签字体大小
plt.rcParams.update({'xtick.labelsize': 14}) # x轴刻度字体大小
plt.rcParams.update({'ytick.labelsize': 14}) # y轴刻度字体大小
plt.rcParams.update({'legend.fontsize': 16}) # 图例字体大小
plt.rcParams.update({'lines.linewidth': 2.5}) # 全局线条粗细

# 创建 Figure 和 Axes
if mode == 'compare' and compare_style == 'one':
    fig, ax = plt.subplots(figsize=(10, 8))
    axes = [ax] * 4 # Duplicate reference for loop compatibility
elif mode == 'compare' and compare_style != 'one':
    # 2x2 的子图，共享 x 和 y 轴
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()
else:
    # 2x2 subplots for wedge mode (4 mu bins)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

# Colors for 'one' style
# colors = ['k', 'r', 'b', 'g']
colors = ['C0', 'C1', 'C2', 'C3']
# Linestyles for 'one' style
linestyles = ['-', '--', '-.', ':']

# Prepare collector for auto y-limits in compare-one mode
def _round_up_strict(x, step=0.05):
    k = np.floor(x / step) + 1.0
    return np.round(k * step, 2)
def _round_down_strict(x, step=0.05):
    k = np.ceil(x / step) - 1.0
    return np.round(k * step, 2)
ratio_curves_one = [] if (mode == 'compare' and compare_style == 'one') else None

for i, (mu_min, mu_max) in enumerate(mu_bins):
    ax = axes[i]
    r_edges = np.linspace(r_min_plot, r_max_plot, n_rp_out + 1)
    m1_fine, rp_tuple1, rt_tuple1, vol_w1 = _build_fine_model(rp_model1, rt_model1, model_pts1, scale_factor=10)
    w1 = Wedge(mu=(mu_min, mu_max), rp=rp_tuple1, rt=rt_tuple1, r=(r_min_plot, r_max_plot, n_rp_out), abs_mu=True)
    r, m1 = w1(m1_fine, weights=vol_w1)
    m2 = None
    if model_pts2 is not None:
        m2_fine, rp_tuple2, rt_tuple2, vol_w2 = _build_fine_model(rp_model2, rt_model2, model_pts2, scale_factor=10)
        w2 = Wedge(mu=(mu_min, mu_max), rp=rp_tuple2, rt=rt_tuple2, r=(r_min_plot, r_max_plot, n_rp_out), abs_mu=True)
        _, m2 = w2(m2_fine, weights=vol_w2)
        
    if mode == 'wedge':
        ax.set_xlim(0, 200)
        ax.grid()
        y1 = gaussian_filter1d(m1, sigma=0.5) * r**2
        ax.plot(r, y1, '-', label='model 1')
        if m2 is not None:
            y2 = gaussian_filter1d(m2, sigma=0.5) * r**2
            ax.plot(r, y2, ':', label='model 2')
        rd = None
        if config_path is not None:
            try:
                from vega.vega_interface import VegaInterface
                vi = VegaInterface(config_path)
                corr_key_vi = _resolve_key_in_dict(vi.data, corr)
                data_obj = vi.data[corr_key_vi]
                def compute_wedge_with_errors_from_vega(data_obj_local, mu_range, n_r_out, abs_mu=True):
                    rp_min0 = data_obj_local.data_coordinates.rp_min
                    rp_max0 = data_obj_local.data_coordinates.rp_max
                    rt_max0 = data_obj_local.data_coordinates.rt_max
                    n_rp0 = data_obj_local.data_coordinates.rp_nbins
                    n_rt0 = data_obj_local.data_coordinates.rt_nbins
                    w_obj0 = Wedge(
                        mu=mu_range,
                        rp=(rp_min0, rp_max0, n_rp0),
                        rt=(0.0, rt_max0, n_rt0),
                        r=(0.0, 200.0, 50),
                        abs_mu=abs_mu
                    )
                    data_vec0 = data_obj_local.data_vec
                    cov0 = data_obj_local.cov_mat
                    rd0, dd0, cd0 = w_obj0(data_vec0, covariance=cov0)
                    errs0 = np.sqrt(np.diag(cd0))
                    return rd0, dd0, errs0
                rd, wd, ed = compute_wedge_with_errors_from_vega(data_obj, (mu_min, mu_max), n_rp_out, True)
                ax.errorbar(rd, wd*rd**2, yerr=ed*rd**2, fmt='o', color='k', markersize=6, alpha=0.8, label='data' if i == 0 else None)
            except Exception:
                pass
        if rd is None:
            r_edges_data = np.arange(0.0, 200.0 + 1e-9, 4.0)
            rd2, wd2 = _compute_wedge_mean(data_vec1, rp_data1, rt_data1, (mu_min, mu_max), r_edges_data, abs_mu=True)
            _, ed2 = _compute_wedge_error_diag(var_vec1, rp_data1, rt_data1, (mu_min, mu_max), r_edges_data, abs_mu=True)
            ax.errorbar(rd2, wd2*rd2**2, yerr=ed2*rd2**2, fmt='o', color='k', markersize=6, alpha=0.8, label='data' if i == 0 else None)
        # Only show y-label for left column (indices 0 and 2)
        if i % 2 == 0:
            ax.set_ylabel(r'$r^2 \xi(r)$')
        if i == 0:
            ax.legend(frameon=False)
        ax.set_title(r'${} < \mu < {}$'.format(mu_min, mu_max))
        # 只在底部两个子图显示 x 轴标签
        if i >= 2:
            ax.set_xlabel('r [Mpc/h]')
            
    elif mode == 'compare':
        ax.set_xlim(10, 180)
        
        if m2 is None:
            print('Warning: compare mode requires two FITS files (--fit1 and --fit2). Skipping this panel.')
            continue
        ratio = get_smooth_ratio(m1, m2, r)
        
        if compare_style == 'four':
            ax.plot(r, ratio, 'k-')
            # Only show y-label for left column (indices 0 and 2)
            if i % 2 == 0:
                ax.set_ylabel('Ratio')
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
            if i == 0:
                ax.legend(frameon=False)
            ax.set_title(r'${} < \mu < {}$'.format(mu_min, mu_max))
            # 只在底部两个子图显示 x 轴标签
            if i >= 2:
                ax.set_xlabel('r [Mpc/h]')
                
        elif compare_style == 'one':
            # In 'one' style, we plot all lines on the same axes (axes[0])
            # But since we duplicated ax in axes list, 'ax' is always the single main axis
            label_text = r'${} < \mu < {}$'.format(mu_min, mu_max)
            ax.plot(r, ratio, color=colors[i], linestyle=linestyles[i], label=label_text)
            ratio_curves_one.append(ratio)
            
            # Only set labels and legend once (at the end or specific iteration)
            if i == 3: # After last plot
                ax.set_ylabel('Ratio')
                ax.set_xlabel('r [Mpc/h]')
                ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
                # Auto y-limits across all lines with 0.05 grid (strict bounds)
                try:
                    rmin = np.nanmin([np.nanmin(rc) for rc in ratio_curves_one])
                    rmax = np.nanmax([np.nanmax(rc) for rc in ratio_curves_one])
                    ylo = _round_down_strict(rmin, 0.05)
                    yhi = _round_up_strict(rmax, 0.05)
                    ax.set_ylim(ylo, yhi)
                except Exception:
                    pass
                ax.legend(frameon=False)
                # ax.set_title('Ratio Comparison') # Optional title

# 调整子图间距，减少留白
if not (mode == 'compare' and compare_style == 'one'):
    # 增加 hspace 以避免上下子图标签重叠
    plt.subplots_adjust(hspace=0.13, wspace=0.1)

try:
    out_dir = os.path.dirname(__file__)
    base1 = os.path.splitext(os.path.basename(fit1))[0]
    base2 = os.path.splitext(os.path.basename(fit2))[0] if fit2 is not None else None
    if mode == 'compare':
        style = 'one' if compare_style == 'one' else 'four'
        if base2 is None:
            fname = f"fits_{corr}_compare_{base1}_single_{style}.png"
        else:
            fname = f"fits_{corr}_compare_{base1}_vs_{base2}_{style}.png"
    else:
        if base2 is None:
            fname = f"fits_{corr}_wedge_{base1}.png"
        else:
            fname = f"fits_{corr}_wedge_{base1}_vs_{base2}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure to {out_path}")
except Exception as e:
    print(f"Failed to save figure: {e}")
plt.show()
