
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
import argparse
import csv
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RectBivariateSpline

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from vega import VegaInterface
from vega.vega_interface import BLIND_FIXED_PARS
from vega.plots.wedges import Wedge

style_path = os.path.join(os.path.dirname(__file__), '5par.mplstyle')
if os.path.exists(style_path):
    plt.style.use(style_path)
else:
    print(f"Warning: Style file {style_path} not found. Using default style.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify the wedge bins where reio model gives the largest chi2 improvement over fid model."
    )
    parser.add_argument('--chain-fid', type=str, required=True,
                        help="Path to Fiducial Polychord chain directory.")
    parser.add_argument('--chain-reio', type=str, required=True,
                        help="Path to Reionization Polychord chain directory.")
    parser.add_argument('--config-fid', type=str, default='analysis/main_fid.ini',
                        help="Path to Vega config for the Fiducial model.")
    parser.add_argument('--config-reio', type=str, default='analysis/main_reio.ini',
                        help="Path to Vega config for the Reionization model.")
    parser.add_argument('--corr', type=str, default='lyaxlya',
                        help="Correlation name to analyze (e.g. lyaxlya).")
    parser.add_argument('--plot-rmin', type=float, default=40.0,
                        help="Minimum r for wedge binning [Mpc/h].")
    parser.add_argument('--plot-rmax', type=float, default=180.0,
                        help="Maximum r for wedge binning [Mpc/h].")
    parser.add_argument('--nr', type=int, default=50,
                        help="Number of r bins per wedge.")
    parser.add_argument('--top-k', type=int, default=10,
                        help="Number of most-improved points to mark in red.")
    parser.add_argument('--use-mean', action='store_true',
                        help="Use posterior mean (.stats) instead of the default MAP (max logL sample in chain.txt).")
    parser.add_argument('--zoom', action='store_true',
                        help="Zoom into a specified wedge/r range (--zoom-wedge/--zoom-rmin/--zoom-rmax) to see small model differences.")
    parser.add_argument('--zoom-wedge', type=int, default=0, choices=[0, 1, 2, 3],
                        help="Which wedge to zoom into (0=0.95<μ<1.0, 1=0.8<μ<0.95, 2=0.5<μ<0.8, 3=0.0<μ<0.5). Default: 0.")
    parser.add_argument('--zoom-rmin', type=float, default=40.0,
                        help="Lower r bound of the zoom window [Mpc/h]. Default: 40.")
    parser.add_argument('--zoom-rmax', type=float, default=70.0,
                        help="Upper r bound of the zoom window [Mpc/h]. Default: 70.")
    parser.add_argument('--inset-x', type=float, default=None,
                        help="Normalized X position of the 1st zoom inset (slot 0) inside its parent wedge (0=left, 1=right). Default (None): use auto layout.")
    parser.add_argument('--inset-y', type=float, default=None,
                        help="Normalized Y position of the 1st zoom inset (slot 0) inside its parent wedge (0=bottom, 1=top). Default (None): use auto layout.")
    parser.add_argument('--inset-w', type=float, default=None,
                        help="Normalized width of the 1st zoom inset (slot 0). Default (None): use auto layout.")
    parser.add_argument('--inset-h', type=float, default=None,
                        help="Normalized height of the 1st zoom inset (slot 0). Default (None): use auto layout.")
    parser.add_argument('--inset2-x', type=float, default=None,
                        help="Normalized X position of the 2nd zoom inset (slot 1) when two insets share the same wedge. Default (None): use auto layout.")
    parser.add_argument('--inset2-y', type=float, default=None,
                        help="Normalized Y position of the 2nd zoom inset (slot 1) when two insets share the same wedge. Default (None): use auto layout.")
    parser.add_argument('--inset2-w', type=float, default=None,
                        help="Normalized width of the 2nd zoom inset (slot 1). Default (None): use auto layout.")
    parser.add_argument('--inset2-h', type=float, default=None,
                        help="Normalized height of the 2nd zoom inset (slot 1). Default (None): use auto layout.")
    parser.add_argument('--zoom-top5', action='store_true',
                        help="Zoom inset for the Top-5 most-improved bins instead of the default Top-2. "
                             "Assumes each wedge contains at most 2 zoom bins, so the existing 2-slot "
                             "layout (--inset-* / --inset2-*) is reused as-is. Default: off (Top-2).")
    parser.add_argument('--legend-wedge', type=int, default=None, choices=[0, 1, 2, 3],
                        help="Force the legend into a specific wedge (0=0.95<mu<1.0, 1=0.8<mu<0.95, "
                             "2=0.5<mu<0.8, 3=0.0<mu<0.5). Default (None): auto-pick the first empty wedge "
                             "(no inset), upper/lower center chosen to avoid data.")
    return parser.parse_args()


def read_params(chain_dir, use_map=False):
    if not os.path.exists(chain_dir):
        raise FileNotFoundError(f"Directory {chain_dir} not found.")

    files = os.listdir(chain_dir)
    stats_file = None
    chain_file = None
    paramnames_file = None

    for f in files:
        if f.endswith('.stats'):
            stats_file = os.path.join(chain_dir, f)
        elif f.endswith('.txt') and not ('dead' in f or 'resume' in f):
            chain_file = os.path.join(chain_dir, f)
        elif f.endswith('.paramnames'):
            paramnames_file = os.path.join(chain_dir, f)

    if not paramnames_file:
        raise FileNotFoundError(f".paramnames not found in {chain_dir}")

    with open(paramnames_file, 'r') as f:
        param_names = [line.strip().split()[0] for line in f if line.strip()]

    if use_map:
        if not chain_file:
            raise RuntimeError("Requested MAP point but no .txt chain file in {chain_dir}.")
        arr = np.loadtxt(chain_file)
        if arr.shape[1] != 2 + len(param_names):
            raise RuntimeError(f"Chain columns {arr.shape[1]} != 2 + npar ({len(param_names)}).")
        weights = arr[:, 0]
        m2logL = arr[:, 1]
        theta = arr[:, 2:]
        logL = -0.5 * m2logL
        imax = int(np.argmax(logL))
        params = {name: float(theta[imax, i]) for i, name in enumerate(param_names)}
        print(f"  -> MAP sample (row {imax}) logL = {logL[imax]:.4f}, weight = {weights[imax]:.4e}")
        return params

    mean_params = {}
    if stats_file:
        print(f"  -> Reading posterior means from {stats_file}")
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            start_idx = -1
            for i, line in enumerate(lines):
                if "Dim No.       Mean        Sigma" in line:
                    start_idx = i + 1
                    break
            if start_idx != -1:
                for line in lines[start_idx:]:
                    if "----------------" in line:
                        break
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            idx = int(parts[0]) - 1
                            mean_val = float(parts[1])
                            if idx < len(param_names):
                                mean_params[param_names[idx]] = mean_val
                        except ValueError:
                            continue
    if chain_file and not mean_params:
        print(f"  -> Computing weighted posterior means from chain file.")
        data = np.loadtxt(chain_file)
        weights = data[:, 0]
        samples = data[:, 2:]
        for i, name in enumerate(param_names):
            mean_params[name] = float(np.average(samples[:, i], weights=weights))
    if not mean_params:
        raise RuntimeError(f"Could not determine parameters from {chain_dir}.")
    return mean_params


def _init_blinding(vega):
    if vega._blind is not None:
        return
    vega._blind = any(getattr(d, "blind", False) for d in vega.data.values())


def build_local_params(vega, params):
    _init_blinding(vega)
    local_params = copy.deepcopy(vega.params)
    for par, val in params.items():
        if par in local_params:
            local_params[par] = val
    if vega._blind:
        for par in list(local_params.keys()):
            if par in BLIND_FIXED_PARS:
                local_params[par] = 1.0
    return local_params


def compute_diff_y_r_mu_for_corr(vega, local_params, corr_name):
    """Compute (diff, y=inv_cov.dot(diff), r, mu) per data pixel inside a single corr item."""
    if corr_name not in vega.models or corr_name not in vega.data:
        raise KeyError(f"{corr_name} not present in vega models or data.")

    if vega._use_global_cov:
        # walk all corr_items, identify which slice of the concatenated vector is ours
        full_diff_parts = []
        full_y_parts = []
        full_r_parts = []
        full_mu_parts = []
        our_start = 0
        target_idx = None
        for k, name in enumerate(vega.corr_items):
            model_cf = vega.models[name].compute(
                local_params, vega.fiducial["pk_full"], vega.fiducial["pk_smooth"]
            )
            data_obj = vega.data[name]
            dvec = data_obj.masked_data_vec
            r = data_obj.data_coordinates.r_regular_grid[data_obj.data_mask]
            rp = data_obj.data_coordinates.rp_regular_grid[data_obj.data_mask]
            rt = data_obj.data_coordinates.rt_regular_grid[data_obj.data_mask]
            mu = rp / np.maximum(r, 1e-12)
            diff_piece = dvec - model_cf[data_obj.model_mask]
            sl = slice(our_start, our_start + diff_piece.size)
            our_start += diff_piece.size
            if name == corr_name:
                target_idx = k
                target_diff = diff_piece
                target_r = r
                target_mu = mu
            full_diff_parts.append(diff_piece)
            full_r_parts.append(r)
            full_mu_parts.append(mu)
        if target_idx is None:
            raise RuntimeError(f"{corr_name} not in corr_items.")
        full_diff = np.concatenate(full_diff_parts)
        full_y = vega.masked_global_invcov.dot(full_diff)
        # Now recover the slice for the corr of interest
        our_start = 0
        corr_slice = None
        for k, name in enumerate(vega.corr_items):
            length = full_diff_parts[k].size
            if name == corr_name:
                corr_slice = slice(our_start, our_start + length)
                break
            our_start += length
        return full_diff[corr_slice], full_y[corr_slice], target_r, np.abs(target_mu)

    data_obj = vega.data[corr_name]
    model_cf = vega.models[corr_name].compute(
        local_params, vega.fiducial["pk_full"], vega.fiducial["pk_smooth"]
    )
    diff = data_obj.masked_data_vec - model_cf[data_obj.model_mask]
    y = data_obj.inv_masked_cov.dot(diff)
    r = data_obj.data_coordinates.r_regular_grid[data_obj.data_mask]
    rp = data_obj.data_coordinates.rp_regular_grid[data_obj.data_mask]
    mu = rp / np.maximum(r, 1e-12)
    return diff, y, r, np.abs(mu)


def bin_per_pixel_chi2_into_wedges(r_vals, mu_vals, per_pix_chi2, mu_bins, r_min, r_max, n_r):
    """Return 2D array (n_mu, n_r) of summed chi2 per wedge/r bin."""
    r_edges = np.linspace(r_min, r_max, n_r + 1)
    n_mu = len(mu_bins)
    out = np.zeros((n_mu, n_r))
    for i, (mulo, muhi) in enumerate(mu_bins):
        mask_mu = (mu_vals >= mulo) & (mu_vals <= muhi)
        r_sub = r_vals[mask_mu]
        chi2_sub = per_pix_chi2[mask_mu]
        for j in range(n_r):
            sel = (r_sub >= r_edges[j]) & (r_sub < r_edges[j + 1])
            out[i, j] = float(np.sum(chi2_sub[sel])) if np.any(sel) else 0.0
    return out, r_edges


def interpolate_to_finer_grid(m_input, n_rp, n_rt, rp_min, rp_max, rt_min, rt_max, scale_factor=4):
    """Copy of replot_polychord.interpolate_to_finer_grid (RectBivariateSpline, kx=ky=1)."""
    drp = (rp_max - rp_min) / n_rp
    drt = (rt_max - rt_min) / n_rt
    rp_centers = np.linspace(rp_min + drp/2, rp_max - drp/2, n_rp)
    rt_centers = np.linspace(rt_min + drt/2, rt_max - drt/2, n_rt)
    try:
        grid_data = m_input.reshape((n_rp, n_rt))
    except ValueError:
        print("Warning: reshape to (n_rp,n_rt) failed; skipping interpolation.")
        return m_input, n_rp, n_rt
    interp_func = RectBivariateSpline(rp_centers, rt_centers, grid_data, kx=1, ky=1)
    n_rp_new = n_rp * scale_factor
    n_rt_new = n_rt * scale_factor
    drp_new = (rp_max - rp_min) / n_rp_new
    drt_new = (rt_max - rt_min) / n_rt_new
    rp_centers_new = np.linspace(rp_min + drp_new/2, rp_max - drp_new/2, n_rp_new)
    rt_centers_new = np.linspace(rt_min + drt_new/2, rt_max - drt_new/2, n_rt_new)
    new_grid_data = interp_func(rp_centers_new, rt_centers_new)
    m_input_new = new_grid_data.flatten()
    return m_input_new, n_rp_new, n_rt_new


def compute_wedge_data_points(data_obj, mu_range, n_r):
    """Project real data vector (and its cov) to wedge r bins: r ∈ [0, 200].

    This matches replot_polychord's compute_wedge_with_errors: it projects
    the FULL data range (0..200) for plotting. The [plot_rmin, plot_rmax]
    fitting range is only used INSIDE bin_per_pixel_chi2_into_wedges for
    the delta chi2 calculation, not for drawing points.
    """
    rp_min = data_obj.data_coordinates.rp_min
    rp_max = data_obj.data_coordinates.rp_max
    rt_max = data_obj.data_coordinates.rt_max
    n_rp = data_obj.data_coordinates.rp_nbins
    n_rt = data_obj.data_coordinates.rt_nbins
    w_obj = Wedge(mu=mu_range,
                  rp=(rp_min, rp_max, n_rp),
                  rt=(0.0, rt_max, n_rt),
                  r=(0.0, 200.0, n_r),
                  abs_mu=True)
    dvec = data_obj.data_vec
    try:
        cov = data_obj.cov_mat
        rd, dd, cd = w_obj(dvec, covariance=cov)
        errs = np.sqrt(np.maximum(np.diag(cd), 0.0))
    except Exception:
        rd, dd = w_obj(dvec)
        errs = np.zeros_like(dd)
    return rd, dd, errs


def compute_wedge_model_curve(model_fine, rp_centers, rt_centers, mu_range, r_min, r_max, n_r):
    """Project a model 2D vector onto wedge r bins with volume weights (matching replot)."""
    n_rp = rp_centers.size
    n_rt = rt_centers.size
    if n_rp > 1:
        drp = rp_centers[1] - rp_centers[0]
    else:
        drp = (r_max - r_min) / max(1, n_r)
    if n_rt > 1:
        drt = rt_centers[1] - rt_centers[0]
    else:
        drt = drp
    rp_min_edge = rp_centers[0] - drp / 2
    rp_max_edge = rp_centers[-1] + drp / 2
    rt_min_edge = max(0.0, rt_centers[0] - drt / 2)
    rt_max_edge = rt_centers[-1] + drt / 2
    rt_mesh, rp_mesh = np.meshgrid(rt_centers, rp_centers)
    vol_weights = rt_mesh.flatten()
    w_obj = Wedge(mu=mu_range,
                  rp=(rp_min_edge, rp_max_edge, n_rp),
                  rt=(rt_min_edge, rt_max_edge, n_rt),
                  r=(r_min, r_max, n_r),
                  abs_mu=True)
    r, m_wedged = w_obj(model_fine, weights=vol_weights)
    return r, m_wedged


def _choose_grids_for_vec(data_obj, vec_len):
    """Copy replot_polychord's _choose_grids_for_vec search order."""
    cands = []
    try:
        cands.append((data_obj.model_coordinates.rp_regular_grid,
                      data_obj.model_coordinates.rt_regular_grid))
    except Exception:
        pass
    try:
        cands.append((data_obj.dist_model_coordinates.rp_regular_grid,
                      data_obj.dist_model_coordinates.rt_regular_grid))
    except Exception:
        pass
    try:
        cands.append((data_obj.data_coordinates.rp_regular_grid,
                      data_obj.data_coordinates.rt_regular_grid))
    except Exception:
        pass
    for rp_g, rt_g in cands:
        if rp_g is not None and rt_g is not None and rp_g.size == vec_len and rt_g.size == vec_len:
            return rp_g, rt_g
    return None, None


def get_rp_rt_centers(vega_if, corr_name, model_vec):
    data_obj = vega_if.data[corr_name]
    rp_g, rt_g = _choose_grids_for_vec(data_obj, model_vec.size)
    if rp_g is None:
        raise RuntimeError(
            f"No coordinate grid matched model vector length {model_vec.size}."
        )
    return np.unique(rp_g), np.unique(rt_g)


def main():
    args = parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    mu_bins = [(0.95, 1.0), (0.8, 0.95), (0.5, 0.8), (0.0, 0.5)]

    # ---- Unified r grid across the ENTIRE pipeline: chi2-binning, CSV rows, data points rd,
    # and Top-10 matching all use the SAME r_edges / r_centers: r ∈ [0, 200], n_r = args.nr.
    # This guarantees:
    #   - matches replot_polychord wedge's compute_wedge_with_errors(r=(0,200,n_rp_out)),
    #   - makes CSV's r_center values EXACTLY equal rd's values (no "nearest-neighbor drift),
    #   - makes top-10 mask index j directly maps to rd[j] with zero ambiguity.
    # The FITTING range [plot_rmin, plot_rmax] is only used:
    #   - to SELECT which bins are eligible for Top-K ranking,
    #   - for Wedge projection of model curves (which replot clips to that range).
    fit_rmin = float(args.plot_rmin)
    fit_rmax = float(args.plot_rmax)
    n_r = int(args.nr)
    r_min = 0.0
    r_max = 200.0
    r_edges = np.linspace(r_min, r_max, n_r + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # ====== 1. Init vegas, read params ======
    print("\n[1/5] Loading configs and params.")

    if not os.path.exists(args.config_fid):
        raise FileNotFoundError(f"{args.config_fid} not found.")
    if not os.path.exists(args.config_reio):
        raise FileNotFoundError(f"{args.config_reio} not found.")

    vega_fid = VegaInterface(args.config_fid)
    print(f"  Fid Vega ({args.config_fid}) init done. Real data.")
    vega_reio = VegaInterface(args.config_reio)
    print(f"  Reio Vega ({args.config_reio}) init done. Real data.")

    if args.corr not in vega_fid.data or args.corr not in vega_reio.data:
        raise KeyError(f"Corr {args.corr} missing from at least one vega instance.")

    tag = "MAP" if not args.use_mean else "MEAN"
    print(f"  chain_fid = {args.chain_fid}  ({tag})")
    params_fid = read_params(args.chain_fid, use_map=(not args.use_mean))
    print(f"  chain_reio = {args.chain_reio}  ({tag})")
    params_reio = read_params(args.chain_reio, use_map=(not args.use_mean))

    # Build local parameter dicts (with blinding support)
    local_fid = build_local_params(vega_fid, params_fid)
    local_reio = build_local_params(vega_reio, params_reio)

    # ====== 2. Compute per-pixel chi2 contributions and project ======
    print("\n[2/5] Computing data-model differences on each 2D pixel.")

    diff_fid, y_fid, r_px, mu_px = compute_diff_y_r_mu_for_corr(
        vega_fid, local_fid, args.corr
    )
    diff_reio, y_reio, r_px2, mu_px2 = compute_diff_y_r_mu_for_corr(
        vega_reio, local_reio, args.corr
    )

    if r_px.shape != r_px2.shape or not np.allclose(r_px, r_px2, equal_nan=True):
        print("  Warning: FID and REIO data coordinates/r_masks differ slightly; using FID's pixels.")
        # The pixel masks may differ between ini files; but we only need binning of chi2, so using FID
        # grid for bin assignment is fine.  Both diff/reio lengths must match though:
    if diff_reio.size != diff_fid.size:
        # fall back to the smaller one if compatible is too complex; instead, re-run with same data pixels
        raise RuntimeError(
            f"FID pixel count ({diff_fid.size}) != REIO pixel count ({diff_reio.size}). "
            f"Make sure both configs use the same data mask and corr binning."
        )

    per_pix_chi2_fid = (diff_fid * y_fid).real.astype(float)
    per_pix_chi2_reio = (diff_reio * y_reio).real.astype(float)

    # ===== Per-pixel model/data values for the NEW "top-5 bin per-pixel zoom-inset" mode =====
    # User's new zoom-in request: for the TOP-5 best-improved bins, show an inset that plots
    # each INDIVIDUAL raw 2D data pixel contained inside that bin as a separate point, with
    # x = pixel_r (sorted by r, ignoring small mu differences inside the same wedge/bin)
    # and y = r_pix**2 * xi(r_pix, mu_pix), together with FID/REIO model curves at those
    # same pixel positions. This requires per-pixel access to: true data, 2 models, and a
    # per-pixel 1sigma error bar (diagonal of the masked cov matrix).
    data_obj_fid_corr = vega_fid.data[args.corr]
    # True data vector AFTER applying data_mask (this is exactly the vector diff is computed from)
    d_pix = np.asarray(data_obj_fid_corr.masked_data_vec, dtype=float)
    # diff_fid = d_pix - model_fid_masked  =>  m_fid = d - diff (per-pixel true FID model value)
    # diff_reio = d_pix - model_reio_masked (we forced same pixel mask/dimensions earlier L413 check)
    m_fid_pix = d_pix - np.asarray(diff_fid, dtype=float)
    m_reio_pix = d_pix - np.asarray(diff_reio, dtype=float)
    # Per-pixel marginal sigma: sqrt of diag of the (masked × masked) covariance block.
    # Must handle two cases: (a) vega._use_global_cov  OR  (b) per-corr cov_mat on data_obj.
    if vega_fid._use_global_cov:
        # Find the slice of global masked cov that belongs to our corr (same logic as L200-206 above)
        _gs = 0
        _corr_slice = None
        for _k, _name in enumerate(vega_fid.corr_items):
            # data vector length for this corr (already masked -> equals diff_parts[k].size)
            _obj = vega_fid.data[_name]
            _len = int(np.count_nonzero(_obj.data_mask) if isinstance(_obj.data_mask, np.ndarray)
                       else _obj.data_mask.sum())
            if _name == args.corr:
                _corr_slice = slice(_gs, _gs + _len)
            _gs += _len
        if _corr_slice is None:
            raise RuntimeError(f"{args.corr} slice in global masked cov not found.")
        global_cov_full = getattr(vega_fid, 'global_cov', None)
        if global_cov_full is None:
            # Fall back: build diagonal from global masked invcov diag pseudo-inverse (over-pessimistic)
            print("  Warning: global_cov not accessible; per-pixel sigma set to 0 for inset err bars.")
            err_pix = np.zeros_like(d_pix)
        else:
            # global_cov_full is un-masked size N×N; apply data_mask to rows/cols then pick corr_slice
            # First, build combined data_mask in global order (concatenate masks in corr_items order)
            gmasks = []
            for _n in vega_fid.corr_items:
                _obj = vega_fid.data[_n]
                gm = np.asarray(_obj.data_mask, dtype=bool)
                if gm.ndim == 0:
                    gm = np.full(int(np.asarray(_obj.data_vec).size), bool(gm))
                gmasks.append(gm)
            combined_mask = np.concatenate(gmasks)
            # Row+column apply combined mask -> get masked_cov_global size = (masked_N, masked_N)
            masked_cov_global = global_cov_full[combined_mask][:, combined_mask]
            corr_cov = masked_cov_global[_corr_slice][:, _corr_slice]
            diag_corr_cov = np.diag(corr_cov)
            diag_corr_cov = np.where(diag_corr_cov < 0.0, 0.0, diag_corr_cov)
            err_pix = np.sqrt(diag_corr_cov).astype(float)
    else:
        # (b) per-corr cov_mat (size N_pixel_unmasked × N_pixel_unmasked -> apply data_mask rows+cols)
        cov_per_corr_full = np.asarray(data_obj_fid_corr.cov_mat, dtype=float)
        dmask = np.asarray(data_obj_fid_corr.data_mask, dtype=bool)
        if dmask.ndim == 0:
            dmask = np.full(cov_per_corr_full.shape[0], bool(dmask))
        cov_masked = cov_per_corr_full[dmask][:, dmask]
        diag_m = np.diag(cov_masked)
        diag_m = np.where(diag_m < 0.0, 0.0, diag_m)
        err_pix = np.sqrt(diag_m).astype(float)
    # Safety ensure dimensions (should already match diff_fid exactly)
    if err_pix.size != d_pix.size:
        print(f"  Warning: err_pix dim {err_pix.size} != d_pix {d_pix.size}; fallback 0 err.")
        err_pix = np.zeros_like(d_pix)

    # Verify total chi2 sum against vega chi2 (data-only)
    saved_priors_fid = vega_fid.priors
    vega_fid.priors = {}
    total_fid_vega = float(vega_fid.chi2(local_fid))
    vega_fid.priors = saved_priors_fid
    saved_priors_reio = vega_reio.priors
    vega_reio.priors = {}
    total_reio_vega = float(vega_reio.chi2(local_reio))
    vega_reio.priors = saved_priors_reio

    sum_pix_fid = float(np.sum(per_pix_chi2_fid))
    sum_pix_reio = float(np.sum(per_pix_chi2_reio))
    print(f"  FID   total chi2 (pixelsum) = {sum_pix_fid:.4f}, (Vega.data-only) = {total_fid_vega:.4f}")
    print(f"  REIO  total chi2 (pixelsum) = {sum_pix_reio:.4f}, (Vega.data-only) = {total_reio_vega:.4f}")
    print(f"  Delta chi2 (REIO-FID)       = {sum_pix_reio - sum_pix_fid:.4f} "
          f"(Vega diff = {total_reio_vega - total_fid_vega:.4f})")

    chi2_fid, _ = bin_per_pixel_chi2_into_wedges(
        r_px, mu_px, per_pix_chi2_fid, mu_bins, r_min, r_max, n_r
    )
    chi2_reio, _ = bin_per_pixel_chi2_into_wedges(
        r_px, mu_px, per_pix_chi2_reio, mu_bins, r_min, r_max, n_r
    )
    delta_chi2 = chi2_reio - chi2_fid   # negative where reio improves (lower chi2)

    # ====== 3. Output all points and find Top-K ======
    print("\n[3/5] Listing all wedge bins and picking the Top-%d most improved." % args.top_k)

    rows = []
    for i, (mulo, muhi) in enumerate(mu_bins):
        for j in range(n_r):
            rows.append({
                'wedge_idx': i,
                'mu_min': mulo,
                'mu_max': muhi,
                'r_bin_idx': j,
                'r_center': float(r_centers[j]),
                'r_bin_lo': float(r_edges[j]),
                'r_bin_hi': float(r_edges[j + 1]),
                'chi2_fid': float(chi2_fid[i, j]),
                'chi2_reio': float(chi2_reio[i, j]),
                'delta_chi2_reio_minus_fid': float(delta_chi2[i, j]),
            })
    fieldnames = [
        'wedge_idx',
        'mu_min',
        'mu_max',
        'r_bin_idx',
        'r_center',
        'r_bin_lo',
        'r_bin_hi',
        'chi2_fid',
        'chi2_reio',
        'delta_chi2_reio_minus_fid',
    ]

    # Filename tag prevents overwriting when rerunning with different --nr / --plot-rmin / --plot-rmax
    # (was the root cause of "all.csv bins != top.csv bins" because all was from an older run.)
    fn_tag = f"{args.corr}_nr{args.nr}_rmin{int(round(args.plot_rmin))}_rmax{int(round(args.plot_rmax))}"

    all_csv = os.path.join(out_dir, f"point_chi2_all_{fn_tag}.csv")
    with open(all_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Saved ALL per-bin delta chi2 table -> {all_csv}")

    # Only bins within FITTING range [fit_rmin, fit_rmax] are eligible for Top-K ranking.
    eligible = [
        r for r in rows
        if fit_rmin - 1e-9 <= r['r_center'] <= fit_rmax + 1e-9
    ]
    if len(eligible) == 0:
        raise RuntimeError(
            f"No bins in fitting range [{fit_rmin:.1f}, {fit_rmax:.1f}]; "
            f"check --plot-rmin/--plot-rmax and --nr."
        )
    # Sorted by delta_chi2 (ascending): most negative (biggest improvement) first.
    sorted_rows = sorted(eligible, key=lambda r: r['delta_chi2_reio_minus_fid'])
    top_rows = list(sorted_rows[:args.top_k])
    top_csv = os.path.join(out_dir, f"point_chi2_top{args.top_k}_{fn_tag}.csv")
    with open(top_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in top_rows:
            writer.writerow(row)
    print(f"  Saved Top-{args.top_k} bins -> {top_csv}")
    print(f"\n  Top-{args.top_k} bins (only within r ∈ [{fit_rmin:.0f},{fit_rmax:.0f}],"
          f" smallest delta_chi2 = biggest REIO improvement):")
    for k, row in enumerate(top_rows):
        print(
            f"    rank{k+1:>2d}  μ∈({row['mu_min']:.2f},{row['mu_max']:.2f})  "
            f"j={row['r_bin_idx']:<3d} r={row['r_center']:.1f}  "
            f"Δχ²={row['delta_chi2_reio_minus_fid']:+.3f}  "
            f"(χ²_fid={row['chi2_fid']:.3f} → χ²_reio={row['chi2_reio']:.3f})"
        )

    # ====== Top-TOP-5: per-pixel decomposition inside the best-improved bins ======
    # User wants: among the top (most improved) bins, further take the TOP 5 bins with the
    # largest |Δχ²| improvement, and for EACH of those 5 bins list every real data pixel
    # (from compute_diff_y_r_mu_for_corr) that fell inside it, together with that pixel's
    # own per-pixel Δχ² contribution. This lets the user diagnose whether the per-bin
    # improvement comes from a single outlier pixel, or from a uniform average improvement
    # across all pixels inside the bin.
    #
    # Strategy: reuse the same (mu_bin, r_edges[j..j+1]) definition as bin_per_pixel_chi2_into_wedges
    # (L227 mu_vals in [mulo,muhi] closed AND L231 r_sub in [r_edges[j], r_edges[j+1[ half-open),
    # so that the sum of per-pixel Δχ² listed here EXACTLY equals row['delta_chi2_reio_minus_fid']
    # (this is verified by the cross-check printed at the end of the block for each bin).
    n_top_pixel_decompose = max(int(args.top_k), 5)  # cover all red-marked bins so each can be diagnosed.
    # First, gather the true top-5 bins (by delta_chi2 asc = most negative first). These
    # come from eligible (same fit-range-restricted set that feeds top_rows), not the full
    # rows, because user is ranking the top-improved bins in the "qualifying" fitting region.
    top5_bins_for_pixel = list(sorted_rows[:n_top_pixel_decompose])
    per_pixel_delta = per_pix_chi2_reio - per_pix_chi2_fid  # each pixel's own Δ contribution.
    r_edges_np = np.asarray(r_edges, dtype=float)

    # Build the pixel rows list that goes into the new per-pixel CSV.
    top5_pixel_fieldnames = [
        'global_bin_rank',        # 1..n_top_pixel_decompose: 1 = biggest total Δχ² improvement
        'wedge_idx',
        'mu_min',
        'mu_max',
        'r_bin_idx',
        'r_center',
        'r_bin_lo',
        'r_bin_hi',
        'total_bin_delta_chi2',   # = row['delta_chi2_reio_minus_fid'] (the per-bin total we wrote to top CSV)
        'n_pixels_in_bin',        # how many real data pixels fell inside this bin
        'pixel_local_rank_in_bin',# 1..N inside this bin (best improvement = most negative first)
        'pixel_r',                # this pixel's true r (from data_coordinates.r_regular_grid[mask])
        'pixel_mu',               # this pixel's true |μ| = rp/r (as defined L182/L217)
        'pixel_per_pix_chi2_fid',
        'pixel_per_pix_chi2_reio',
        'pixel_delta_chi2_reio_minus_fid',  # = per_pixel_delta[p] (this row's contribution to total)
        'frac_of_bin_total',      # pixel_delta / total_bin_delta (negative => this pixel drove the improvement;
                                  # close to 1/N => uniform average; 1 => single pixel dominates.)
    ]
    top5_pixel_rows = []
    print(
        f"\n  [Top-{n_top_pixel_decompose}-bins per-pixel Δχ² breakdown]  "
        f"(bins ordered by global Δχ² best→worst; pixels per bin ordered by their own Δχ² best→worst)\n"
        f"    Note: frac_of_bin_total = pixel_delta / total_bin_delta.  "
        f"≈ uniform ~1/N → average improvement; single pixel ≈ 1 → one pixel dominates."
    )
    for g_rank, bin_row in enumerate(top5_bins_for_pixel, start=1):
        i_w = int(bin_row['wedge_idx'])
        j_r = int(bin_row['r_bin_idx'])
        mulo = float(bin_row['mu_min'])
        muhi = float(bin_row['mu_max'])
        r_lo = float(bin_row['r_bin_lo'])
        r_hi = float(bin_row['r_bin_hi'])
        total_bin_delta = float(bin_row['delta_chi2_reio_minus_fid'])
        # EXACT same masks as bin_per_pixel_chi2_into_wedges L227 + L231 so sum(pixels)==total_bin_delta.
        in_mu = (mu_px >= mulo) & (mu_px <= muhi)
        # r_edges_np uses half-open [lo,hi[ on the r axis (per bin_per_pixel L231). For the very
        # last r bin (j == n_r-1), we include the == r_hi endpoint so pixels exactly at r_max
        # don't get dropped (otherwise a single pixel at r_max would miss the bin and the
        # sum check below would fail by exactly that pixel's delta).
        if j_r == n_r - 1:
            in_r = (r_px >= r_lo) & (r_px <= r_hi + 1e-12)
        else:
            in_r = (r_px >= r_lo) & (r_px < r_hi)
        bin_mask = in_mu & in_r
        pixel_ids = np.where(bin_mask)[0]
        n_pix = int(pixel_ids.size)
        # Compute local ranking inside this bin: most-negative (best) pixel_delta first.
        local_deltas = per_pixel_delta[pixel_ids]
        local_sort = np.argsort(local_deltas, kind='stable')  # asc → best first
        ordered_ids = pixel_ids[local_sort]
        ordered_r = r_px[ordered_ids]
        ordered_mu = mu_px[ordered_ids]
        ordered_fid = per_pix_chi2_fid[ordered_ids]
        ordered_reio = per_pix_chi2_reio[ordered_ids]
        ordered_delta = local_deltas[local_sort]
        # Cross-check: pixel sum should match the per-bin total we reported in the CSV (within
        # floating rounding; if not, print a warning so the user does not misinterpret the list).
        sum_check = float(np.sum(ordered_delta))
        match_str = "OK" if abs(sum_check - total_bin_delta) < max(1e-6, 1e-5 * abs(total_bin_delta) + 1e-9) \
                    else f"MISMATCH sum_pix={sum_check:+.5f} vs bin_total={total_bin_delta:+.5f}"
        print(
            f"\n    Bin global rank #{g_rank}:  μ∈({mulo:.2f},{muhi:.2f})  "
            f"j={j_r:<3d} r_c={bin_row['r_center']:.1f}  "
            f"total Δχ²={total_bin_delta:+.4f}  (N_pixels={n_pix})  sum_check: {match_str}"
        )
        if n_pix == 0:
            print("      (no pixels fell inside this bin; check masks/coordinates).")
            continue
        # Table-like short terminal print to eyeball quickly; full detail lives in the CSV below.
        max_show = min(8, n_pix)
        header_line = (
            f"      {'pix#':>4s}  {'r':>7s}  {'|mu|':>6s}  "
            f"{'chi2_fid':>9s}  {'chi2_reio':>10s}  {'delta':>9s}  {'frac':>7s}"
        )
        print(header_line)
        for loc_rank_1based, p in enumerate(range(max_show), start=1):
            d = float(ordered_delta[p])
            if abs(total_bin_delta) > 1e-15:
                frac = d / total_bin_delta  # both same sign when this pixel contributes to improvement
            else:
                frac = float('nan')
            print(
                f"      {loc_rank_1based:>4d}  {float(ordered_r[p]):>7.3f}  "
                f"{float(ordered_mu[p]):>6.3f}  "
                f"{float(ordered_fid[p]):>9.4f}  {float(ordered_reio[p]):>10.4f}  "
                f"{d:>+9.4f}  {frac:>7.3f}"
            )
        if n_pix > max_show:
            print(f"      ... +{n_pix - max_show} more pixels (see per-pixel CSV for full list).")
        # --- Covariance-effect diagnostic: decompose chi2 = diff * (C^-1 @ diff) ---
        # Per-pixel chi2 = diff[p] * y[p], where y = inv_cov . diff.
        # A pixel can have |diff_reio| > |diff_fid| (model visually further from data)
        # yet chi2_reio < chi2_fid (better fit) IF y_reio is smaller / opposite-signed.
        # This block quantifies exactly that for the current bin.
        _d_f = np.asarray(diff_fid, dtype=float)[ordered_ids]
        _d_r = np.asarray(diff_reio, dtype=float)[ordered_ids]
        _y_f = np.asarray(y_fid, dtype=float)[ordered_ids]
        _y_r = np.asarray(y_reio, dtype=float)[ordered_ids]
        _ad_f = np.abs(_d_f)
        _ad_r = np.abs(_d_r)
        # Pixels where reio |diff| is LARGER but reio chi2 is SMALLER -> covariance-driven.
        _cov_mask = (_ad_r > _ad_f) & (ordered_reio < ordered_fid)
        # Pixels where reio |diff| is SMALLER (visually better) and chi2 smaller (naive improvement).
        _naive_mask = (_ad_r <= _ad_f) & (ordered_reio < ordered_fid)
        print(
            f"      [cov-diag] N_pix={n_pix}  "
            f"reio|diff|>fid & reio_chi2<fid_chi2 (covariance-driven): {int(np.sum(_cov_mask))}  |  "
            f"reio|diff|<=fid & reio_chi2<fid (naive): {int(np.sum(_naive_mask))}"
        )
        print(
            f"                 sum|diff|_fid={np.sum(_ad_f):.4f}  sum|diff|_reio={np.sum(_ad_r):.4f}  "
            f"ratio={np.sum(_ad_r)/max(np.sum(_ad_f),1e-30):.3f}"
        )
        # Per-pixel detail table (diff, y=C^-1diff, chi2) — show all pixels in this bin
        # so the user can see exactly which pixels drive the improvement and why.
        print(
            f"      {'pix':>4s}  {'r':>7s}  "
            f"{'diff_fid':>10s}  {'diff_reio':>10s}  "
            f"{'y_fid':>10s}  {'y_reio':>10s}  "
            f"{'chi2_fid':>9s}  {'chi2_reio':>10s}  why"
        )
        for _p in range(n_pix):
            _df = float(_d_f[_p]); _dr = float(_d_r[_p])
            _yf = float(_y_f[_p]); _yr = float(_y_r[_p])
            _cf = float(ordered_fid[_p]); _cr = float(ordered_reio[_p])
            if _cov_mask[_p]:
                _why = "COV"
            elif _naive_mask[_p]:
                _why = "naive"
            else:
                _why = "worse"
            print(
                f"      {_p+1:>4d}  {float(ordered_r[_p]):>7.3f}  "
                f"{_df:>+10.5f}  {_dr:>+10.5f}  "
                f"{_yf:>+10.5f}  {_yr:>+10.5f}  "
                f"{_cf:>+9.4f}  {_cr:>+10.4f}  {_why}"
            )
        # Flush the full list (all n_pix, not just max_show) into CSV rows.
        for loc_rank_1based, p in enumerate(range(n_pix), start=1):
            d = float(ordered_delta[p])
            frac = (d / total_bin_delta) if abs(total_bin_delta) > 1e-15 else float('nan')
            top5_pixel_rows.append({
                'global_bin_rank': g_rank,
                'wedge_idx': i_w,
                'mu_min': f"{mulo:.4f}",
                'mu_max': f"{muhi:.4f}",
                'r_bin_idx': j_r,
                'r_center': f"{float(bin_row['r_center']):.5f}",
                'r_bin_lo': f"{r_lo:.5f}",
                'r_bin_hi': f"{r_hi:.5f}",
                'total_bin_delta_chi2': f"{total_bin_delta:+.6f}",
                'n_pixels_in_bin': n_pix,
                'pixel_local_rank_in_bin': loc_rank_1based,
                'pixel_r': f"{float(ordered_r[p]):.6f}",
                'pixel_mu': f"{float(ordered_mu[p]):.6f}",
                'pixel_per_pix_chi2_fid': f"{float(ordered_fid[p]):+.6f}",
                'pixel_per_pix_chi2_reio': f"{float(ordered_reio[p]):+.6f}",
                'pixel_delta_chi2_reio_minus_fid': f"{d:+.6f}",
                'frac_of_bin_total': (f"{frac:.5f}" if np.isfinite(frac) else "nan"),
            })
    # Write the combined per-pixel CSV (single file for all top-5 bins). Use a top5-specific
    # file name so multiple runs with different nr/rmin/rmax don't overwrite each other.
    topk_pix_csv = os.path.join(
        out_dir,
        f"point_chi2_top{n_top_pixel_decompose}_pixels_{fn_tag}.csv"
    )
    with open(topk_pix_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=top5_pixel_fieldnames)
        writer.writeheader()
        for row in top5_pixel_rows:
            writer.writerow(row)
    print(f"\n  Saved per-pixel breakdown for top-{n_top_pixel_decompose} most-improved bins -> {topk_pix_csv}")

    # ====== 4. Compute wedge model curves + data points for plotting ======
    # NOTE: we now replicate replot_polychord's plotting pipeline EXACTLY:
    #   1. Compute coarse models via vega.compute_model(current_params, run_init=True).
    #      current_params = vega.params.copy(), then override only keys that exist in Polychord.
    #      (This is the same pattern as replot L457-462 / L540-544.)
    #   2. For each coarse model, use the exact same _choose_grids_for_vec search, then
    #      interpolate via RectBivariateSpline to a FINE grid (scale_factor=10, as in replot L500).
    #      This interpolation (and the fine-grid r_mesh volume-weights) is what makes replot's
    #      model lines smooth and consistent. Previously we skipped interpolation entirely
    #      and projected the coarse grid, which is the main reason the lines looked different.
    #   3. Project the FINE vector (with rt_mesh.flatten() weights) through Wedge(r=(r_min_plot,
    #      r_max_plot, n_rp_out), and r_min_plot/r_max_plot are args.plot_rmin/plot_rmax (not
    #      clipped 0..200 here; Wedge uses exactly those endpoints -- matches replot L726/L754/L779).
    #   4. data projection (compute_wedge_data_points) still r=(0,200,n_rp_out) as replot does.
    #   5. smooth sigma=0.5, multiply by r^2, plot '-'/':' lines, y-limits identical to replot.
    print("\n[4/5] Computing wedge model curves for 2x2 plot (replot_polychord pipeline).")

    # --- replot naming: r_min_plot / r_max_plot / n_rp_out = args.nr (replot L615-617 default is 50) ---
    r_min_plot = float(args.plot_rmin)
    r_max_plot = float(args.plot_rmax)
    # IMPORTANT: use the SAME n_r we used for chi2 binning / CSV so that data point rd[j]
    # corresponds EXACTLY to r_centers[j] (= row['r_bin_idx']). This is now args.nr (no longer hardcoded 50).
    n_rp_out = int(n_r)
    assert n_rp_out == int(args.nr), "Pipeline n_r consistency check failed"

    # Reference data object / coordinates.
    data_obj_ref = vega_fid.data[args.corr]
    rp_min = float(data_obj_ref.data_coordinates.rp_min)
    rp_max = float(data_obj_ref.data_coordinates.rp_max)
    rt_min = 0.0
    rt_max = float(data_obj_ref.data_coordinates.rt_max)
    n_rp = int(data_obj_ref.data_coordinates.rp_nbins)
    n_rt = int(data_obj_ref.data_coordinates.rt_nbins)
    data_obj_reio = vega_reio.data[args.corr]
    rp_min2 = float(data_obj_reio.data_coordinates.rp_min)
    rp_max2 = float(data_obj_reio.data_coordinates.rp_max)
    rt_max2 = float(data_obj_reio.data_coordinates.rt_max)
    n_rp2 = int(data_obj_reio.data_coordinates.rp_nbins)
    n_rt2 = int(data_obj_reio.data_coordinates.rt_nbins)

    # --- Compute FID coarse model and interpolate to fine grid ---
    print("  [plot-pipeline] FID: vega.compute_model on vega.params override keys.")
    current_fid = vega_fid.params.copy()
    for k, v in params_fid.items():
        if k in current_fid:
            current_fid[k] = v
    models_fid = vega_fid.compute_model(current_fid, run_init=True)
    if args.corr not in models_fid:
        raise RuntimeError(f"Fid model missing {args.corr}")
    m1_coarse = models_fid[args.corr]

    # Replicate replot's _choose_grids_for_vec + RectBivariateSpline interpolation for FID.
    rp_g0, rt_g0 = _choose_grids_for_vec(data_obj_ref, m1_coarse.size)
    if rp_g0 is not None and rt_g0 is not None:
        rp_centers0 = np.unique(rp_g0)
        rt_centers0 = np.unique(rt_g0)
        scale_factor = 10
        try:
            grid0 = m1_coarse.reshape((rp_centers0.size, rt_centers0.size))
            interp_func = RectBivariateSpline(rp_centers0, rt_centers0, grid0, kx=1, ky=1)
            n_rp_fine = rp_centers0.size * scale_factor
            n_rt_fine = rt_centers0.size * scale_factor
            drp0 = rp_centers0[1] - rp_centers0[0] if rp_centers0.size > 1 else (rp_max - rp_min)
            drt0 = rt_centers0[1] - rt_centers0[0] if rt_centers0.size > 1 else (rt_max - rt_min)
            rp_centers_active = np.linspace(
                rp_centers0[0] - drp0/2 + drp0/(2*scale_factor),
                rp_centers0[-1] + drp0/2 - drp0/(2*scale_factor),
                n_rp_fine,
            )
            rt_centers_active = np.linspace(
                rt_centers0[0] - drt0/2 + drt0/(2*scale_factor),
                rt_centers0[-1] + drt0/2 - drt0/(2*scale_factor),
                n_rt_fine,
            )
            new_grid = interp_func(rp_centers_active, rt_centers_active)
            m1_fine = new_grid.flatten()
        except Exception:
            m1_fine = m1_coarse
            n_rp_fine = rp_centers0.size
            n_rt_fine = rt_centers0.size
            rp_centers_active = rp_centers0.copy()
            rt_centers_active = rt_centers0.copy()
    else:
        scale_factor = 10
        m1_fine, n_rp_fine, n_rt_fine = interpolate_to_finer_grid(
            m1_coarse, n_rp, n_rt, rp_min, rp_max, rt_min, rt_max, scale_factor
        )
        rp_centers_active = np.linspace(
            rp_min + (rp_max - rp_min)/(2*n_rp_fine),
            rp_max - (rp_max - rp_min)/(2*n_rp_fine),
            n_rp_fine,
        )
        rt_centers_active = np.linspace(
            rt_min + (rt_max - rt_min)/(2*n_rt_fine),
            rt_max - (rt_max - rt_min)/(2*n_rt_fine),
            n_rt_fine,
        )
    L1 = m1_fine.size
    if L1 != n_rp_fine * n_rt_fine:
        print(f"  Warning: FID model length {L1} != n_rp_fine*n_rt_fine ({n_rp_fine*n_rt_fine}); reconciling.")
        if L1 % n_rt_fine == 0:
            n_rp_fine = L1 // n_rt_fine
        elif L1 % n_rp_fine == 0:
            n_rt_fine = L1 // n_rp_fine

    # --- Compute REIO coarse model and interpolate to fine grid (matching replot L535-601) ---
    print("  [plot-pipeline] REIO: vega.compute_model on vega.params override keys.")
    current_reio = vega_reio.params.copy()
    for k, v in params_reio.items():
        if k in current_reio:
            current_reio[k] = v
    models_reio = vega_reio.compute_model(current_reio, run_init=True)
    if args.corr not in models_reio:
        raise RuntimeError(f"Reio model missing {args.corr}")
    m2_coarse = models_reio[args.corr]

    rp_centers_active2 = None
    rt_centers_active2 = None
    data_obj_local_reio = data_obj_reio
    rp_g02, rt_g02 = _choose_grids_for_vec(data_obj_local_reio, m2_coarse.size)
    if rp_g02 is not None and rt_g02 is not None:
        rp_centers02 = np.unique(rp_g02)
        rt_centers02 = np.unique(rt_g02)
        try:
            grid02 = m2_coarse.reshape((rp_centers02.size, rt_centers02.size))
            interp_func2 = RectBivariateSpline(rp_centers02, rt_centers02, grid02, kx=1, ky=1)
            n_rp_fine2 = rp_centers02.size * scale_factor
            n_rt_fine2 = rt_centers02.size * scale_factor
            drp02 = rp_centers02[1] - rp_centers02[0] if rp_centers02.size > 1 else (rp_max2 - rp_min2)
            drt02 = rt_centers02[1] - rt_centers02[0] if rt_centers02.size > 1 else (rt_max2 - rt_min)
            rp_centers_active2 = np.linspace(
                rp_centers02[0] - drp02/2 + drp02/(2*scale_factor),
                rp_centers02[-1] + drp02/2 - drp02/(2*scale_factor),
                n_rp_fine2,
            )
            rt_centers_active2 = np.linspace(
                rt_centers02[0] - drt02/2 + drt02/(2*scale_factor),
                rt_centers02[-1] + drt02/2 - drt02/(2*scale_factor),
                n_rt_fine2,
            )
            new_grid2 = interp_func2(rp_centers_active2, rt_centers_active2)
            m2_fine = new_grid2.flatten()
        except Exception:
            m2_fine = m2_coarse
            rp_centers_active2 = rp_centers02.copy()
            rt_centers_active2 = rt_centers02.copy()
    else:
        m2_fine = m2_coarse
        rp_centers_active2 = rp_centers_active.copy()
        rt_centers_active2 = rt_centers_active.copy()

    # Reconcile grid for model 2 (replot L578-601)
    L2 = m2_fine.size
    if L2 != n_rp_fine * n_rt_fine:
        print(f"  Warning: REIO model length {L2} != n_rp_fine*n_rt_fine ({n_rp_fine*n_rt_fine}); reconciling.")
        if L2 % n_rt_fine == 0:
            n_rp_fine = L2 // n_rt_fine
            print(f"  Adjusted n_rp_fine to {n_rp_fine} (n_rt_fine stays {n_rt_fine}).")
        elif L2 % n_rp_fine == 0:
            n_rt_fine = L2 // n_rp_fine
            print(f"  Adjusted n_rt_fine to {n_rt_fine}.")
        else:
            best_n_rp = None
            for k in range(1, int(np.sqrt(L2)) + 1):
                if L2 % k == 0:
                    if best_n_rp is None or abs(k - n_rp_fine) < abs(best_n_rp - n_rp_fine):
                        best_n_rp = k
                    if abs((L2 // k) - n_rp_fine) < abs(best_n_rp - n_rp_fine):
                        best_n_rp = L2 // k
            if best_n_rp is not None:
                n_rp_fine = best_n_rp
                n_rt_fine = L2 // best_n_rp
                print(f"  Guessed grid2 to ({n_rp_fine}, {n_rt_fine}).")

    # Build Top-K mask (i, rd_index) using rd (full 0..200 range). Done later per wedge.

    # ====== 5. Plot 2x2 wedges, marking Top-K in red (EXACT replot wedge look) ======
    print("\n[5/5] Plotting 2x2 wedges with Top-%d improvement points in red." % args.top_k)

    # rcParams: appearance base (5par.mplstyle tick layout) + *smaller font sizes* than replot.
    # User requested reducing axis/tick fonts to avoid overlap between adjacent subplot labels,
    # legend, curves, and axis text. Sizes are scaled down proportionally from the original
    # 18/24/22/22/20 set so visual hierarchy is preserved.
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'axes.labelsize': 21})
    plt.rcParams.update({'xtick.labelsize': 15})
    plt.rcParams.update({'ytick.labelsize': 15})
    plt.rcParams.update({'legend.fontsize': 16})
    plt.rcParams.update({'lines.linewidth': 2.5})
    plt.rcParams.update({'xtick.top': True})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.right': True})
    plt.rcParams.update({'ytick.direction': 'in'})
    plt.rcParams.update({'legend.frameon': False})
    plt.rcParams.update({'savefig.pad_inches': 0.08})

    fig, axes_grid = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes_grid.flatten()
    smooth_sigma = 0.5
    # Cache per-wedge data points and Top-K mask for the post-loop zoom section:
    #   - the 4-wedge loop runs before our per-pixel inset drawing, so loop-local rd/data_y/err
    #     are inaccessible from outside; caching them in lists avoids re-computing Wedge() twice.
    #   - rd_list[i][j] / data_y_list[i][j] is exactly the main-plot red point at wedge i, r bin j.
    rd_list = [None for _ in range(4)]
    data_y_list = [None for _ in range(4)]
    data_err_list = [None for _ in range(4)]
    ylim_list = [None for _ in range(4)]

    for i, (mu_min, mu_max) in enumerate(mu_bins):
        ax = axes[i]
        ax.grid(linestyle=':', alpha=0.5)
        # Axis-level tick layout (5par.mplstyle: ticks on all 4 edges + inward direction).
        # Match the reduced rcParams tick labelsize so axis text does not crowd/overlap the
        # model curves, data points, or neighboring subplot labels.
        ax.tick_params(which='both', top=True, right=True, direction='in',
                       length=6, width=1.2, labelsize=15)
        fit_rmin = max(0.0, min(200.0, r_min_plot))
        fit_rmax = max(0.0, min(200.0, r_max_plot))
        if fit_rmin > 0.0:
            ax.axvspan(0.0, fit_rmin, color="0.85", alpha=0.6, zorder=0)
        if fit_rmax < 200.0:
            ax.axvspan(fit_rmax, 200.0, color="0.85", alpha=0.6, zorder=0)
        ax.set_xlim(0, 200)

        # Project FID (and optionally REIO) fine vectors -> wedge r bins,
        # with Wedge construction + volume weights exactly as replot L732-783.
        rp_centers = rp_centers_active
        rt_centers = rt_centers_active
        n_rp_vec = rp_centers.size
        n_rt_vec = rt_centers.size
        if n_rp_vec > 1:
            drp_vec = rp_centers[1] - rp_centers[0]
        else:
            drp_vec = (rp_max - rp_min) / max(1, n_rp_out)
        if n_rt_vec > 1:
            drt_vec = rt_centers[1] - rt_centers[0]
        else:
            drt_vec = (rt_max - rt_min) / max(1, n_rp_out)
        rp_min_edge = rp_centers[0] - drp_vec/2
        rp_max_edge = rp_centers[-1] + drp_vec/2
        rt_min_edge = max(0.0, rt_centers[0] - drt_vec/2)
        rt_max_edge = rt_centers[-1] + drt_vec/2
        rt_mesh_vec, rp_mesh_vec = np.meshgrid(rt_centers, rp_centers)
        vol_weights_vec = rt_mesh_vec.flatten()
        w = Wedge(mu=(mu_min, mu_max),
                  rp=(rp_min_edge, rp_max_edge, n_rp_vec),
                  rt=(rt_min_edge, rt_max_edge, n_rt_vec),
                  r=(r_min_plot, r_max_plot, n_rp_out),
                  abs_mu=True)
        r, wedge1 = w(m1_fine, weights=vol_weights_vec)

        wedge2 = None
        rp_centers2 = rp_centers_active2 if rp_centers_active2 is not None else rp_centers
        rt_centers2 = rt_centers_active2 if rt_centers_active2 is not None else rt_centers
        if (rp_centers2.size != n_rp_vec or rt_centers2.size != n_rt_vec
                or np.any(np.abs(rp_centers2 - rp_centers) > 1e-8)
                or np.any(np.abs(rt_centers2 - rt_centers) > 1e-8)):
            rt_mesh_vec2, rp_mesh_vec2 = np.meshgrid(rt_centers2, rp_centers2)
            vol_weights_vec2 = rt_mesh_vec2.flatten()
            if rp_centers2.size > 1:
                drp_vec2 = rp_centers2[1] - rp_centers2[0]
            else:
                drp_vec2 = drp_vec
            if rt_centers2.size > 1:
                drt_vec2 = rt_centers2[1] - rt_centers2[0]
            else:
                drt_vec2 = drt_vec
            rp_min_edge2 = rp_centers2[0] - drp_vec2/2
            rp_max_edge2 = rp_centers2[-1] + drp_vec2/2
            rt_min_edge2 = max(0.0, rt_centers2[0] - drt_vec2/2)
            rt_max_edge2 = rt_centers2[-1] + drt_vec2/2
            w2 = Wedge(mu=(mu_min, mu_max),
                       rp=(rp_min_edge2, rp_max_edge2, rp_centers2.size),
                       rt=(rt_min_edge2, rt_max_edge2, rt_centers2.size),
                       r=(r_min_plot, r_max_plot, n_rp_out),
                       abs_mu=True)
            _, wedge2 = w2(m2_fine, weights=vol_weights_vec2)
        else:
            _, wedge2 = w(m2_fine, weights=vol_weights_vec)

        # --- Data points (full 0..200) + Top-10 mask ---
        rd, dd, errd = compute_wedge_data_points(data_obj_ref, (mu_min, mu_max), n_rp_out)
        data_y = dd * rd ** 2
        data_err = errd * rd ** 2

        # Since the chi2-binning grid (rows), the CSV output, and rd ALL share the SAME
        # r ∈ [0,200], n_r = n_rp_out grid, we can directly index via row['r_bin_idx'].
        # No nearest-neighbor argmin any more -> zero bin-matching drift.
        is_top_mask = np.zeros(rd.size, dtype=bool)
        assert rd.size == n_r, "rd.size != n_r; grid mismatch (check Wedge output dims)"
        for row in top_rows:
            if int(row['wedge_idx']) != i:
                continue
            j_idx = int(row['r_bin_idx'])
            is_top_mask[j_idx] = True
        is_other = ~is_top_mask

        # --- Plot: replot_polychord L796-816 EXACTLY ---
        y1 = gaussian_filter1d(wedge1, sigma=smooth_sigma) * r**2
        ax.plot(r, y1, '-', label='Without Reionization')
        # Non-top black points (elinewidth/capsize explicit for pixel-match with replot;
        # markeredgewidth left at matplotlib default ~1.0 to match replot L803 exactly)
        ax.errorbar(rd[is_other], data_y[is_other], yerr=data_err[is_other],
                    fmt='o', color='k', markersize=6, alpha=0.8, label='DESI DR1',
                    elinewidth=2.5, capsize=0)
        # Top-K: same markersize/elinewidth/capsize as black points, ONLY color -> pure red
        # (face + edge both red, no dark outline -> looks like a direct solid red dot)
        if np.any(is_top_mask):
            ax.errorbar(rd[is_top_mask], data_y[is_top_mask], yerr=data_err[is_top_mask],
                        fmt='o', color='red', markersize=6, alpha=1.0,
                        ecolor='red', label=r'Top-$\Delta\chi^2$ bins',
                        elinewidth=2.5, capsize=0)
        # With Reionization: ':' (matches replot L806)
        y2 = gaussian_filter1d(wedge2, sigma=smooth_sigma) * r**2
        ax.plot(r, y2, ':', label='With Reionization')

        # y-limits: replot L807-816
        y_min = np.nanmin([np.nanmin(y1), np.nanmin(y2)])
        y_max = np.nanmax([np.nanmax(y1), np.nanmax(y2)])
        if np.any(np.isfinite(data_y)):
            y_min = min(float(y_min), float(np.nanmin(data_y)))
            y_max = max(float(y_max), float(np.nanmax(data_y)))
        pad = 0.08 * max(1e-12, (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)
        # Cache final data + axes limits for the post-loop per-pixel zoom section.
        rd_list[i] = np.asarray(rd, dtype=float).copy()
        data_y_list[i] = np.asarray(data_y, dtype=float).copy()
        data_err_list[i] = np.asarray(data_err, dtype=float).copy()
        ylim_list[i] = (float(y_min - pad), float(y_max + pad))

        if i % 2 == 0:
            # Smaller labels + slightly tighter labelpad than before to match the reduced tick font sizes;
            # this prevents the label itself from overlapping ticks/obscuring the plot content.
            ax.set_ylabel(r'$r^2 \xi(r) \; [h^{-2}\,{\rm Mpc}^2]$', fontsize=18, labelpad=10)
        # Legend: in non-zoom mode, keep the original behaviour (last wedge, upper right).
        # In zoom mode, defer legend placement until after we know which wedges contain
        # zoom insets — the legend will then be placed in a wedge WITHOUT any inset, at
        # either upper-centre or lower-centre (whichever avoids the data).
        _legend_wedge_target = int(args.legend_wedge) if args.legend_wedge is not None else 3
        if not args.zoom and i == _legend_wedge_target:
            handles, labels = ax.get_legend_handles_labels()
            _dummy_eb = ax.errorbar([0], [0], yerr=[[0.1], [0.1]],
                                    fmt='o', color='red', markersize=6,
                                    ecolor='red', elinewidth=2.5, capsize=0,
                                    label=r'Top-$\Delta\chi^2$ bins')
            for _ch in _dummy_eb.get_children():
                try:
                    _ch.remove()
                except Exception:
                    pass
            if _dummy_eb in ax.containers:
                ax.containers.remove(_dummy_eb)
            red_proxy = _dummy_eb
            if r'Top-$\Delta\chi^2$ bins' not in labels:
                handles.append(red_proxy)
                labels.append(r'Top-$\Delta\chi^2$ bins')
            else:
                idx = labels.index(r'Top-$\Delta\chi^2$ bins')
                handles[idx] = red_proxy
            ax.legend(handles, labels, frameon=False, fontsize=16,
                      handler_map={ErrorbarContainer: HandlerErrorbar(numpoints=1)})
        # For qsoxlya / qsoxlyb the mu binning is done on |mu| (rp can be negative
        # for these correlations), so reflect that in the subplot title.
        if args.corr in ('qsoxlya', 'qsoxlyb'):
            title_text = r'${} < |\mu| < {}$'.format(mu_min, mu_max)
        else:
            title_text = r'${} < \mu < {}$'.format(mu_min, mu_max)
        ax.set_title(title_text, fontsize=22, pad=8)
        if i >= 2:
            ax.set_xlabel('r [Mpc/h]', fontsize=18, labelpad=8)

    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.96,
                        hspace=0.18, wspace=0.18)

    # ---- NEW ZOOM MODE: per-pixel scatter inset for Top-2 best-improved bins ----
    # User request 2026-10-09: only show details for top 2, not top 5; and keep both
    # insets in their NATIVE wedge panels (no cross-wedge borrowing).
    # Also: smaller default inset size, no duplicated axis names, enlarged Δχ²/Npix font.
    if args.zoom:
        lw = plt.rcParams.get('lines.linewidth', 2.5)
        # --- CLI inset geometry overrides ---
        # Slot 0 parameters (single-inset case = slot 0 only; N=2 shared-wedge = left slot).
        user_ix = float(args.inset_x) if hasattr(args, 'inset_x') and args.inset_x is not None else None
        user_iy = float(args.inset_y) if hasattr(args, 'inset_y') and args.inset_y is not None else None
        user_iw = float(args.inset_w) if hasattr(args, 'inset_w') and args.inset_w is not None else None
        user_ih = float(args.inset_h) if hasattr(args, 'inset_h') and args.inset_h is not None else None
        # Slot 1 parameters (only used when N=2 insets live in the same wedge -> right slot).
        user2_ix = float(args.inset2_x) if hasattr(args, 'inset2_x') and args.inset2_x is not None else None
        user2_iy = float(args.inset2_y) if hasattr(args, 'inset2_y') and args.inset2_y is not None else None
        user2_iw = float(args.inset2_w) if hasattr(args, 'inset2_w') and args.inset2_w is not None else None
        user2_ih = float(args.inset2_h) if hasattr(args, 'inset2_h') and args.inset2_h is not None else None
        r_edges_np = np.asarray(r_edges, dtype=float)
        n_r = int(r_edges_np.size - 1)

        # Slot layout helper: now only needs N=1 and N=2 (Top-2, no borrowing).
        def _auto_slots_zoom(n, wedge_idx=None):
            if n <= 0:
                return []
            LEGEND_WEDGE = 3
            is_legend_wedge = (wedge_idx == LEGEND_WEDGE) if (wedge_idx is not None) else False
            single_w, single_h = (0.41, 0.37)
            if n == 1:
                if is_legend_wedge:
                    # Keep single inset upper-left to avoid overlapping the legend box (upper-right).
                    return [(0.05, 0.62, single_w, single_h)]
                return [(0.14, 0.60, single_w, single_h)]
            if n == 2:
                # Same wedge has both top-2 insets: place them side-by-side on top row, slightly
                # smaller widths so they don't touch, with decent height.
                return [
                    (0.12, 0.60, 0.39, 0.35),
                    (0.58, 0.60, 0.39, 0.35),
                ]
            return []

        NUM_TOP_INSETS = min(5 if args.zoom_top5 else 2, int(args.top_k))
        # Match the grey-out / fitting-range clamp logic used by the main plot (L961-962):
        # even if user passes --plot-rmin=-3 or --plot-rmax=300 (nonsensical), we still clamp.
        r_min_plot = max(0.0, min(200.0, float(args.plot_rmin)))
        r_max_plot = max(0.0, min(200.0, float(args.plot_rmax)))

        def _bin_pixels_ok(row):
            """Return (ok, pix_idx, pr_arr): ok iff bin has pixels & all pixel_r within [r_min_plot, r_max_plot].

            Fast path first: if the aggregated bin itself is *fully* outside (r_bin_hi <= r_min_plot
            or r_bin_lo >= r_max_plot), reject immediately without touching the pixel arrays;
            if the bin straddles the boundary (r_bin_lo < r_min_plot or r_bin_hi > r_max_plot),
            we still need to scan actual pixels in case some masked-out pixel coordinate leaves
            the bin fully inside (rare, but safe and cheap).
            """
            r_lo = float(row['r_bin_lo'])
            r_hi = float(row['r_bin_hi'])
            j_r = int(row['r_bin_idx'])
            # Fast bin-level reject: bin clearly wholly outside the fitting range.
            if r_hi <= (r_min_plot - 1e-9) or r_lo >= (r_max_plot + 1e-9):
                return False, np.asarray([], dtype=int), None
            mulo = float(row['mu_min'])
            muhi = float(row['mu_max'])
            in_mu = (mu_px >= mulo) & (mu_px <= muhi)
            if j_r == n_r - 1:
                in_r = (r_px >= r_lo) & (r_px <= r_hi + 1e-12)
            else:
                in_r = (r_px >= r_lo) & (r_px < r_hi)
            mask = in_mu & in_r
            idx = np.where(mask)[0]
            if idx.size == 0:
                return False, idx, None
            pr = np.asarray(r_px[idx], dtype=float)
            if np.any(pr < (r_min_plot - 1e-9)) or np.any(pr > (r_max_plot + 1e-9)):
                return False, idx, pr
            return True, idx, pr

        zoom_bins = []
        original_ranks = []
        n_skipped_boundary = 0
        n_skipped_empty = 0
        n_scanned = 0
        for ri, row in enumerate(sorted_rows):
            orig_rank = ri + 1
            if len(zoom_bins) >= NUM_TOP_INSETS:
                break
            n_scanned += 1
            ok, pidx, pr_arr = _bin_pixels_ok(row)
            if not ok:
                if pidx.size == 0:
                    n_skipped_empty += 1
                    print(f"  Zoom candidate Top-{orig_rank} (w{int(row['wedge_idx'])}, j={int(row['r_bin_idx'])}): skipped — no pixels in mask.")
                else:
                    n_skipped_boundary += 1
                    if pr_arr is not None and pr_arr.size > 0:
                        print(f"  Zoom candidate Top-{orig_rank} (w{int(row['wedge_idx'])}, j={int(row['r_bin_idx'])}): skipped — pixel_r outside fitting range "
                              f"[{r_min_plot:.1f}, {r_max_plot:.1f}] (pr min={pr_arr.min():.2f}, max={pr_arr.max():.2f}).")
                    else:
                        print(f"  Zoom candidate Top-{orig_rank} (w{int(row['wedge_idx'])}, j={int(row['r_bin_idx'])}): skipped — bin outside fitting range.")
                continue
            zoom_bins.append(row)
            original_ranks.append(orig_rank)

        # Explicit selection summary so the user can immediately tell from stdout whether the
        # boundary-skip logic fired (avoids the previous "my figure looks identical -> is code working?" confusion).
        print(f"\n  ===== Zoom inset selection summary =====")
        print(f"  Target NUM_TOP_INSETS = {NUM_TOP_INSETS}")
        print(f"  Fitting / plot r range = [{r_min_plot:.1f}, {r_max_plot:.1f}]")
        print(f"  Scanned top {n_scanned} candidates from sorted delta-chi2 list")
        print(f"  Accepted : {len(zoom_bins)}  — original ranks: {original_ranks}")
        print(f"  Skipped  : {n_skipped_boundary} boundary-outside + {n_skipped_empty} empty-bin")
        if len(zoom_bins) < NUM_TOP_INSETS:
            print(f"  ⚠ Warning: only {len(zoom_bins)}/{NUM_TOP_INSETS} valid candidates found.")
        print(f"  ========================================\n")

        # NO cross-wedge borrowing. Count native insets per wedge; that is the final count.
        zoom_native_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _b in zoom_bins:
            zoom_native_counts[int(_b['wedge_idx'])] += 1
        wedge_slots = {iw: _auto_slots_zoom(zoom_native_counts[iw], wedge_idx=iw) for iw in range(4)}
        wedge_slot_cursor = {0: 0, 1: 0, 2: 0, 3: 0}

        # ---- Legend placement in zoom mode ----
        # Pick the first wedge that has NO zoom inset. If all 4 wedges have insets
        # (extremely unlikely with Top-5 and <=2 insets per wedge), fall back to
        # wedge 3. Then place the legend at upper-centre or lower-centre, whichever
        # has fewer data points in its horizontal band (to avoid blocking content).
        empty_wedges = [iw for iw in range(4) if zoom_native_counts[iw] == 0]
        if args.legend_wedge is not None:
            legend_wedge = int(args.legend_wedge)
        else:
            legend_wedge = empty_wedges[0] if empty_wedges else 3
        ax_leg = axes[legend_wedge]
        _leg_handles, _leg_labels = ax_leg.get_legend_handles_labels()
        _leg_dummy = ax_leg.errorbar([0], [0], yerr=[[0.1], [0.1]],
                                     fmt='o', color='red', markersize=6,
                                     ecolor='red', elinewidth=2.5, capsize=0,
                                     label=r'Top-$\Delta\chi^2$ bins')
        for _ch in _leg_dummy.get_children():
            try:
                _ch.remove()
            except Exception:
                pass
        if _leg_dummy in ax_leg.containers:
            ax_leg.containers.remove(_leg_dummy)
        _leg_red = _leg_dummy
        if r'Top-$\Delta\chi^2$ bins' not in _leg_labels:
            _leg_handles.append(_leg_red)
            _leg_labels.append(r'Top-$\Delta\chi^2$ bins')
        else:
            _idx = _leg_labels.index(r'Top-$\Delta\chi^2$ bins')
            _leg_handles[_idx] = _leg_red
        # Decide upper-centre vs lower-centre by counting data points in each half.
        _rd_leg = rd_list[legend_wedge]
        _dy_leg = data_y_list[legend_wedge]
        _ylo_leg, _yhi_leg = ylim_list[legend_wedge] if ylim_list[legend_wedge] is not None else ax_leg.get_ylim()
        _ymid_leg = 0.5 * (_ylo_leg + _yhi_leg)
        _n_upper = int(np.sum(_dy_leg[np.isfinite(_dy_leg)] >= _ymid_leg))
        _n_lower = int(np.sum(_dy_leg[np.isfinite(_dy_leg)] < _ymid_leg))
        _leg_loc = 'lower center' if _n_upper >= _n_lower else 'upper center'
        ax_leg.legend(_leg_handles, _leg_labels, loc=_leg_loc, frameon=False,
                      fontsize=16,
                      handler_map={ErrorbarContainer: HandlerErrorbar(numpoints=1)})
        print(f"  Legend placed in wedge {legend_wedge} (no zoom inset) at {_leg_loc}.")

        zoom_draw_plan = []
        for _gi, _b in enumerate(zoom_bins):
            native_iw = int(_b['wedge_idx'])
            zoom_draw_plan.append((_gi, _b, native_iw, False))

        # Track already-placed labels+arrows per wedge so subsequent labels can avoid
        # (a) overlapping another label's bbox, and (b) crossing another label's arrow.
        # Each entry: (txt_x, txt_y, xpt, ypt, box_half_x, box_half_y) in data coords.
        placed_per_wedge = {0: [], 1: [], 2: [], 3: []}

        def _segments_intersect(p1, p2, p3, p4):
            """Return True if open segment p1-p2 intersects open segment p3-p4.

            Uses orientation-based standard segment-intersection test.
            """
            def _orient(a, b, c):
                v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if abs(v) < 1e-12:
                    return 0
                return 1 if v > 0 else 2

            o1 = _orient(p1, p2, p3)
            o2 = _orient(p1, p2, p4)
            o3 = _orient(p3, p4, p1)
            o4 = _orient(p3, p4, p2)
            if o1 != o2 and o3 != o4:
                return True
            # Collinear cases are extremely unlikely for two random arrows and we
            # skip them to keep the test cheap and robust.
            return False

        for g_rank_0based, bin_row, i_draw_wedge, _borrowed in zoom_draw_plan:
            g_rank = original_ranks[g_rank_0based]
            i_w_native = int(bin_row['wedge_idx'])
            assert i_draw_wedge == i_w_native, "Top-N insets must live in their native wedge."
            j_r = int(bin_row['r_bin_idx'])
            mulo = float(bin_row['mu_min'])
            muhi = float(bin_row['mu_max'])
            r_lo = float(bin_row['r_bin_lo'])
            r_hi = float(bin_row['r_bin_hi'])
            total_bin_delta = float(bin_row['delta_chi2_reio_minus_fid'])

            # --- Place inset FIRST, so we know its exact axes box before choosing the label ---
            #     quadrant that maximises distance from the inset (avoids overlap).
            ax_draw_par = axes[i_draw_wedge]
            slots_this_wedge = wedge_slots[i_draw_wedge]
            cursor = wedge_slot_cursor[i_draw_wedge]
            if cursor < len(slots_this_wedge):
                bx0, by0, inset_w, inset_h = slots_this_wedge[cursor]
                if len(slots_this_wedge) == 2:
                    # Two insets share this wedge -> pick slot-0 or slot-1 CLI overrides by cursor.
                    slot_user_x = [user_ix, user2_ix]
                    slot_user_y = [user_iy, user2_iy]
                    slot_user_w = [user_iw, user2_iw]
                    slot_user_h = [user_ih, user2_ih]
                    ux = slot_user_x[cursor] if cursor < 2 else None
                    uy = slot_user_y[cursor] if cursor < 2 else None
                    uw = slot_user_w[cursor] if cursor < 2 else None
                    uh = slot_user_h[cursor] if cursor < 2 else None
                    if ux is not None:
                        bx0 = ux
                    if uy is not None:
                        by0 = uy
                    if uw is not None:
                        inset_w = uw
                    if uh is not None:
                        inset_h = uh
                elif len(slots_this_wedge) == 1 and (user_ix is not None or user_iy is not None
                                                     or user_iw is not None or user_ih is not None):
                    if user_ix is not None:
                        bx0 = user_ix
                    if user_iy is not None:
                        by0 = user_iy
                    if user_iw is not None:
                        inset_w = user_iw
                    if user_ih is not None:
                        inset_h = user_ih
                    # In --zoom-top5 mode, a wedge with only ONE inset gets a wider panel
                    # and a slight rightward shift so the single inset uses the extra
                    # horizontal space (the 2nd-slot area is free).
                    # User request 2026-10-11: width = 1.5 * --inset-w, x = --inset-x + 0.05.
                    if args.zoom_top5:
                        if user_iw is not None:
                            inset_w = 1.5 * user_iw
                        if user_ix is not None:
                            bx0 = user_ix + 0.15
            else:
                bx0, by0, inset_w, inset_h = (0.35, 0.07, 0.26, 0.28)
            # Inset box center in parent (native wedge) axes NORMALIZED coords (0..1):
            inset_cx_norm = bx0 + 0.5 * inset_w
            inset_cy_norm = by0 + 0.5 * inset_h

            # Pixel filter result is already checked by _bin_pixels_ok above, but recompute
            # for explicit indexing (safe & consistent, negligible cost).
            in_mu = (mu_px >= mulo) & (mu_px <= muhi)
            if j_r == n_r - 1:
                in_r = (r_px >= r_lo) & (r_px <= r_hi + 1e-12)
            else:
                in_r = (r_px >= r_lo) & (r_px < r_hi)
            bin_mask = in_mu & in_r
            pix_idx = np.where(bin_mask)[0]
            if pix_idx.size == 0:
                print(f"  Top-{g_rank} inset skipped (w{i_w_native}, j={j_r}): no pixels in mask.")
                continue

            pr = np.asarray(r_px[pix_idx], dtype=float)
            pmu = np.asarray(mu_px[pix_idx], dtype=float)
            pd = np.asarray(d_pix[pix_idx], dtype=float)
            pm1 = np.asarray(m_fid_pix[pix_idx], dtype=float)
            pm2 = np.asarray(m_reio_pix[pix_idx], dtype=float)
            perr = np.asarray(err_pix[pix_idx], dtype=float)
            pdelta = np.asarray(per_pixel_delta[pix_idx], dtype=float)
            # --- De-duplicate identical pixel_r: keep only the largest-contributor. ---
            # When two or more pixels in this bin share the exact same pixel_r (e.g. a
            # pixel pair with opposite mu that both fold into the same |mu| bin), the
            # model curve would plot as a vertical segment overlapping the error bar.
            # User request 2026-10-13: instead of horizontally shifting duplicates, keep
            # ONLY the pixel whose per-pixel delta_chi^2 is most negative (largest
            # improvement) within each same-r group. N_pix (computed from pix_idx.size
            # below) still reports the full pixel count, so the user knows how many
            # real pixels were collapsed into each displayed point.
            order = np.lexsort((pmu, pr))
            pr_o = pr[order]
            pd_o = pd[order]
            pm1_o = pm1[order]
            pm2_o = pm2[order]
            perr_o = perr[order]
            pdelta_o = pdelta[order]
            keep_idx = []
            _n_o = pr_o.size
            _i = 0
            while _i < _n_o:
                _j = _i
                while _j < _n_o and pr_o[_j] == pr_o[_i]:
                    _j += 1
                _best_local = int(np.argmin(pdelta_o[_i:_j]))
                keep_idx.append(_i + _best_local)
                _i = _j
            keep_idx = np.asarray(keep_idx, dtype=int)
            pr_s = pr_o[keep_idx]
            pd_s = pd_o[keep_idx]
            pm1_s = pm1_o[keep_idx]
            pm2_s = pm2_o[keep_idx]
            perr_s = perr_o[keep_idx]
            rsq = pr_s ** 2
            yd_s = rsq * pd_s
            ym1_s = rsq * pm1_s
            ym2_s = rsq * pm2_s
            yerr_s = rsq * perr_s
            all_ymin = min(float(np.nanmin(ym1_s)), float(np.nanmin(ym2_s)),
                           float(np.nanmin(yd_s - yerr_s)))
            all_ymax = max(float(np.nanmax(ym1_s)), float(np.nanmax(ym2_s)),
                           float(np.nanmax(yd_s + yerr_s)))
            yspan = max(1e-12, (all_ymax - all_ymin))
            pad = 0.10 * yspan
            # --- Robust single-pixel y-range guard (ONLY for N_pix < 2). ---
            # User report 2026-10-10: the minimum-halfspan guard introduced for single-pixel
            # insets was also firing for 2..N-pixel bins, painting unnecessary empty margins
            # around the data and making the inset look "crowded" (curves/pixels only occupy
            # the center strip). Fix: only expand to a min span when the bin truly has < 2
            # pixels. For multi-pixel bins, use the original proportional 10% padding so
            # the data fills ~90% of the inset vertical area.
            if pr_s.size < 2:
                y_mid = 0.5 * (all_ymin + all_ymax)
                min_y_halfspan = max(1e-3, 0.05 * abs(y_mid)) if np.isfinite(y_mid) else 1e-3
                if yspan < 2.0 * min_y_halfspan:
                    all_ymin_c = y_mid - min_y_halfspan
                    all_ymax_c = y_mid + min_y_halfspan
                else:
                    all_ymin_c = all_ymin - pad
                    all_ymax_c = all_ymax + pad
            else:
                all_ymin_c = all_ymin - pad
                all_ymax_c = all_ymax + pad
            ins_ylim = (all_ymin_c, all_ymax_c)

            # --- Robust x-range guard: tight padding for multi-pixel, min-span for 1-pixel. ---
            # User report 2026-10-10: the minimum-halfspan guard added for single-pixel insets
            # (needed to prevent matplotlib's AutoLocator from pulling ticks to "nice tens"
            # below plot_rmin) was also triggered on bins with 2..N pixels. The wide fixed
            # margins meant the actual pixel points + model lines only occupied a narrow
            # band in the middle, making the inset look "crowded". Fix:
            #   * pr_s.size >= 2 (multi-pixel bin) -> ONLY apply the original 8% proportional
            #     padding, so pixels/curves fill ~92% of the horizontal inset area, margins
            #     are minimal and symmetric. We already have explicit xticks below, so there
            #     is no risk of AutoLocator bleeding ticks outside the fitting range (plus
            #     we still hard-clamp xlim at the end).
            #   * pr_s.size < 2  (single-pixel bin) -> keep the min-halfspan expansion so
            #     tick labels remain readable on a zero-span window.
            pr_min = float(np.min(pr_s))
            pr_max = float(np.max(pr_s))
            xspan = max(1e-12, pr_max - pr_min)
            r_full_span = max(1e-12, r_max_plot - r_min_plot)
            if pr_s.size < 2:
                pr_mean = float(np.mean(pr_s))
                min_x_halfspan = max(0.5, 0.05 * abs(pr_mean) if np.isfinite(pr_mean) else 0.0,
                                     0.01 * r_full_span)
                if xspan < 2.0 * min_x_halfspan:
                    x0 = pr_mean - min_x_halfspan
                    x1 = pr_mean + min_x_halfspan
                else:
                    xpad = 0.08 * xspan
                    x0 = pr_min - xpad
                    x1 = pr_max + xpad
            else:
                xpad = 0.08 * xspan
                x0 = pr_min - xpad
                x1 = pr_max + xpad
            # Clamp x-window strictly inside the fitting range the user asked for,
            # so axis labels/ticks can never bleed outside [plot_rmin, plot_rmax].
            x0_clamped = max(x0, r_min_plot)
            x1_clamped = min(x1, r_max_plot)
            if x1_clamped - x0_clamped < 1e-9:
                # Extra safety: clamp created a zero window (shouldn't happen), fall back
                # to the original unclamped window so we at least show something.
                x0_clamped, x1_clamped = x0, x1
            ins_xlim = (x0_clamped, x1_clamped)
            # --- Explicit xticks to avoid matplotlib AutoLocator "nice tens" bleed ---
            try:
                n_xticks = 3
                xt_arr = np.linspace(ins_xlim[0], ins_xlim[1], n_xticks)
                # Round to reasonable precision (1e-2 Mpc/h is ~20x smaller than typical
                # bin width 4 Mpc/h) so tick labels don't carry meaningless decimals.
                decimals = 2 if (ins_xlim[1] - ins_xlim[0]) < 10.0 else 1
                xt_arr = np.round(xt_arr, decimals=decimals)
                # Remove duplicates (can happen due to rounding at tiny spans)
                xt_arr = np.unique(xt_arr)
                ax_ins_proxy = None
                _inset_xticks = xt_arr
                _inset_xticks_decimals = decimals
            except Exception:
                _inset_xticks = None
                _inset_xticks_decimals = 2

            # Now that inset location is known, draw the actual inset axes.
            wedge_slot_cursor[i_draw_wedge] += 1
            ax_ins = ax_draw_par.inset_axes([bx0, by0, inset_w, inset_h])

            # --- Main-plot red-dot diagonal label (away from inset) + thin solid arrow. ---
            # User request 2026-10-10:
            #   - Label diagonally (NE/NW/SE/SW), NOT directly left/right/up/down of the dot,
            #     so the arrow crosses the *corner* area between horizontal-vertical axes and
            #     is guaranteed NOT to overlap the vertical error bar (which is purely +-y).
            #   - Place it specifically on the side OPPOSITE to the inset box so label and
            #     inset box don't visually fight or overlap.
            #   - Arrow style: thin solid line with a filled triangular head.
            ax_native = axes[i_w_native]
            rd_w = rd_list[i_w_native]
            dy_w = data_y_list[i_w_native]
            xpt = float(rd_w[j_r])
            ypt = float(dy_w[j_r]) if np.isfinite(dy_w[j_r]) else float(np.nanmean(dy_w))
            pylo_n, pyhi_n = ylim_list[i_w_native] if ylim_list[i_w_native] is not None else ax_native.get_ylim()
            xlo_n, xhi_n = ax_native.get_xlim()
            x_range_n = xhi_n - xlo_n
            y_range_n = pyhi_n - pylo_n

            # --- Step 0b: vertical-label mode for low-r dots (xpt < 16). ---
            # User request 2026-10-11: if the zoomed dot sits at r < 16, place the #N label
            # directly ABOVE or BELOW the dot (no diagonal offset, no arrow). The side is
            # chosen by comparing the dot with the data point immediately to its right:
            #   * right data point HIGHER  -> label BELOW  (v_side = -1)
            #   * right data point LOWER   -> label ABOVE  (v_side = +1)
            # so the label sits on the side where the data is NOT going.
            _VERT_R_THRESH = 16.0
            _vert_mode = (xpt < _VERT_R_THRESH)
            _vert_v_side = +1  # default: above
            if _vert_mode:
                _rd_w = rd_list[i_w_native]
                _dy_w = data_y_list[i_w_native]
                if _rd_w is not None and _dy_w is not None and _rd_w.size > 1:
                    _right_mask = _rd_w > (xpt + 1e-9)
                    _right_idx = np.where(_right_mask)[0]
                    if _right_idx.size > 0:
                        _jr = int(_right_idx[0])
                        _y_right = float(_dy_w[_jr]) if np.isfinite(_dy_w[_jr]) else np.nan
                        if np.isfinite(_y_right) and np.isfinite(ypt):
                            _vert_v_side = -1 if (_y_right > ypt) else +1
                    else:
                        _left_mask = _rd_w < (xpt - 1e-9)
                        _left_idx = np.where(_left_mask)[0]
                        if _left_idx.size > 0:
                            _jl = int(_left_idx[-1])
                            _y_left = float(_dy_w[_jl]) if np.isfinite(_dy_w[_jl]) else np.nan
                            if np.isfinite(_y_left) and np.isfinite(ypt):
                                _vert_v_side = +1 if (_y_left > ypt) else -1

            # --- Step 0c: horizontal-label mode for local-extremum dots. ---
            # User request 2026-10-13: if the zoomed dot is a strict local extremum
            # (both the immediate left and right data points are HIGHER than it → local
            # minimum, OR both are LOWER than it → local maximum), place the #N label
            # directly to the RIGHT of the dot (no vertical offset, no arrow). This keeps
            # the label out of the "valley/peak" between the two data wings where a
            # diagonal arrow would be cramped. Horizontal mode takes precedence over
            # vertical mode when both apply.
            _horiz_mode = False
            if rd_w is not None and dy_w is not None and rd_w.size > 1:
                _left_mask_h = rd_w < (xpt - 1e-9)
                _left_idx_h = np.where(_left_mask_h)[0]
                _right_mask_h = rd_w > (xpt + 1e-9)
                _right_idx_h = np.where(_right_mask_h)[0]
                if _left_idx_h.size > 0 and _right_idx_h.size > 0:
                    _jl_h = int(_left_idx_h[-1])
                    _jr_h = int(_right_idx_h[0])
                    _y_left_h = float(dy_w[_jl_h]) if np.isfinite(dy_w[_jl_h]) else np.nan
                    _y_right_h = float(dy_w[_jr_h]) if np.isfinite(dy_w[_jr_h]) else np.nan
                    if np.isfinite(_y_left_h) and np.isfinite(_y_right_h) and np.isfinite(ypt):
                        _is_local_min = (_y_left_h > ypt) and (_y_right_h > ypt)
                        _is_local_max = (_y_left_h < ypt) and (_y_right_h < ypt)
                        if _is_local_min or _is_local_max:
                            _horiz_mode = True

            # --- Step 1: where is the dot in axes-normalized coords? ---
            xpt_norm = (xpt - xlo_n) / x_range_n if x_range_n > 1e-12 else 0.5
            ypt_norm = (ypt - pylo_n) / y_range_n if y_range_n > 1e-12 else 0.5

            # --- Step 2: pick diagonal quadrant opposite to the inset box. ---
            # inset box center is known in normalized coords (0..1). We pick h_side and v_side
            # by comparing the inset center with the dot's normalized position:
            #   If inset is to the RIGHT of the dot (inset_cx_norm > xpt_norm), label LEFT (h=-1).
            #   If inset is to the LEFT  of the dot (inset_cx_norm < xpt_norm), label RIGHT (h=+1).
            #   ... same for vertical (above/below).
            # This guarantees the label lives in the inset-free quadrant.
            _TINY = 1e-6
            if inset_cx_norm > (xpt_norm + _TINY):
                h_side = -1     # Label LEFT of the dot
            elif inset_cx_norm < (xpt_norm - _TINY):
                h_side = +1     # Label RIGHT of the dot
            else:
                # Inset & dot share the same x-center (very rare): fall back to "label on the
                # side that has more empty space in the parent axes" to avoid edge clipping.
                h_side = +1 if xpt_norm < 0.5 else -1

            if inset_cy_norm > (ypt_norm + _TINY):
                v_side = -1     # Label BELOW the dot
            elif inset_cy_norm < (ypt_norm - _TINY):
                v_side = +1     # Label ABOVE the dot
            else:
                v_side = +1 if ypt_norm < 0.5 else -1

            # Ordered list of candidate quadrants: PRIMARY (opposite inset) first, then
            # the three alternatives. If the primary quadrant causes label-bbox overlap
            # or arrow-crossing with already-placed labels in this wedge (common when two
            # zoom bins share a wedge and both insets sit on the same side), we try the
            # other quadrants BEFORE falling back to large outward pushes. Outward pushing
            # alone often fails because (a) it hits the axes boundary wall, and (b) two
            # arrows going in the same general direction will cross no matter how far out
            # you push — only a quadrant flip can separate them.
            #
            # IMPORTANT ordering rule for the 2nd (and later) labels: if there are already
            # placed labels in this wedge, DEPRIORITISE quadrants that coincide with an
            # already-placed label's quadrant. Pushing the 2nd label into the same quadrant
            # as the 1st (even if non-overlapping after outward push) makes both numbers
            # visually cluster on the same side of their red dots, which looks like they
            # are "in the same quadrant". We therefore try the *different* quadrants first
            # so the 2nd label naturally lands on the opposite side of its dot.
            _base_quads = [
                (h_side, v_side),        # primary
                (-h_side, v_side),       # flip horizontal
                (h_side, -v_side),       # flip vertical
                (-h_side, -v_side),      # flip both
            ]
            _placed_quads = set()
            for (_ptx, _pty, _pxpt, _pypt, _bhx, _bhy) in placed_per_wedge.get(i_w_native, []):
                _ph = 1 if (_ptx - _pxpt) > 0 else (-1 if (_ptx - _pxpt) < 0 else 0)
                _pv = 1 if (_pty - _pypt) > 0 else (-1 if (_pty - _pypt) < 0 else 0)
                if _ph != 0 and _pv != 0:
                    _placed_quads.add((_ph, _pv))
            # Build the final ordered list: quadrants NOT used by any placed label first
            # (in the original base order), then the used ones last.
            _candidate_quads = [q for q in _base_quads if q not in _placed_quads] + \
                               [q for q in _base_quads if q in _placed_quads]

            # --- Step 3: label offset in DATA coords (quadrant loop + per-quadrant auto-escape). ---
            # Diagonal offset = (±Δx, ±Δy). The arrow stays diagonal (never collinear with
            # the vertical error bar).
            _dx = 0.036 * x_range_n   # ~3.6% of axes x-range
            _dy = 0.085 * y_range_n   # ~8.5% of axes y-range

            # --- Step 3b: label-bbox footprint approximation (for overlap tests). ---
            box_half_x = 0.028 * x_range_n   # ~2.8% half-width in x
            box_half_y = 0.065 * y_range_n   # ~6.5% half-height in y

            err_w = data_err_list[i_w_native]
            n_rd = rd_w.size
            MAX_ROUNDS = 4
            ESCALE = 1.25
            txt_x, txt_y = xpt + h_side * _dx, ypt + v_side * _dy
            _found_valid = False
            # Fallback: the best candidate that satisfies (A) inside-axes + (C) no label
            # overlap + (D) no arrow crossing, even if it fails (B) overlapping other red
            # dots/error bars. Since the label bbox has alpha=0 (fully transparent white
            # background), a slight overlap with a non-zoom dot's error bar is acceptable
            # — the curve/error bar shows through. What we must NEVER do is overlap
            # another zoom label or cross another zoom arrow.
            _fallback_pos = None

            # --- Horizontal-label mode (local minimum): place directly to the RIGHT, no arrow. ---
            # User request 2026-10-13: when the dot is a strict local minimum, the #N label
            # goes to the right of the dot with no vertical offset and no arrow. We still
            # enforce (A) inside-axes and (C) no overlap with already-placed labels,
            # escalating the horizontal offset. If this fails, fall through to the diagonal
            # quadrant loop below. Horizontal mode takes precedence over vertical mode.
            if _horiz_mode:
                _horiz_found = False
                _horiz_fallback = None
                cur_dx_h = 1 * _dx
                for _round in range(MAX_ROUNDS + 1):
                    _tx = xpt + cur_dx_h
                    _ty = ypt
                    bx_lo = _tx - box_half_x
                    bx_hi = _tx + box_half_x
                    by_lo = _ty - box_half_y
                    by_hi = _ty + box_half_y
                    inside_axes = (bx_lo >= xlo_n and bx_hi <= xhi_n and
                                   by_lo >= pylo_n and by_hi <= pyhi_n)
                    overlap_zoom_label = False
                    for (_ptx, _pty, _pxpt, _pypt, _bhx, _bhy) in placed_per_wedge.get(i_w_native, []):
                        _p_bx_lo = _ptx - _bhx
                        _p_bx_hi = _ptx + _bhx
                        _p_by_lo = _pty - _bhy
                        _p_by_hi = _pty + _bhy
                        if not (bx_hi < _p_bx_lo or bx_lo > _p_bx_hi or
                                by_hi < _p_by_lo or by_lo > _p_by_hi):
                            overlap_zoom_label = True
                            break
                    hard_ok = inside_axes and (not overlap_zoom_label)
                    if hard_ok:
                        txt_x, txt_y = _tx, _ty
                        _horiz_found = True
                        break
                    if inside_axes and _horiz_fallback is None:
                        _horiz_fallback = (_tx, _ty)
                    txt_x, txt_y = _tx, _ty
                    cur_dx_h = cur_dx_h * ESCALE
                    cur_dx_h = min(cur_dx_h, 0.22 * x_range_n)
                if _horiz_found:
                    _found_valid = True
                elif _horiz_fallback is not None:
                    txt_x, txt_y = _horiz_fallback
                    _found_valid = True

            # --- Vertical-label mode (xpt < 16): place directly above/below, no arrow. ---
            # We still enforce (A) inside-axes and (C) no overlap with already-placed labels.
            # Try the trend-chosen direction first, then the opposite direction, escalating
            # the vertical offset within each direction. If even this fails, fall through to
            # the diagonal quadrant loop below as a last resort.
            # (Skipped when horizontal mode already found a valid position.)
            if _vert_mode and not _found_valid:
                _vert_found = False
                _vert_fallback = None
                for _v_try in (_vert_v_side, -_vert_v_side):
                    if _vert_found:
                        break
                    cur_dy = 0.5 * _dy
                    for _round in range(MAX_ROUNDS + 1):
                        _tx = xpt
                        _ty = ypt + _v_try * cur_dy
                        bx_lo = _tx - box_half_x
                        bx_hi = _tx + box_half_x
                        by_lo = _ty - box_half_y
                        by_hi = _ty + box_half_y
                        inside_axes = (bx_lo >= xlo_n and bx_hi <= xhi_n and
                                       by_lo >= pylo_n and by_hi <= pyhi_n)
                        overlap_zoom_label = False
                        for (_ptx, _pty, _pxpt, _pypt, _bhx, _bhy) in placed_per_wedge.get(i_w_native, []):
                            _p_bx_lo = _ptx - _bhx
                            _p_bx_hi = _ptx + _bhx
                            _p_by_lo = _pty - _bhy
                            _p_by_hi = _pty + _bhy
                            if not (bx_hi < _p_bx_lo or bx_lo > _p_bx_hi or
                                    by_hi < _p_by_lo or by_lo > _p_by_hi):
                                overlap_zoom_label = True
                                break
                        hard_ok = inside_axes and (not overlap_zoom_label)
                        if hard_ok:
                            txt_x, txt_y = _tx, _ty
                            _vert_found = True
                            break
                        if inside_axes and _vert_fallback is None:
                            _vert_fallback = (_tx, _ty)
                        txt_x, txt_y = _tx, _ty
                        cur_dy = cur_dy * ESCALE
                        cur_dy = min(cur_dy, 0.22 * y_range_n)
                if _vert_found:
                    _found_valid = True
                elif _vert_fallback is not None:
                    txt_x, txt_y = _vert_fallback
                    _found_valid = True

            for _q_h, _q_v in _candidate_quads:
                if _found_valid:
                    break
                # Reset offset magnitude for each new quadrant.
                cur_dx, cur_dy = _dx, _dy
                for _round in range(MAX_ROUNDS + 1):
                    _tx = xpt + _q_h * cur_dx
                    _ty = ypt + _q_v * cur_dy
                    # --- Build the candidate label bbox in data coords. ---
                    bx_lo = _tx - box_half_x
                    bx_hi = _tx + box_half_x
                    by_lo = _ty - box_half_y
                    by_hi = _ty + box_half_y

                    # --- (A) Strictly-inside-axes check (NO tolerance, user requires fully inside). ---
                    inside_axes = (bx_lo >= xlo_n and bx_hi <= xhi_n and
                                   by_lo >= pylo_n and by_hi <= pyhi_n)

                    # --- (B) Overlap check against all other red dots + error bars. ---
                    # This is a SOFT constraint: a label that slightly overlaps a non-zoom
                    # dot's error bar is acceptable (alpha=0 bbox lets it show through).
                    overlap_other = False
                    if n_rd > 0 and err_w is not None and err_w.size == n_rd:
                        bx_w = bx_hi - bx_lo
                        bh_w = by_hi - by_lo
                        err_half_x = max(1e-3 * x_range_n, 0.004 * x_range_n)
                        for jj in range(n_rd):
                            if jj == j_r:
                                continue
                            x_other = float(rd_w[jj])
                            if not np.isfinite(x_other):
                                continue
                            y_other = float(dy_w[jj]) if np.isfinite(dy_w[jj]) else float('nan')
                            if not np.isfinite(y_other):
                                continue
                            e_other = float(err_w[jj]) if np.isfinite(err_w[jj]) else 0.0
                            ex_lo = x_other - err_half_x
                            ex_hi = x_other + err_half_x
                            ey_lo = y_other - e_other
                            ey_hi = y_other + e_other
                            if not (ex_lo > (bx_hi + 1e-9) or ex_hi < (bx_lo - 1e-9)):
                                if not (ey_lo > (by_hi + 1e-9) or ey_hi < (by_lo - 1e-9)):
                                    overlap_other = True
                                    break
                            dxp = abs(x_other - 0.5 * (bx_lo + bx_hi))
                            dyp = abs(y_other - 0.5 * (by_lo + by_hi))
                            if dxp < (0.5 * bx_w + 0.008 * x_range_n) and dyp < (0.5 * bh_w + 0.015 * y_range_n):
                                overlap_other = True
                                break
                    # --- (C) Overlap check against already-placed labels in THIS wedge. ---
                    # HARD constraint: two zoom labels must never sit on top of each other.
                    overlap_zoom_label = False
                    for (_ptx, _pty, _pxpt, _pypt, _bhx, _bhy) in placed_per_wedge.get(i_w_native, []):
                        _p_bx_lo = _ptx - _bhx
                        _p_bx_hi = _ptx + _bhx
                        _p_by_lo = _pty - _bhy
                        _p_by_hi = _pty + _bhy
                        if not (bx_hi < _p_bx_lo or bx_lo > _p_bx_hi or
                                by_hi < _p_by_lo or by_lo > _p_by_hi):
                            overlap_zoom_label = True
                            break
                    # --- (D) Arrow-crossing check against already-placed arrows. ---
                    # HARD constraint: two zoom arrows must never cross.
                    arrow_crosses = False
                    if not overlap_zoom_label:
                        _cur_seg = ((_tx, _ty), (xpt, ypt))
                        for (_ptx, _pty, _pxpt, _pypt, _bhx, _bhy) in placed_per_wedge.get(i_w_native, []):
                            if _segments_intersect(_cur_seg[0], _cur_seg[1],
                                                   (_ptx, _pty), (_pxpt, _pypt)):
                                arrow_crosses = True
                                break
                    # --- Candidate accepted (all constraints satisfied) -> break both loops. ---
                    hard_ok = inside_axes and (not overlap_zoom_label) and (not arrow_crosses)
                    if hard_ok and (not overlap_other):
                        txt_x, txt_y = _tx, _ty
                        _found_valid = True
                        break
                    # --- Fallback: hard constraints OK but soft (other-dot overlap) failed. ---
                    if hard_ok and _fallback_pos is None:
                        _fallback_pos = (_tx, _ty)
                    # --- Otherwise remember latest candidate; will be clamp()'d at the end. ---
                    txt_x, txt_y = _tx, _ty
                    # Enlarge the diagonal offset and try again within this quadrant.
                    cur_dx = cur_dx * ESCALE
                    cur_dy = cur_dy * ESCALE
                    cur_dx = min(cur_dx, 0.22 * x_range_n)
                    cur_dy = min(cur_dy, 0.22 * y_range_n)

            # If no fully-valid candidate was found but a fallback exists (hard constraints
            # satisfied, only soft other-dot overlap failed), use it.
            if (not _found_valid) and (_fallback_pos is not None):
                txt_x, txt_y = _fallback_pos

            # --- FINAL hard clamp: force the label bbox to lie COMPLETELY inside the ---
            #     subplot frame (xlo_n..xhi_n, pylo_n..pyhi_n). User explicit requirement
            #     2026-10-10: "不要超出子图范围，确保标号完全落在子图边界线内". Even if
            #     the auto-escape loop pushed the label out, we slide it back in along
            #     the axis it spilled over, preserving the other coordinate.
            bx_lo_f = txt_x - box_half_x
            bx_hi_f = txt_x + box_half_x
            by_lo_f = txt_y - box_half_y
            by_hi_f = txt_y + box_half_y
            dx_shift = 0.0
            dy_shift = 0.0
            if bx_lo_f < xlo_n:
                dx_shift = xlo_n - bx_lo_f
            elif bx_hi_f > xhi_n:
                dx_shift = xhi_n - bx_hi_f
            if by_lo_f < pylo_n:
                dy_shift = pylo_n - by_lo_f
            elif by_hi_f > pyhi_n:
                dy_shift = pyhi_n - by_hi_f
            if (abs(dx_shift) + abs(dy_shift)) > 1e-15:
                txt_x = txt_x + dx_shift
                txt_y = txt_y + dy_shift

            # Register this label+arrow so later labels in the same wedge can avoid
            # overlapping it or crossing its arrow. In no-arrow modes (vertical-label
            # for low-r dots, or horizontal-label for local minima) there is no
            # arrow, so we record a zero-length "arrow" (tail==head) so the segment-
            # intersection test never treats this label as blocking other arrows.
            _no_arrow = _vert_mode or _horiz_mode
            _arr_x = float(txt_x) if _no_arrow else float(xpt)
            _arr_y = float(txt_y) if _no_arrow else float(ypt)
            placed_per_wedge[i_w_native].append(
                (float(txt_x), float(txt_y), _arr_x, _arr_y,
                 float(box_half_x), float(box_half_y))
            )

            # --- Step 4: label text + (separate) solid arrow drawn on TOP of the label. ---
            # User requests:
            #   (a) lower label bbox opacity (so the lines/curves behind are still readable)
            #   (b) larger label -> dot distance (done above via _dx/_dy)
            #   (c) arrow z-order ABOVE the label bbox/text (so the thin arrow doesn't
            #       disappear behind the white label box).
            #
            # We can't achieve (c) with a single `annotate` call: matplotlib always paints
            # the arrow FIRST, then the bbox, then the text. So split into two artists:
            #   [1] ax.text with low-opacity white bbox (zorder=12, bottom artist)
            #   [2] ax.annotate with empty text and only an arrow (zorder=13, sits on top).
            label_alpha = 0
            # Match the Top-Δχ² red-dot color exactly (L1053: color='red', ecolor='red'),
            # not a darker shade, and bump up fontsize for readability at a glance.
            TOP_RED = 'red'
            LABEL_FONTSIZE = 15
            # Extra-bold text via path_effects: draw the bold glyph a second time with a
            # thin same-colored stroke -> visually "fatter" than plain fontweight='bold'.
            _label_effects = [pe.Stroke(linewidth=1.2, foreground=TOP_RED), pe.Normal()]
            ax_native.text(txt_x, txt_y,
                           rf"\#{g_rank}",
                           transform=ax_native.transData,
                           color=TOP_RED, fontsize=LABEL_FONTSIZE, fontweight='bold',
                           ha='center', va='center', zorder=12, clip_on=False,
                           path_effects=_label_effects,
                           bbox=dict(facecolor='white', edgecolor='none',
                                     alpha=label_alpha, pad=0.9))
            # Arrow (thin solid filled-triangle) drawn AFTER (on top of) the label box.
            # shrinkA/shrinkB tuned so the tip terminates cleanly at the red-dot surface
            # and the tail peeks out just from behind the label bbox center.
            # In no-arrow modes (vertical-label for low-r dots, or horizontal-label
            # for local minima) the label sits directly adjacent to the dot with no
            # arrow, per user requests 2026-10-11 and 2026-10-13.
            if not _no_arrow:
                ax_native.annotate("",
                                   xy=(xpt, ypt),
                                   xytext=(txt_x, txt_y),
                                   xycoords='data',
                                   textcoords='data',
                                   zorder=13,
                                   arrowprops=dict(arrowstyle='-|>', color=TOP_RED, lw=1.1,
                                                   shrinkA=7, shrinkB=4,
                                                   connectionstyle="arc3,rad=0.1"))

            # --- Plot model curves (with single-pixel guard) ---
            # Root cause of "function lines invisible on the 2nd inset":
            #   Top-2 bin has exactly 1 pixel (n_pix=1), so pr_s/ym1_s/ym2_s all have length 1.
            #   ax.plot with two length-1 arrays draws a ZERO-LENGTH segment, which is
            #   invisible regardless of linewidth. Data errorbar still shows (because fmt='o'
            #   draws a point marker + vertical error bar), which makes it look like
            #   "model curves are missing". Fix: when < 2 pixels, plot a constant-y
            #   HORIZONTAL LINE segment spanning the inset x-range, so the two model
            #   levels (FID solid / REIO dotted) are clearly visible. This mirrors exactly
            #   how a "multi-pixel narrow bin" would look if the r bin were homogeneous.
            if pr_s.size >= 2:
                x_m1, y_m1 = pr_s, ym1_s
                x_m2, y_m2 = pr_s, ym2_s
            else:
                x_hline = np.asarray([ins_xlim[0], ins_xlim[1]], dtype=float)
                y1_const = float(ym1_s[0]) if ym1_s.size > 0 and np.isfinite(ym1_s[0]) else np.nan
                y2_const = float(ym2_s[0]) if ym2_s.size > 0 and np.isfinite(ym2_s[0]) else np.nan
                x_m1, y_m1 = x_hline, np.asarray([y1_const, y1_const], dtype=float)
                x_m2, y_m2 = x_hline, np.asarray([y2_const, y2_const], dtype=float)
            ax_ins.plot(x_m1, y_m1, '-', linewidth=lw)
            ax_ins.plot(x_m2, y_m2, ':', linewidth=lw)
            ax_ins.errorbar(pr_s, yd_s, yerr=yerr_s,
                            fmt='o', color='black', markersize=5.5, alpha=1.0,
                            ecolor='black', elinewidth=2.0, capsize=0, zorder=5)
            ax_ins.set_xlim(ins_xlim)
            ax_ins.set_ylim(ins_ylim)
            # Apply explicit xticks if available (avoids matplotlib AutoLocator bleeding to "nice tens"
            # below plot_rmin on single-pixel / few-pixel insets).
            if '_inset_xticks' in dir() and _inset_xticks is not None and len(_inset_xticks) > 0:
                ax_ins.set_xticks(_inset_xticks)
            # User request "小图的坐标轴标签字号也稍微增大一些": tick labels enlarged by +1 point.
            tick_fs = 9 if (inset_w <= 0.28 or inset_h <= 0.29) else 10
            ax_ins.tick_params(which='both', top=True, right=True, direction='in',
                               labelsize=tick_fs, length=3.2, width=1.0)
            ax_ins.grid(linestyle=':', alpha=0.35)

            # Upper-left big #N marker (kept as-is):
            ax_ins.text(0.05, 0.96, rf"\textbf{{\#{g_rank}}}",
                        transform=ax_ins.transAxes, fontsize=14, fontweight='bold',
                        va='top', ha='left', color='black', zorder=10,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0, pad=1.2))
            # User request: "delta_chi^2和N_pix放到小图的右下角，分两行显示，字号再大一点，并且加粗"
            # Previously top-right single-line fs=12 -> now bottom-right TWO-LINE fs=14 bold.
            # 2026-10-10 additional request: highlight the MINUS sign on negative delta_chi2
            # (the "improved by reionization" direction):
            #   * wrap the minus sign with single spaces:  " Δχ² = ␣ <BIG_MINUS> ␣ 2.28 "
            #   * enlarge the minus glyph alone by one size step using LaTeX \mathlarger{}
            #     (if mathtext supports it; else fall back to \textstyle with explicit size
            #      control). We keep the numeric magnitude unchanged for alignment.
            delta_val = float(total_bin_delta)
            mag_str = "{:.2f}".format(abs(delta_val))
            # Build the delta-chi^2 line with / without the "highlighted minus" decoration.
            if delta_val < 0:
                # Negative -> reionization IMPROVES the fit (the scientifically interesting
                # direction). Surround the minus glyph with single spaces as requested, and
                # enlarge the minus alone using mathtext's built-in size step. `\LARGE` is
                # supported by matplotlib's mathtext out of the box (~+3pt vs surrounding
                # 14pt), no graphicx/scalebox required.
                dchi2_line = (r"$\Delta\chi^2$ = "
                              r" ${\LARGE -}$ "
                              + mag_str)
            else:
                # Zero / positive: keep a single regular "+" so the layout stays clean.
                dchi2_line = r"$\Delta\chi^2$ = +" + mag_str
            n_pix_line = r"$N_{{\rm pix}}$ = {}".format(int(pix_idx.size))
            # Emphasize the minus-stroke further with a path effect: stroke the whole text
            # very thinly, which disproportionately thickens the single-stroke minus glyph
            # (already the visually dominant feature of the string).
            _inset_val_effects = [pe.Stroke(linewidth=0.55, foreground='black'), pe.Normal()]
            # Split the former two-line single text into two independent text artists
            # so the second line can be shifted horizontally until its "=" sign sits
            # exactly below the first line's "=" sign (visual column alignment).
            fig.canvas.draw()
            _renderer = fig.canvas.get_renderer()
            _ins_bb = ax_ins.get_window_extent(renderer=_renderer)

            def _measure_after_eq(_s):
                _t = ax_ins.text(0, 0, _s, fontsize=14, fontweight='bold',
                                 path_effects=_inset_val_effects,
                                 transform=ax_ins.transAxes)
                _bb = _t.get_window_extent(renderer=_renderer)
                _t.remove()
                return _bb.width / _ins_bb.width, _bb.height / _ins_bb.height

            # Width (in inset axes-fraction units) of everything to the RIGHT of "=".
            _w1, _h1 = _measure_after_eq(dchi2_line.split('=', 1)[1])
            _w2, _h2 = _measure_after_eq(n_pix_line.split('=', 1)[1])
            # Line 1 right edge stays at x=0.95 (ha='right') -> its "=" is at 0.95 - _w1.
            # For line 2's "=" to coincide, its right edge must be (0.95 - _w1) + _w2.
            _line1_x = 0.95
            _line2_x = max(0.02, min(0.98, 0.95 - _w1 + _w2))
            # --- Auto-place the two-line block vertically to minimise overlap with ---
            #     inset content (data points, error bars, model curves).
            # We keep the relative vertical gap between the two lines and the right-edge
            # anchoring unchanged; only the block's overall vertical position is searched.
            _full_w1, _ = _measure_after_eq(dchi2_line)
            _full_w2, _ = _measure_after_eq(n_pix_line)
            _blk_right = 0.95
            _blk_left = max(0.02, _blk_right - max(_full_w1, _full_w2))
            _blk_h = _h1 + 0.015 + _h2
            _ins_xlo, _ins_xhi = ins_xlim
            _ins_ylo, _ins_yhi = ins_ylim
            _xrange_ins = max(1e-12, _ins_xhi - _ins_xlo)
            _yrange_ins = max(1e-12, _ins_yhi - _ins_ylo)

            def _axes_to_data(ax_fx, ax_fy):
                return (_ins_xlo + ax_fx * _xrange_ins,
                        _ins_ylo + ax_fy * _yrange_ins)

            # Collect content points (data coords): data dots, error-bar endpoints,
            # and densely-sampled model curve points.
            _content_x, _content_y = [], []
            if pr_s.size > 0:
                _content_x.extend(pr_s.tolist())
                _content_y.extend(yd_s.tolist())
                if yerr_s is not None and yerr_s.size == pr_s.size:
                    _content_x.extend(pr_s.tolist())
                    _content_y.extend((yd_s - yerr_s).tolist())
                    _content_x.extend(pr_s.tolist())
                    _content_y.extend((yd_s + yerr_s).tolist())
            for _xc, _yc in ((x_m1, y_m1), (x_m2, y_m2)):
                if _xc is not None and _yc is not None and len(_xc) >= 2:
                    _xfine = np.linspace(float(np.min(_xc)), float(np.max(_xc)), 80)
                    _yfine = np.interp(_xfine, np.asarray(_xc, dtype=float),
                                       np.asarray(_yc, dtype=float))
                    _content_x.extend(_xfine.tolist())
                    _content_y.extend(_yfine.tolist())
            _content_x = np.asarray(_content_x, dtype=float)
            _content_y = np.asarray(_content_y, dtype=float)

            _best_line2_y = 0.06
            _best_overlap = None
            _n_steps = 30
            _y_min = 0.0
            _y_max = max(_y_min, 1.0 - _blk_h)
            for _s in range(_n_steps + 1):
                _cand = _y_min + (_y_max - _y_min) * _s / _n_steps
                _blk_bot = _cand
                _blk_top = _cand + _blk_h
                _dx_lo, _dy_lo = _axes_to_data(_blk_left, _blk_bot)
                _dx_hi, _dy_hi = _axes_to_data(_blk_right, _blk_top)
                _mask = ((_content_x >= _dx_lo) & (_content_x <= _dx_hi) &
                         (_content_y >= _dy_lo) & (_content_y <= _dy_hi))
                _ov = int(np.sum(_mask))
                if _best_overlap is None or _ov < _best_overlap:
                    _best_overlap = _ov
                    _best_line2_y = _cand
                    if _ov == 0:
                        break
            _line2_y = _best_line2_y
            _line1_y = _line2_y + _h2 + 0.015
            ax_ins.text(_line1_x, _line1_y, dchi2_line,
                        transform=ax_ins.transAxes, fontsize=14, fontweight='bold',
                        va='bottom', ha='right', color='black', zorder=10,
                        path_effects=_inset_val_effects,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0, pad=1.2))
            ax_ins.text(_line2_x, _line2_y, n_pix_line,
                        transform=ax_ins.transAxes, fontsize=14, fontweight='bold',
                        va='bottom', ha='right', color='black', zorder=10,
                        path_effects=_inset_val_effects,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0, pad=1.2))
            # Inset axis labels (r_pix / r²ξ) explicitly removed per previous user request.


    fid_tag = os.path.basename(os.path.normpath(args.chain_fid))
    reio_tag = os.path.basename(os.path.normpath(args.chain_reio))
    plot_path = os.path.join(
        out_dir,
        f"point_chi2_{args.corr}_{fid_tag}_vs_{reio_tag}_top{args.top_k}.png"
    )
    saved = False
    try:
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"  Saved figure -> {plot_path}")
        saved = True
    except Exception as e:
        print(f"  savefig with current rcParams failed: {e}")
    if not saved:
        # Cluster LaTeX may lack unicode input setup or some packages.
        # Retry with matplotlib's native mathtext renderer (no external latex).
        print("  Retrying savefig with usetex=False (using matplotlib mathtext)...")
        prev_usetex = plt.rcParams.get('text.usetex', False)
        prev_text_latex = plt.rcParams.get('text.latex.preamble', None)
        try:
            plt.rcParams['text.usetex'] = False
            try:
                del plt.rcParams['text.latex.preamble']
            except Exception:
                pass
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            print(f"  Saved figure (mathtext fallback) -> {plot_path}")
            saved = True
        except Exception as e2:
            print(f"  Mathtext fallback also failed: {e2}")
        finally:
            plt.rcParams['text.usetex'] = prev_usetex
            if prev_text_latex is not None:
                plt.rcParams['text.latex.preamble'] = prev_text_latex
    if not saved:
        # Last fallback: Agg backend with pure ASCII labels
        print("  Last-resort fallback: redraw legend with ASCII labels and retry once.")
        try:
            # Replace any label on the first axis that still has raw unicode with ascii.
            ax0 = axes[0]
            handles, old_labels = ax0.get_legend_handles_labels()
            new_labels = []
            for lab in old_labels:
                if 'Top-' in lab and ('Delta' in lab or 'χ' in lab or 'Δ' in lab):
                    new_labels.append('Top-dchi2 bins')
                else:
                    new_labels.append(lab)
            # Only re-do if actually changed.
            if any(a != b for a, b in zip(old_labels, new_labels)):
                ax0.legend(handles, new_labels, frameon=False)
            plt.rcParams['text.usetex'] = False
            try:
                del plt.rcParams['text.latex.preamble']
            except Exception:
                pass
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            print(f"  Saved figure (ASCII fallback) -> {plot_path}")
            saved = True
        except Exception as e3:
            print(f"  Figure could not be saved in any fallback mode: {e3}")
    plt.close(fig)


if __name__ == '__main__':
    main()

# Example usage:
# python analysis/point_chi2.py \
#   --chain-fid /Users/zhengyifan/vega_result/40_180/auto_fid_40_180/Poly_bbt_40_180_auto_fid \
#   --chain-reio /Users/zhengyifan/vega_result/40_180/auto_reio_40_180_3free/Poly_bbt_40_180_auto \
#   --config-fid analysis/main_fid.ini --config-reio analysis/main_reio.ini \
#   --corr lyaxlya --plot-rmin 40 --plot-rmax 180 --top-k 10
