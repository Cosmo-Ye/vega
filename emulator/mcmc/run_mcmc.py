from __future__ import annotations

import argparse
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from mcmc.script_mcmc_obs import load_pmpsi_from_chain
from pmxh_shape_emulator import PMXHShapeEmulator

from scipy import interpolate
from scipy.integrate import quad


def _default_k_bin() -> list[float]:
    return [0.0621205, 0.07542991, 0.09031192, 0.1082159, 0.1290225, 0.1547179, 0.1857322, 0.2232069, 0.2680392, 0.3214029, 0.3854592]


def _parse_float_list(xs: Sequence[str]) -> list[float]:
    return [float(x) for x in xs]


def _interp_logk(*, k_src: np.ndarray, y_src: np.ndarray, k_tgt: np.ndarray) -> np.ndarray:
    k_src = np.asarray(k_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    k_tgt = np.asarray(k_tgt, dtype=float)

    if k_src.ndim != 1 or y_src.ndim != 1 or k_src.size != y_src.size:
        raise ValueError("Expected 1D k_src and y_src arrays with the same length")
    if k_tgt.ndim != 1:
        raise ValueError("Expected 1D k_tgt array")
    if np.any(k_src <= 0) or np.any(k_tgt <= 0):
        raise ValueError("k values must be positive for log interpolation")

    x = np.log10(k_src)
    xq = np.log10(k_tgt)
    order = np.argsort(x)
    return np.interp(xq, x[order], y_src[order])


def _trapz_weights(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z must be a 1D array with at least 2 points")
    dz = np.diff(z)
    if np.any(dz <= 0):
        raise ValueError("z must be strictly increasing for trapezoid integration")
    w = np.empty_like(z, dtype=float)
    w[0] = 0.5 * dz[0]
    w[-1] = 0.5 * dz[-1]
    if z.size > 2:
        w[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    return w


_PSI_Z_NODES = np.asarray([2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)
_PSI_X = np.asarray([6, 7, 8, 9, 10, 11, 12], dtype=float)
_PSI_Y_BY_Z = np.asarray(
    [
        [0.06064, 0.02112, 0.0, -0.008787, -0.01300, -0.01425, -0.01552],
        [0.06910, 0.02399, 0.0, -0.006496, -0.004075, -0.000525, 0.001997],
        [0.09772, 0.03612, 0.0, -0.01083, -0.008478, -0.003189, 0.002835],
        [0.14616, 0.05769, 0.0, -0.02548, -0.02950, -0.02626, -0.02128],
        [0.22522, 0.09275, 0.0, -0.04682, -0.06344, -0.06610, -0.06400],
    ],
    dtype=float,
)
_PSI_TCKS = tuple(interpolate.splrep(_PSI_X, _PSI_Y_BY_Z[i], k=1, s=0) for i in range(_PSI_Z_NODES.size))

_MANUAL_PRIOR_ZETA = np.asarray([15.455066, 36.072697, 50.0, 100.0, 150.0, 250.0], dtype=float)
_MANUAL_PRIOR_LOGT = np.asarray([4.498625, 5.035638, 5.25529, 5.59915, 5.76, 5.925], dtype=float)


def pmpsi_from_pmxh(
    *,
    pmxh_mean_grid: np.ndarray,
    pmxh_std_grid: np.ndarray,
    z_grid: np.ndarray,
    k_grid: np.ndarray,
    k_obs: np.ndarray,
    z_select: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_grid = np.asarray(z_grid, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    k_obs = np.asarray(k_obs, dtype=float)

    if z_grid.ndim != 1 or k_grid.ndim != 1 or k_obs.ndim != 1:
        raise ValueError("Expected 1D z_grid, k_grid and k_obs arrays")
    if pmxh_mean_grid.shape != pmxh_std_grid.shape:
        raise ValueError("Expected pmxh_mean_grid and pmxh_std_grid to have the same shape")
    if pmxh_mean_grid.ndim != 2 or pmxh_mean_grid.shape != (z_grid.size, k_grid.size):
        raise ValueError("Expected pmxh_*_grid shape to be (nz, nk)")

    z_eval = z_grid
    psi_partial_by_z = np.stack([interpolate.splev(z_eval, tck, der=1, ext=0) for tck in _PSI_TCKS], axis=0)

    w = _trapz_weights(z_grid)[:, None]
    D_grid = np.asarray([growth_factor_calculation(float(z)) for z in z_grid], dtype=float)
    D_nodes = np.asarray([growth_factor_calculation(float(z)) for z in _PSI_Z_NODES], dtype=float)
    ratio_by_node = (D_nodes[:, None] / D_grid[None, :]).astype(float)

    kernel = (-psi_partial_by_z * ratio_by_node).astype(float)

    pmpsi_by_node_mean = (w * (kernel[:, :, None] * pmxh_mean_grid[None, :, :])).sum(axis=1)
    pmpsi_by_node_var = ((w * w) * ((kernel * kernel)[:, :, None] * (pmxh_std_grid[None, :, :] ** 2))).sum(axis=1)
    pmpsi_by_node_std = np.sqrt(np.maximum(pmpsi_by_node_var, 0.0))

    z_target = float(z_select)
    f_mean = interpolate.interp1d(_PSI_Z_NODES, pmpsi_by_node_mean, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
    f_std = interpolate.interp1d(_PSI_Z_NODES, pmpsi_by_node_std, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate")
    pmpsi_mean_k = np.asarray(f_mean(z_target), dtype=float)
    pmpsi_std_k = np.asarray(f_std(z_target), dtype=float)

    pmpsi_mean = _interp_logk(k_src=k_grid, y_src=pmpsi_mean_k, k_tgt=k_obs)
    pmpsi_std = np.abs(_interp_logk(k_src=k_grid, y_src=pmpsi_std_k, k_tgt=k_obs))
    return pmpsi_mean, pmpsi_std

@lru_cache(maxsize=None)
def growth_factor_calculation(z, Om=0.315094454725487, OL=0.684825913083526): #DESI's cosmology
# def growth_factor_calculation(z, Om=0.3088, OL=0.6912): #21cmFAST's cosmology
    '''
    Implements eq. 7.77 from S. Dodelson's Modern Cosmology book
    '''

    def hubble(z, Om, OL):
        return np.sqrt(Om*(1+z)**3 + OL + (1-Om-OL)*(1+z)**2)

    def dD1(a, Om, OL):
        z = 1/a-1
        return 1./(a*hubble(z=z, Om=Om, OL=OL))**3

    pars = (Om, OL)

    a = 1/(1+z)
    D1 = 5/2.*Om*hubble(z, *pars)*quad(dD1, 0, a, args=pars)[0]

    return D1


def _build_full_params(
    *,
    base_params: np.ndarray,
    zeta_uv: float,
    tvir: float,
    zeta_idx: int,
    tvir_idx: int,
    tvir_in_log10: bool,
) -> np.ndarray:
    p = np.asarray(base_params, dtype=float).copy()
    if p.ndim != 1:
        raise ValueError("base_params must be 1D")
    if not (0 <= int(zeta_idx) < p.size and 0 <= int(tvir_idx) < p.size):
        raise ValueError("zeta_idx/tvir_idx out of range for base_params")
    p[int(zeta_idx)] = float(zeta_uv)
    p[int(tvir_idx)] = float(math.log10(float(tvir))) if bool(tvir_in_log10) else float(tvir)
    return p


def _manual_prior_boundary_logt(zeta_uv: float) -> float:
    zeta_uv = float(zeta_uv)
    z0 = _MANUAL_PRIOR_ZETA
    y0 = _MANUAL_PRIOR_LOGT

    # Extend the first segment to the left down to zeta=10.
    if zeta_uv <= float(z0[1]):
        slope = (float(y0[1]) - float(y0[0])) / (float(z0[1]) - float(z0[0]))
        return float(y0[0] + slope * (zeta_uv - float(z0[0])))

    for i in range(1, z0.size - 1):
        lo = float(z0[i])
        hi = float(z0[i + 1])
        if lo < zeta_uv <= hi:
            slope = (float(y0[i + 1]) - float(y0[i])) / (hi - lo)
            return float(y0[i] + slope * (zeta_uv - lo))

    slope = (float(y0[-1]) - float(y0[-2])) / (float(z0[-1]) - float(z0[-2]))
    return float(y0[-1] + slope * (zeta_uv - float(z0[-1])))


def _custom_prior_status(*, zeta_uv: float, logtvir: float) -> str:
    if float(logtvir) > _manual_prior_boundary_logt(float(zeta_uv)):
        return "reject"
    return "accept"


def _log_prior(*, zeta_uv: float, tvir: float, zeta_min: float, zeta_max: float, tvir_min: float, tvir_max: float) -> float:
    if not (float(zeta_min) <= float(zeta_uv) <= float(zeta_max)):
        return float("-inf")
    if not (float(tvir_min) <= float(tvir) <= float(tvir_max)):
        return float("-inf")
    status = _custom_prior_status(zeta_uv=float(zeta_uv), logtvir=float(math.log10(float(tvir))))
    if status == "reject":
        return float("-inf")
    return 0.0


def _log_likelihood_gaussian_diag(*, y_obs: np.ndarray, y_model: np.ndarray, sigma: np.ndarray) -> float:
    y_obs = np.asarray(y_obs, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if y_obs.shape != y_model.shape or y_obs.shape != sigma.shape:
        raise ValueError("Shape mismatch for y_obs/y_model/sigma")
    if np.any(~np.isfinite(y_obs)) or np.any(~np.isfinite(y_model)) or np.any(~np.isfinite(sigma)):
        return float("-inf")
    if np.any(sigma <= 0):
        return float("-inf")
    r = (y_obs - y_model) / sigma
    return float(-0.5 * np.sum(r * r + 2.0 * np.log(sigma)))


def _log_likelihood_asymmetric_gaussian(
    *,
    y_obs: np.ndarray,
    y_model: np.ndarray,
    sigma_minus: np.ndarray,
    sigma_plus: np.ndarray,
    emu_std: np.ndarray | None = None,
) -> float:
    y_obs = np.asarray(y_obs, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    sigma_minus = np.asarray(sigma_minus, dtype=float)
    sigma_plus = np.asarray(sigma_plus, dtype=float)
    if y_obs.shape != y_model.shape or y_obs.shape != sigma_minus.shape or sigma_minus.shape != sigma_plus.shape:
        raise ValueError("Shape mismatch for y_obs/y_model/sigma_minus/sigma_plus")
    if emu_std is None:
        emu2 = np.zeros_like(y_obs, dtype=float)
    else:
        emu_std_arr = np.asarray(emu_std, dtype=float)
        if emu_std_arr.shape != y_obs.shape:
            raise ValueError("Shape mismatch between emu_std and y_obs")
        if np.any(~np.isfinite(emu_std_arr)):
            return float("-inf")
        emu2 = emu_std_arr * emu_std_arr
    if np.any(~np.isfinite(y_obs)) or np.any(~np.isfinite(y_model)):
        return float("-inf")
    if np.any(~np.isfinite(sigma_minus)) or np.any(~np.isfinite(sigma_plus)):
        return float("-inf")
    if np.any(sigma_minus <= 0) or np.any(sigma_plus <= 0):
        return float("-inf")
    choose_plus = (y_model >= y_obs)
    sigma_obs_choice = np.where(choose_plus, sigma_plus, sigma_minus)
    sigma_tot = np.sqrt(sigma_obs_choice * sigma_obs_choice + emu2)
    if np.any(~np.isfinite(sigma_tot)) or np.any(sigma_tot <= 0):
        return float("-inf")
    r = (y_obs - y_model) / sigma_tot
    return float(-0.5 * np.sum(r * r + 2.0 * np.log(sigma_tot)))


def _npz_scalar(npz, key: str):
    v = npz[key]
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    if isinstance(v, np.ndarray) and v.size == 1:
        return np.asarray(v).reshape(()).item()
    return v


def _save_npz_atomic(*, out_path: Path, compress: bool, **arrays) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.parent / (out_path.name + ".tmp")
    with open(str(tmp_path), "wb") as f:
        if bool(compress):
            np.savez_compressed(f, **arrays)
        else:
            np.savez(f, **arrays)
    os.replace(str(tmp_path), str(out_path))


def run_metropolis_hastings(
    *,
    rng: np.random.Generator,
    log_posterior,
    x0: np.ndarray,
    step_scales: np.ndarray,
    nsteps: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x0, dtype=float).copy()
    step_scales = np.asarray(step_scales, dtype=float)
    if x.ndim != 1 or step_scales.ndim != 1 or x.size != step_scales.size:
        raise ValueError("x0 and step_scales must be 1D arrays of the same length")
    if int(nsteps) <= 0:
        raise ValueError("nsteps must be > 0")

    chain = np.empty((int(nsteps), x.size), dtype=float)
    logp = np.empty(int(nsteps), dtype=float)

    lp = float(log_posterior(x))
    n_accept = 0

    for i in range(int(nsteps)):
        prop = x + rng.normal(loc=0.0, scale=step_scales, size=x.size)
        lp_prop = float(log_posterior(prop))

        accept = False
        if math.isfinite(lp_prop):
            if lp_prop >= lp:
                accept = True
            else:
                if rng.random() < math.exp(lp_prop - lp):
                    accept = True

        if accept:
            x = prop
            lp = lp_prop
            n_accept += 1

        chain[i, :] = x
        logp[i] = lp

    acc_rate = float(n_accept) / float(nsteps)
    return chain, logp, acc_rate


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_mcmc")

    p.add_argument("--restart", default=None)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--print-every", type=int, default=5000)

    p.add_argument("--pmpsi-chain-root", default="mcmc/Polychord_result/full_reio_extendprior/Polychord_fullcombined_extendprior")
    p.add_argument("--ignore-rows", type=float, default=0.2)
    p.add_argument("--k-bin", nargs="*", type=float, default=None)

    p.add_argument("--emu-model", default="pmxh_shape_emulator.joblib")
    p.add_argument("--base-params", nargs="*", type=float, default=None)
    p.add_argument("--zeta-idx", type=int, default=0)
    p.add_argument("--tvir-idx", type=int, default=1)
    p.add_argument("--tvir-in-log10", action="store_true", default=False)
    p.add_argument("--tvir-sample-log10", action="store_true", default=False)

    p.add_argument("--zeta-min", type=float, default=0.0)
    p.add_argument("--zeta-max", type=float, default=200.0)
    p.add_argument("--tvir-min", type=float, default=1e4)
    p.add_argument("--tvir-max", type=float, default=1e6)
    p.add_argument("--logtvir-min", type=float, default=4.0)
    p.add_argument("--logtvir-max", type=float, default=5.0)

    p.add_argument("--z-select", type=float, default=2.33)
    p.add_argument("--no-emu-std", action="store_true", default=False)

    p.add_argument("--xhi-model", default="xhi_shape_emulator.joblib")
    p.add_argument("--xhi-z", type=float, default=5.37)
    p.add_argument("--xhi-max", type=float, default=0.05)

    p.add_argument("--nsteps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zeta-init", type=float, default=30.0)
    p.add_argument("--tvir-init", type=float, default=3e4)
    p.add_argument("--logtvir-init", type=float, default=math.log10(3e4))
    p.add_argument("--zeta-step", type=float, default=2.0)
    p.add_argument("--tvir-step", type=float, default=3000.0)
    p.add_argument("--logtvir-step", type=float, default=0.05)

    p.add_argument("--burn", type=int, default=2000)
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--out", default="mcmc/mh_zeta_tvir_chain.npz")

    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    restart_chain = None
    restart_logp = None
    restart_path = None
    restart_last_x = None
    restart_last_logp = None
    restart_n_steps = 0
    restart_n_accept = 0
    if args.restart is not None and str(args.restart).strip() != "":
        restart_path = Path(str(args.restart))
        if not restart_path.exists():
            raise FileNotFoundError(str(restart_path))
        restart_npz = np.load(str(restart_path), allow_pickle=False)
        if "chain" not in restart_npz or "logp" not in restart_npz:
            raise ValueError("restart file must contain 'chain' and 'logp' arrays")
        restart_chain = np.asarray(restart_npz["chain"], dtype=float)
        restart_logp = np.asarray(restart_npz["logp"], dtype=float)
        if restart_chain.ndim != 2 or restart_chain.shape[1] != 2:
            raise ValueError("restart file 'chain' must have shape (n, 2) for (zeta_uv, tvir)")
        if restart_logp.ndim != 1 or restart_logp.shape[0] != restart_chain.shape[0]:
            raise ValueError("restart file 'logp' must be 1D and match chain length")
        required_keys = [
            "pmpsi_chain_root",
            "ignore_rows",
            "emu_model",
            "base_params",
            "zeta_idx",
            "tvir_idx",
            "tvir_in_log10",
            "tvir_sample_log10",
            "zeta_min",
            "zeta_max",
            "tvir_min",
            "tvir_max",
            "logtvir_min",
            "logtvir_max",
            "z_select",
            "no_emu_std",
            "xhi_model",
            "xhi_z",
            "xhi_max",
            "seed",
            "zeta_step",
            "tvir_step",
            "logtvir_step",
            "burn",
            "thin",
            "k_obs",
            "pmpsi_obs",
            "pmpsi_sigma",
            "pmpsi_sigma_minus",
            "pmpsi_sigma_plus",
            "last_x",
            "last_logp",
            "n_total_steps",
            "n_total_accept",
        ]
        missing = [k for k in required_keys if k not in restart_npz]
        if len(missing) > 0:
            raise ValueError(f"restart file missing required keys for strict reuse: {missing}")

        args.pmpsi_chain_root = str(_npz_scalar(restart_npz, "pmpsi_chain_root"))
        args.ignore_rows = float(_npz_scalar(restart_npz, "ignore_rows"))
        args.emu_model = str(_npz_scalar(restart_npz, "emu_model"))
        args.base_params = np.asarray(restart_npz["base_params"], dtype=float).tolist()
        args.zeta_idx = int(_npz_scalar(restart_npz, "zeta_idx"))
        args.tvir_idx = int(_npz_scalar(restart_npz, "tvir_idx"))
        args.tvir_in_log10 = bool(_npz_scalar(restart_npz, "tvir_in_log10"))
        args.tvir_sample_log10 = bool(_npz_scalar(restart_npz, "tvir_sample_log10"))
        args.zeta_min = float(_npz_scalar(restart_npz, "zeta_min"))
        args.zeta_max = float(_npz_scalar(restart_npz, "zeta_max"))
        args.tvir_min = float(_npz_scalar(restart_npz, "tvir_min"))
        args.tvir_max = float(_npz_scalar(restart_npz, "tvir_max"))
        args.logtvir_min = float(_npz_scalar(restart_npz, "logtvir_min"))
        args.logtvir_max = float(_npz_scalar(restart_npz, "logtvir_max"))
        args.z_select = float(_npz_scalar(restart_npz, "z_select"))
        args.no_emu_std = bool(_npz_scalar(restart_npz, "no_emu_std"))
        args.xhi_model = str(_npz_scalar(restart_npz, "xhi_model"))
        args.xhi_z = float(_npz_scalar(restart_npz, "xhi_z"))
        args.xhi_max = float(_npz_scalar(restart_npz, "xhi_max"))
        args.seed = int(_npz_scalar(restart_npz, "seed"))
        args.zeta_step = float(_npz_scalar(restart_npz, "zeta_step"))
        args.tvir_step = float(_npz_scalar(restart_npz, "tvir_step"))
        args.logtvir_step = float(_npz_scalar(restart_npz, "logtvir_step"))
        args.burn = int(_npz_scalar(restart_npz, "burn"))
        args.thin = int(_npz_scalar(restart_npz, "thin"))
        restart_last_x = np.asarray(restart_npz["last_x"], dtype=float).reshape(-1)
        restart_last_logp = float(_npz_scalar(restart_npz, "last_logp"))
        restart_n_steps = int(_npz_scalar(restart_npz, "n_total_steps"))
        restart_n_accept = int(_npz_scalar(restart_npz, "n_total_accept"))
        if restart_last_x.size != 2:
            raise ValueError("restart file last_x must have shape (2,)")

    rng = np.random.default_rng(int(args.seed))

    emu = PMXHShapeEmulator.load(str(args.emu_model))
    ndim = int(emu.train_params.shape[1])
    k_grid_full = np.asarray(emu.k_grid, dtype=float)
    k_mask = (k_grid_full > 0.06) & (k_grid_full < 0.2)
    if not np.any(k_mask):
        raise ValueError("No k modes left after applying 0.06<k<0.2 cut to emu.k_grid")
    k_grid = k_grid_full[k_mask]

    pmpsi = load_pmpsi_from_chain(chain_root=str(args.pmpsi_chain_root), k_bin=k_grid.tolist(), ignore_rows=float(args.ignore_rows))
    k_obs = pmpsi["k"]
    pmpsi_obs = pmpsi["median"]
    pmpsi_sigma = pmpsi["sigma"]
    pmpsi_sigma_minus = pmpsi.get("sigma_minus")
    pmpsi_sigma_plus = pmpsi.get("sigma_plus")
    if pmpsi_sigma_minus is None or pmpsi_sigma_plus is None:
        pmpsi_sigma_minus = np.asarray(pmpsi_sigma, dtype=float).copy()
        pmpsi_sigma_plus = np.asarray(pmpsi_sigma, dtype=float).copy()
    else:
        pmpsi_sigma_minus = np.asarray(pmpsi_sigma_minus, dtype=float)
        pmpsi_sigma_plus = np.asarray(pmpsi_sigma_plus, dtype=float)
    if pmpsi_sigma_minus.shape != pmpsi_obs.shape or pmpsi_sigma_plus.shape != pmpsi_obs.shape:
        raise ValueError("pmpsi sigma_minus/sigma_plus shape mismatch pmpsi_obs")
    bad_minus = ~np.isfinite(pmpsi_sigma_minus) | (pmpsi_sigma_minus <= 0)
    bad_plus = ~np.isfinite(pmpsi_sigma_plus) | (pmpsi_sigma_plus <= 0)
    if np.any(bad_minus | bad_plus):
        fallback = np.asarray(pmpsi_sigma, dtype=float)
        pmpsi_sigma_minus = np.where(bad_minus | ~np.isfinite(fallback) | (fallback <= 0), pmpsi_sigma_minus, fallback)
        pmpsi_sigma_minus = np.where(np.isfinite(pmpsi_sigma_minus) & (pmpsi_sigma_minus > 0), pmpsi_sigma_minus, fallback)
        pmpsi_sigma_plus = np.where(bad_plus | ~np.isfinite(fallback) | (fallback <= 0), pmpsi_sigma_plus, fallback)
        pmpsi_sigma_plus = np.where(np.isfinite(pmpsi_sigma_plus) & (pmpsi_sigma_plus > 0), pmpsi_sigma_plus, fallback)

    if restart_path is not None:
        if not np.allclose(np.asarray(restart_npz["k_obs"], dtype=float), np.asarray(k_obs, dtype=float)):
            raise ValueError("restart file k_obs does not match current k_obs")
        if not np.allclose(np.asarray(restart_npz["pmpsi_obs"], dtype=float), np.asarray(pmpsi_obs, dtype=float)):
            raise ValueError("restart file pmpsi_obs does not match current pmpsi_obs")
        if not np.allclose(np.asarray(restart_npz["pmpsi_sigma"], dtype=float), np.asarray(pmpsi_sigma, dtype=float)):
            raise ValueError("restart file pmpsi_sigma does not match current pmpsi_sigma")
        if "pmpsi_sigma_minus" in restart_npz and "pmpsi_sigma_plus" in restart_npz:
            if not np.allclose(np.asarray(restart_npz["pmpsi_sigma_minus"], dtype=float), np.asarray(pmpsi_sigma_minus, dtype=float), equal_nan=False):
                raise ValueError("restart file pmpsi_sigma_minus does not match current pmpsi_sigma_minus")
            if not np.allclose(np.asarray(restart_npz["pmpsi_sigma_plus"], dtype=float), np.asarray(pmpsi_sigma_plus, dtype=float), equal_nan=False):
                raise ValueError("restart file pmpsi_sigma_plus does not match current pmpsi_sigma_plus")

    if args.base_params is None or len(args.base_params) == 0:
        base_params = np.median(np.asarray(emu.train_params, dtype=float), axis=0)
    else:
        base_params = np.asarray(args.base_params, dtype=float)
        if base_params.size != ndim:
            raise ValueError(f"--base-params must have length {ndim} (emu param dimension)")

    if bool(args.tvir_sample_log10):
        step_scales = np.asarray([float(args.zeta_step), float(args.logtvir_step)], dtype=float)
    else:
        step_scales = np.asarray([float(args.zeta_step), float(args.tvir_step)], dtype=float)

    burn = int(max(0, args.burn))
    thin = int(max(1, args.thin))
    checkpoint_every = int(max(0, args.checkpoint_every))
    print_every = int(max(0, args.print_every))

    def log_post(x: np.ndarray) -> float:
        zeta_uv = float(x[0])
        logtvir = float("nan")
        if bool(args.tvir_sample_log10):
            logtvir = float(x[1])
            if not (float(args.logtvir_min) <= logtvir <= float(args.logtvir_max)):
                return float("-inf")
            tvir = float(10.0 ** logtvir)
        else:
            tvir = float(x[1])
            if not (float(args.tvir_min) <= tvir <= float(args.tvir_max)):
                return float("-inf")
            logtvir = float(math.log10(tvir))

        lp = _log_prior(
            zeta_uv=zeta_uv,
            tvir=tvir,
            zeta_min=float(args.zeta_min),
            zeta_max=float(args.zeta_max),
            tvir_min=float(args.tvir_min),
            tvir_max=float(args.tvir_max),
        )
        if not math.isfinite(lp):
            return float("-inf")

        full_params = _build_full_params(
            base_params=base_params,
            zeta_uv=zeta_uv,
            tvir=tvir,
            zeta_idx=int(args.zeta_idx),
            tvir_idx=int(args.tvir_idx),
            tvir_in_log10=bool(args.tvir_in_log10),
        )

        pmxh_mean_grid, pmxh_std_grid = emu.predict_grid(params=full_params.tolist())
        pmxh_mean_grid = np.asarray(pmxh_mean_grid, dtype=float)[:, k_mask]
        pmxh_std_grid = np.asarray(pmxh_std_grid, dtype=float)[:, k_mask]
        pmpsi_model, pmpsi_model_std = pmpsi_from_pmxh(
            pmxh_mean_grid=pmxh_mean_grid,
            pmxh_std_grid=pmxh_std_grid,
            z_grid=np.asarray(emu.z_grid, dtype=float),
            k_grid=k_grid,
            k_obs=k_obs,
            z_select=float(args.z_select),
        )

        emu_std_arg = (np.asarray(pmpsi_model_std, dtype=float) if not bool(args.no_emu_std) else None)
        ll = _log_likelihood_asymmetric_gaussian(
            y_obs=pmpsi_obs,
            y_model=pmpsi_model,
            sigma_minus=pmpsi_sigma_minus,
            sigma_plus=pmpsi_sigma_plus,
            emu_std=emu_std_arg,
        )
        return float(lp + ll)

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step_start = int(restart_n_steps) if restart_path is not None else 0
    n_accept_total = int(restart_n_accept) if restart_path is not None else 0
    if restart_path is not None:
        x = np.asarray(restart_last_x, dtype=float).copy()
        lp = float(restart_last_logp)
        if not math.isfinite(lp):
            lp = float(log_post(x))
    else:
        if bool(args.tvir_sample_log10):
            x = np.asarray([float(args.zeta_init), float(args.logtvir_init)], dtype=float)
        else:
            x = np.asarray([float(args.zeta_init), float(args.tvir_init)], dtype=float)
        lp = float(log_post(x))

    nsteps = int(args.nsteps)
    if nsteps <= 0:
        raise ValueError("--nsteps must be > 0")

    remaining_after_burn = max(0, (step_start + nsteps) - burn)
    n_save_new = (remaining_after_burn + thin - 1) // thin
    already_after_burn = max(0, step_start - burn)
    n_skip_save = (already_after_burn + thin - 1) // thin
    n_save_new = max(0, n_save_new - n_skip_save)

    new_chain = np.empty((int(n_save_new), 2), dtype=float)
    new_logp = np.empty(int(n_save_new), dtype=float)
    save_idx = 0

    def write_checkpoint(*, steps_done: int, compress: bool) -> None:
        nonlocal save_idx
        chain_out = new_chain[:save_idx, :]
        logp_out = new_logp[:save_idx]
        if restart_chain is not None:
            chain_out = np.vstack([restart_chain, chain_out]) if chain_out.size else np.asarray(restart_chain, dtype=float)
            logp_out = np.concatenate([restart_logp, logp_out], axis=0) if logp_out.size else np.asarray(restart_logp, dtype=float)

        n_total_steps = int(step_start + steps_done)
        acc_rate = float(n_accept_total) / float(n_total_steps) if n_total_steps > 0 else float("nan")
        _save_npz_atomic(
            out_path=out_path,
            compress=bool(compress),
            chain=np.asarray(chain_out, dtype=float),
            logp=np.asarray(logp_out, dtype=float),
            acc_rate=float(acc_rate),
            k_obs=np.asarray(k_obs, dtype=float),
            pmpsi_obs=np.asarray(pmpsi_obs, dtype=float),
            pmpsi_sigma=np.asarray(pmpsi_sigma, dtype=float),
            pmpsi_sigma_minus=np.asarray(pmpsi_sigma_minus, dtype=float),
            pmpsi_sigma_plus=np.asarray(pmpsi_sigma_plus, dtype=float),
            pmpsi_chain_root=str(args.pmpsi_chain_root),
            ignore_rows=float(args.ignore_rows),
            emu_model=str(args.emu_model),
            base_params=np.asarray(base_params, dtype=float),
            zeta_idx=int(args.zeta_idx),
            tvir_idx=int(args.tvir_idx),
            tvir_in_log10=bool(args.tvir_in_log10),
            tvir_sample_log10=bool(args.tvir_sample_log10),
            zeta_min=float(args.zeta_min),
            zeta_max=float(args.zeta_max),
            tvir_min=float(args.tvir_min),
            tvir_max=float(args.tvir_max),
            logtvir_min=float(args.logtvir_min),
            logtvir_max=float(args.logtvir_max),
            z_select=float(args.z_select),
            no_emu_std=bool(args.no_emu_std),
            xhi_model=str(args.xhi_model),
            xhi_z=float(args.xhi_z),
            xhi_max=float(args.xhi_max),
            nsteps=int(nsteps),
            seed=int(args.seed),
            zeta_step=float(args.zeta_step),
            tvir_step=float(args.tvir_step),
            logtvir_step=float(args.logtvir_step),
            burn=int(burn),
            thin=int(thin),
            restart=str(restart_path) if restart_path is not None else "",
            n_prev=int(restart_chain.shape[0]) if restart_chain is not None else 0,
            last_x=np.asarray(x, dtype=float),
            last_logp=float(lp),
            n_total_steps=int(n_total_steps),
            n_total_accept=int(n_accept_total),
        )

    write_checkpoint(steps_done=0, compress=False)

    for i in range(int(nsteps)):
        prop = x + rng.normal(loc=0.0, scale=step_scales, size=x.size)
        lp_prop = float(log_post(prop))

        accept = False
        if math.isfinite(lp_prop):
            if lp_prop >= lp:
                accept = True
            else:
                if rng.random() < math.exp(lp_prop - lp):
                    accept = True

        if accept:
            x = prop
            lp = lp_prop
            n_accept_total += 1

        step_idx = int(step_start + i)
        if step_idx >= burn:
            if ((step_idx - burn) % thin) == 0:
                if save_idx < new_chain.shape[0]:
                    new_chain[save_idx, :] = x
                    new_logp[save_idx] = lp
                    save_idx += 1

        steps_done = int(i + 1)
        if checkpoint_every > 0 and (steps_done % checkpoint_every) == 0:
            write_checkpoint(steps_done=steps_done, compress=False)
        if print_every > 0 and (steps_done % print_every) == 0:
            n_total_steps = int(step_start + steps_done)
            acc_rate = float(n_accept_total) / float(n_total_steps) if n_total_steps > 0 else float("nan")
            print(f"steps: {n_total_steps}  acc_rate: {acc_rate:.3f}  n_saved: {int((restart_chain.shape[0] if restart_chain is not None else 0) + save_idx)}")

    write_checkpoint(steps_done=int(nsteps), compress=True)

    n_total_steps = int(step_start + nsteps)
    acc_rate = float(n_accept_total) / float(n_total_steps) if n_total_steps > 0 else float("nan")
    n_saved_total = int((restart_chain.shape[0] if restart_chain is not None else 0) + save_idx)
    print(f"saved: {out_path}")
    print(f"acc_rate: {acc_rate:.3f}")
    print(f"n_saved: {n_saved_total}")


if __name__ == "__main__":
    main()
