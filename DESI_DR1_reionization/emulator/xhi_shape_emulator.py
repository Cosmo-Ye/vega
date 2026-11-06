from __future__ import annotations

import argparse
import dataclasses
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


def _parse_sobol_txt(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"Empty sobol file: {path}")
    return np.asarray(rows, dtype=float)


def _extract_index(p: Path) -> int:
    m = re.search(r"(\d+)$", p.name)
    if m is None:
        raise ValueError(f"Cannot infer index from filename: {p.name}")
    return int(m.group(1))


def _load_xhi_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xHI file format: {path}")
    z = arr[:, 0]
    xhi = arr[:, 1]
    return z, xhi


def _monotone_clip_xhi(z: np.ndarray, xhi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = np.argsort(z)
    z = np.asarray(z[o], dtype=float)
    x = np.asarray(xhi[o], dtype=float)
    x = np.clip(x, 0.0, 1.0)
    x = np.maximum.accumulate(x)
    return z, x


def _z_at_x(z: np.ndarray, xhi: np.ndarray, target: float) -> float:
    z2, x2 = _monotone_clip_xhi(z, xhi)
    if not (x2[0] <= target <= x2[-1]):
        return float("nan")
    return float(np.interp(float(target), x2, z2))

def _log_cosh(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    a = np.abs(u)
    return a + np.log1p(np.exp(-2.0 * a)) - math.log(2.0)


def _solve_slopes(*, z10: float, z50: float, z90: float, tau: float) -> tuple[float, float]:
    logit10 = -math.log(9.0)
    logit90 = math.log(9.0)

    d10 = float(z10 - z50)
    d90 = float(z90 - z50)

    u10 = d10 / tau
    u90 = d90 / tau
    l10 = float(tau * _log_cosh(np.asarray([u10], dtype=float))[0])
    l90 = float(tau * _log_cosh(np.asarray([u90], dtype=float))[0])

    c1_10 = 0.5 * (d10 - l10)
    c2_10 = 0.5 * (d10 + l10)
    c1_90 = 0.5 * (d90 - l90)
    c2_90 = 0.5 * (d90 + l90)

    mat = np.asarray([[c1_10, c2_10], [c1_90, c2_90]], dtype=float)
    rhs = np.asarray([logit10, logit90], dtype=float)
    sol = np.linalg.solve(mat, rhs)
    return float(sol[0]), float(sol[1])


def _mean_curve_skew(
    z: np.ndarray,
    z50: float,
    log_dzl: float,
    log_dzr: float,
    tau_frac: float,
    tau: float | None = None,
) -> np.ndarray:
    dzl = float(math.exp(float(log_dzl)))
    dzr = float(math.exp(float(log_dzr)))

    if tau is None:
        tau = float(max(1e-6, float(tau_frac) * min(dzl, dzr)))
    else:
        tau = float(max(1e-6, float(tau)))
    z10 = float(z50 - dzl)
    z90 = float(z50 + dzr)
    s1, s2 = _solve_slopes(z10=z10, z50=z50, z90=z90, tau=tau)

    dz = np.asarray(z, dtype=float) - float(z50)
    u = dz / tau
    l = tau * _log_cosh(u)
    c1 = 0.5 * (dz - l)
    c2 = 0.5 * (dz + l)
    t = c1 * s1 + c2 * s2
    return _sigmoid(t)

def _pchip_tangents(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    if n < 2:
        raise ValueError("Need at least 2 points for interpolation")

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing")
    delta = np.diff(y) / h
    m = np.zeros(n, dtype=float)

    if n == 2:
        m[0] = delta[0]
        m[1] = delta[0]
        return m

    for i in range(1, n - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or (delta[i - 1] > 0) != (delta[i] > 0):
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    m0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if (m0 > 0) != (delta[0] > 0):
        m0 = 0.0
    elif abs(m0) > 3.0 * abs(delta[0]):
        m0 = 3.0 * delta[0]
    m[0] = m0

    mn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if (mn > 0) != (delta[-1] > 0):
        mn = 0.0
    elif abs(mn) > 3.0 * abs(delta[-1]):
        mn = 3.0 * delta[-1]
    m[-1] = mn

    return m


def _pchip_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    m = _pchip_tangents(x, y)

    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]
    h = x1 - x0
    t = (xq - x0) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def _mean_curve_from_anchors(
    z: np.ndarray,
    *,
    z05: float,
    z25: float,
    z50: float,
    z75: float,
    z10: float,
    z90: float,
    clip_eps: float,
) -> np.ndarray:
    eps_z = 1e-3
    z05 = float(min(z05, z10 - eps_z))
    z25 = float(min(max(z25, z10 + eps_z), z50 - eps_z))
    z75 = float(min(max(z75, z50 + eps_z), z90 - eps_z))

    x_left = float(max(1e-4, 10.0 * float(clip_eps)))
    logit05 = float(_logit(np.asarray([0.05], dtype=float))[0])
    logit10 = float(_logit(np.asarray([0.1], dtype=float))[0])
    logit_left = float(_logit(np.asarray([x_left], dtype=float))[0])
    dz_05_10 = float(z10 - z05)
    z01 = float("nan")
    if dz_05_10 > eps_z:
        slope = (logit10 - logit05) / dz_05_10
        if math.isfinite(slope) and slope > 0:
            z01 = float(z05 + (logit_left - logit05) / slope)
            z01 = float(min(z01, z05 - eps_z))
            z01 = float(max(z01, z05 - 5.0 * dz_05_10))
            z01 = float(min(z01, z05 - 2.0 * dz_05_10))

    if math.isfinite(z01):
        z_knots = [z01, z05, z10, z25, z50, z75, z90]
        x_knots = [x_left, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
    else:
        z_knots = [z05, z10, z25, z50, z75, z90]
        x_knots = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]

    dzr = float(z90 - z50)
    if math.isfinite(dzr) and dzr > eps_z:
        logit90 = float(_logit(np.asarray([0.9], dtype=float))[0])
        if math.isfinite(logit90) and logit90 > 0:
            scale = dzr / logit90
            x_right = float(1.0 - x_left)
            x99 = 0.99
            logit99 = float(_logit(np.asarray([x99], dtype=float))[0])
            logit_right = float(_logit(np.asarray([x_right], dtype=float))[0])
            z99 = float(z50 + scale * logit99) if math.isfinite(logit99) else float("nan")
            z_right = float(z50 + scale * logit_right) if math.isfinite(logit_right) else float("nan")
            if math.isfinite(z99) and z99 > float(z90) + eps_z:
                z_knots.append(z99)
                x_knots.append(x99)
            if math.isfinite(z_right) and z_right > float(z_knots[-1]) + eps_z:
                z_knots.append(z_right)
                x_knots.append(x_right)

    z_knots = np.asarray(z_knots, dtype=float)
    x_knots = np.asarray(x_knots, dtype=float)

    for i in range(1, len(z_knots)):
        if not (z_knots[i] > z_knots[i - 1] + eps_z):
            z_knots[i] = z_knots[i - 1] + eps_z

    y_knots = _logit(x_knots)

    z_arr = np.asarray(z, dtype=float)
    m = _pchip_tangents(z_knots, y_knots)
    zc = np.clip(z_arr, z_knots[0], z_knots[-1])
    yq = _pchip_eval(z_knots, y_knots, zc)
    left = z_arr < z_knots[0]
    if np.any(left):
        yq[left] = y_knots[0] + m[0] * (z_arr[left] - z_knots[0])
    right = z_arr > z_knots[-1]
    if np.any(right):
        yq[right] = y_knots[-1] + m[-1] * (z_arr[right] - z_knots[-1])
    return _sigmoid(yq)

def _poly_features_2d(X: np.ndarray, degree: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] < 2:
        raise ValueError("Expect at least two parameters for polynomial features")
    zeta = X[:, 0]
    tvir = X[:, 1]
    if degree <= 1:
        return np.stack([np.ones_like(zeta), zeta, tvir], axis=1)
    return np.stack(
        [np.ones_like(zeta), zeta, tvir, zeta * zeta, zeta * tvir, tvir * tvir],
        axis=1,
    )


@dataclasses.dataclass
class XHIShapeEmulator:
    param_scaler: StandardScaler
    gp_z05: GaussianProcessRegressor | None
    gp_z25: GaussianProcessRegressor | None
    gp_z50: GaussianProcessRegressor
    gp_z75: GaussianProcessRegressor | None
    gp_log_dzl: GaussianProcessRegressor
    gp_log_dzr: GaussianProcessRegressor
    train_params: np.ndarray
    train_indices: np.ndarray
    clip_eps: float = 1e-6
    d0: float = 0.5
    sigma_z: np.ndarray | None = None
    sigma_xhi: np.ndarray | None = None
    sigma_logit: np.ndarray | None = None
    corr_shape: np.ndarray | None = None
    calib_z_hi: float = 12.0
    calib_blend_width: float = 0.5
    calib_scale_mid: float = 1.0
    calib_scale_high: float = 1.0
    calib_floor_high: float = 0.0
    calib_mean_edge: float = 0.02
    calib_mean_blend: float = 0.01
    calib_floor_low: float = 0.0
    z50_trend_coef: np.ndarray | None = None
    z25_trend_coef: np.ndarray | None = None
    z25_trend_degree: int = 2
    z75_trend_coef: np.ndarray | None = None
    z75_trend_degree: int = 2
    log_dzl_trend_coef: np.ndarray | None = None
    log_dzl_trend_degree: int = 2
    log_dzr_trend_coef: np.ndarray | None = None
    log_dzr_trend_degree: int = 2
    z50_trend_degree: int = 2
    tau_frac: float = 0.08

    @classmethod
    def train_from_files(
        cls,
        sobol_path: str | Path,
        xhi_dir: str | Path,
        file_glob: str = "xH_concat_emu_*",
        clip_eps: float = 1e-6,
        optimize_kernel: bool = False,
        random_state: int = 0,
        min_dz: float = 0.25,
        tau_frac: float = 0.08,
    ) -> "XHIShapeEmulator":
        sobol_path = Path(sobol_path)
        xhi_dir = Path(xhi_dir)
        params_all = _parse_sobol_txt(sobol_path)

        xhi_paths = sorted(xhi_dir.glob(file_glob), key=_extract_index)
        if not xhi_paths:
            raise ValueError(f"No xHI files found under {xhi_dir} with glob {file_glob}")

        indices: list[int] = []
        z50_list: list[float] = []
        log_dzl_list: list[float] = []
        log_dzr_list: list[float] = []
        z05_list: list[float] = []
        z25_list: list[float] = []
        z75_list: list[float] = []

        z_sigma: np.ndarray | None = None
        sumsq_sigma: np.ndarray | None = None
        sumsq_logit: np.ndarray | None = None
        n_sigma = 0
        xhi_train: list[np.ndarray] = []

        for p in xhi_paths:
            idx = _extract_index(p)
            if idx >= len(params_all):
                continue
            z, xhi = _load_xhi_curve(p)
            z10 = _z_at_x(z, xhi, 0.1)
            z50 = _z_at_x(z, xhi, 0.5)
            z90 = _z_at_x(z, xhi, 0.9)
            if not (math.isfinite(z10) and math.isfinite(z50) and math.isfinite(z90)):
                continue
            if not (z10 < z50 < z90):
                continue
            dzl = float(z50 - z10)
            dzr = float(z90 - z50)
            if not (dzl > float(min_dz) and dzr > float(min_dz)):
                continue
            indices.append(idx)
            z50_list.append(float(z50))
            log_dzl_list.append(float(math.log(dzl)))
            log_dzr_list.append(float(math.log(dzr)))
            z05 = _z_at_x(z, xhi, 0.05)
            z25 = _z_at_x(z, xhi, 0.25)
            z75 = _z_at_x(z, xhi, 0.75)
            z05_use = float(z05) if (math.isfinite(z05) and z05 < z10) else float("nan")
            z25_use = float(z25) if (math.isfinite(z25) and z10 < z25 < z50) else float("nan")
            z75_use = float(z75) if (math.isfinite(z75) and z50 < z75 < z90) else float("nan")
            z05_list.append(z05_use)
            z25_list.append(z25_use)
            z75_list.append(z75_use)
            if z_sigma is None:
                z_sigma = np.asarray(z, dtype=float)
            if z_sigma.size != z.size or float(np.max(np.abs(np.asarray(z, dtype=float) - z_sigma))) > 1e-12:
                xhi_train.append(np.interp(z_sigma, np.asarray(z, dtype=float), np.asarray(xhi, dtype=float)).astype(float))
            else:
                xhi_train.append(np.asarray(xhi, dtype=float))

            if z_sigma is None:
                z_sigma = np.asarray(z, dtype=float)
                sumsq_sigma = np.zeros_like(z_sigma, dtype=float)
                sumsq_logit = np.zeros_like(z_sigma, dtype=float)

            if sumsq_sigma is not None:
                log_dzl = float(math.log(dzl))
                log_dzr = float(math.log(dzr))
                dzl_loc = float(math.exp(log_dzl))
                dzr_loc = float(math.exp(log_dzr))
                z10_loc = float(z50 - dzl_loc)
                z90_loc = float(z50 + dzr_loc)
                if not (math.isfinite(z10_loc) and math.isfinite(z90_loc) and z10_loc < z50 < z90_loc):
                    pred = _mean_curve_skew(np.asarray(z, dtype=float), float(z50), log_dzl, log_dzr, float(tau_frac))
                else:
                    if not math.isfinite(z05_use):
                        z05_use2 = float(z10_loc - 0.5 * dzl_loc)
                    else:
                        z05_use2 = float(z05_use)
                    if not math.isfinite(z25_use):
                        z25_use2 = float(0.5 * (z10_loc + z50))
                    else:
                        z25_use2 = float(z25_use)
                    if not math.isfinite(z75_use):
                        z75_use2 = float(0.5 * (z50 + z90_loc))
                    else:
                        z75_use2 = float(z75_use)
                    pred = _mean_curve_from_anchors(
                        np.asarray(z, dtype=float),
                        z05=float(z05_use2),
                        z25=float(z25_use2),
                        z50=float(z50),
                        z75=float(z75_use2),
                        z10=float(z10_loc),
                        z90=float(z90_loc),
                        clip_eps=float(clip_eps),
                    )

                zc = np.asarray(z, dtype=float)
                xc = np.asarray(xhi, dtype=float)
                pc = np.asarray(pred, dtype=float)

                if z_sigma.size != zc.size or float(np.max(np.abs(zc - z_sigma))) > 1e-12:
                    xc = np.interp(z_sigma, zc, xc)
                    pc = np.interp(z_sigma, zc, pc)
                res = pc - xc
                sumsq_sigma += res * res
                x_left = float(max(1e-4, 10.0 * float(clip_eps)))
                xt = np.clip(xc, x_left, 1.0 - x_left)
                xp = np.clip(pc, x_left, 1.0 - x_left)
                rlog = _logit(xp) - _logit(xt)
                if sumsq_logit is not None:
                    sumsq_logit += rlog * rlog
                n_sigma += 1

        if not indices:
            raise ValueError("No usable xHI curves were loaded (z10/z50/z90 missing).")

        X = params_all[np.asarray(indices, dtype=int)]
        y_z05 = np.asarray(z05_list, dtype=float)
        y_z25 = np.asarray(z25_list, dtype=float)
        y_z50 = np.asarray(z50_list, dtype=float)
        y_z75 = np.asarray(z75_list, dtype=float)
        y_log_dzl = np.asarray(log_dzl_list, dtype=float)
        y_log_dzr = np.asarray(log_dzr_list, dtype=float)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        F_poly = _poly_features_2d(Xs, degree=2)
        coef_z50, *_ = np.linalg.lstsq(F_poly, y_z50, rcond=None)
        y_z50_resid = y_z50 - F_poly @ coef_z50
        coef_log_dzl, *_ = np.linalg.lstsq(F_poly, y_log_dzl, rcond=None)
        y_log_dzl_resid = y_log_dzl - F_poly @ coef_log_dzl
        coef_log_dzr, *_ = np.linalg.lstsq(F_poly, y_log_dzr, rcond=None)
        y_log_dzr_resid = y_log_dzr - F_poly @ coef_log_dzr

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(Xs.shape[1]), length_scale_bounds=(1e-2, 1e5))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1.0))
        )

        gp_z50 = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            optimizer=None if not optimize_kernel else "fmin_l_bfgs_b",
            n_restarts_optimizer=0 if not optimize_kernel else 2,
            random_state=random_state,
        )
        gp_z05 = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            optimizer=None if not optimize_kernel else "fmin_l_bfgs_b",
            n_restarts_optimizer=0 if not optimize_kernel else 2,
            random_state=random_state,
        )
        gp_z25 = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            optimizer=None if not optimize_kernel else "fmin_l_bfgs_b",
            n_restarts_optimizer=0 if not optimize_kernel else 2,
            random_state=random_state,
        )
        gp_z75 = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            optimizer=None if not optimize_kernel else "fmin_l_bfgs_b",
            n_restarts_optimizer=0 if not optimize_kernel else 2,
            random_state=random_state,
        )
        gp_log_dzl = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            optimizer=None if not optimize_kernel else "fmin_l_bfgs_b",
            n_restarts_optimizer=0 if not optimize_kernel else 2,
            random_state=random_state,
        )
        gp_log_dzr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            optimizer=None if not optimize_kernel else "fmin_l_bfgs_b",
            n_restarts_optimizer=0 if not optimize_kernel else 2,
            random_state=random_state,
        )

        min_n_aux = 10
        mask_z05 = np.isfinite(y_z05)
        gp_z05_fit: GaussianProcessRegressor | None = None
        if int(np.count_nonzero(mask_z05)) >= min_n_aux:
            gp_z05.fit(Xs[mask_z05], y_z05[mask_z05])
            gp_z05_fit = gp_z05

        mask_z25 = np.isfinite(y_z25)
        gp_z25_fit: GaussianProcessRegressor | None = None
        coef_z25_fit: np.ndarray | None = None
        if int(np.count_nonzero(mask_z25)) >= 6:
            coef_z25_fit, *_ = np.linalg.lstsq(F_poly[mask_z25], y_z25[mask_z25], rcond=None)
            y_z25_resid = y_z25[mask_z25] - F_poly[mask_z25] @ coef_z25_fit
        else:
            y_z25_resid = None
        if int(np.count_nonzero(mask_z25)) >= min_n_aux and y_z25_resid is not None:
            gp_z25.fit(Xs[mask_z25], y_z25_resid)
            gp_z25_fit = gp_z25

        gp_z50.fit(Xs, y_z50_resid)
        mask_z75 = np.isfinite(y_z75)
        coef_z75_fit: np.ndarray | None = None
        if int(np.count_nonzero(mask_z75)) >= 6:
            coef_z75_fit, *_ = np.linalg.lstsq(F_poly[mask_z75], y_z75[mask_z75], rcond=None)
            y_z75_resid = y_z75[mask_z75] - F_poly[mask_z75] @ coef_z75_fit
        else:
            y_z75_resid = None
        gp_z75_fit: GaussianProcessRegressor | None = None
        if int(np.count_nonzero(mask_z75)) >= min_n_aux and y_z75_resid is not None:
            gp_z75.fit(Xs[mask_z75], y_z75_resid)
            gp_z75_fit = gp_z75

        gp_log_dzl.fit(Xs, y_log_dzl)
        gp_log_dzl.fit(Xs, y_log_dzl_resid)
        gp_log_dzr.fit(Xs, y_log_dzr_resid)

        sigma_xhi = None
        sigma_logit = None
        if sumsq_sigma is not None and z_sigma is not None and int(n_sigma) > 0:
            sigma_xhi = np.sqrt(np.maximum(sumsq_sigma / float(n_sigma), 0.0))
        if sumsq_logit is not None and z_sigma is not None and int(n_sigma) > 0:
            sigma_logit = np.sqrt(np.maximum(sumsq_logit / float(n_sigma), 0.0))

        def corr_pair(a: np.ndarray, b: np.ndarray, *, min_n: int = 12) -> float:
            ma = np.isfinite(a)
            mb = np.isfinite(b)
            m = ma & mb
            if int(np.count_nonzero(m)) < int(min_n):
                return 0.0
            aa = np.asarray(a[m], dtype=float)
            bb = np.asarray(b[m], dtype=float)
            sa = float(np.std(aa))
            sb = float(np.std(bb))
            if not (math.isfinite(sa) and math.isfinite(sb) and sa > 0.0 and sb > 0.0):
                return 0.0
            c = float(np.corrcoef(aa, bb)[0, 1])
            if not math.isfinite(c):
                return 0.0
            return float(np.clip(c, -0.98, 0.98))

        z05_resid = np.full_like(y_z50, float("nan"))
        if gp_z05_fit is not None:
            v = np.asarray(y_z05, dtype=float)
            z05_resid = v - float(np.nanmean(v))

        z25_resid = np.full_like(y_z50, float("nan"))
        if y_z25_resid is not None:
            z25_resid[np.asarray(mask_z25, dtype=bool)] = np.asarray(y_z25_resid, dtype=float)

        z75_resid = np.full_like(y_z50, float("nan"))
        if y_z75_resid is not None:
            z75_resid[np.asarray(mask_z75, dtype=bool)] = np.asarray(y_z75_resid, dtype=float)

        feats = [
            np.asarray(y_z50_resid, dtype=float),
            np.asarray(y_log_dzl_resid, dtype=float),
            np.asarray(y_log_dzr_resid, dtype=float),
            np.asarray(z05_resid, dtype=float),
            np.asarray(z25_resid, dtype=float),
            np.asarray(z75_resid, dtype=float),
        ]
        corr_shape = np.eye(len(feats), dtype=float)
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                c = corr_pair(feats[i], feats[j])
                corr_shape[i, j] = c
                corr_shape[j, i] = c

        emu = cls(
            param_scaler=scaler,
            gp_z05=gp_z05_fit,
            gp_z25=gp_z25_fit,
            gp_z50=gp_z50,
            gp_z75=gp_z75_fit,
            gp_log_dzl=gp_log_dzl,
            gp_log_dzr=gp_log_dzr,
            train_params=X,
            train_indices=np.asarray(indices, dtype=int),
            z25_trend_coef=coef_z25_fit,
            log_dzl_trend_coef=coef_log_dzl,
            z75_trend_coef=coef_z75_fit,
            log_dzr_trend_coef=coef_log_dzr,
            z50_trend_coef=coef_z50,
            z50_trend_degree=2,
            clip_eps=clip_eps,
            d0=0.5,
            sigma_z=np.asarray(z_sigma, dtype=float) if z_sigma is not None else None,
            sigma_xhi=np.asarray(sigma_xhi, dtype=float) if sigma_xhi is not None else None,
            sigma_logit=np.asarray(sigma_logit, dtype=float) if sigma_logit is not None else None,
            corr_shape=np.asarray(corr_shape, dtype=float),
            tau_frac=float(tau_frac),
        )

        def coverage(err: np.ndarray, std: np.ndarray, n_sigma: float, min_std: float) -> float:
            e = np.asarray(err, dtype=float).ravel()
            s = np.asarray(std, dtype=float).ravel()
            if e.size == 0:
                return float("nan")
            ss = np.maximum(np.abs(s), float(min_std))
            return float(np.mean(np.abs(e) <= float(n_sigma) * ss))

        def objective(err: np.ndarray, std: np.ndarray, scale: float) -> float:
            s = float(scale)
            c68 = coverage(err, std * s, 1.0, 1e-8)
            c95 = coverage(err, std * s, 2.0, 1e-8)
            return float((c68 - 0.68) * (c68 - 0.68) + (c95 - 0.95) * (c95 - 0.95))

        def solve_scale(err: np.ndarray, std: np.ndarray) -> float:
            e = np.asarray(err, dtype=float).ravel()
            s = np.asarray(std, dtype=float).ravel()
            m = np.isfinite(e) & np.isfinite(s) & (s > 0.0)
            e = e[m]
            s = s[m]
            if e.size < 128:
                return 1.0
            best_log = 0.0
            best_val = float("inf")
            for log_s in np.linspace(-2.0, 1.0, 61):
                val = objective(e, s, math.exp(float(log_s)))
                if val < best_val:
                    best_val = val
                    best_log = float(log_s)
            for log_s in np.linspace(best_log - 0.2, best_log + 0.2, 81):
                val = objective(e, s, math.exp(float(log_s)))
                if val < best_val:
                    best_val = val
                    best_log = float(log_s)
            return float(math.exp(best_log))

        def solve_floor(err: np.ndarray, std: np.ndarray) -> float:
            e = np.asarray(err, dtype=float).ravel()
            s = np.asarray(std, dtype=float).ravel()
            m = np.isfinite(e) & np.isfinite(s) & (s >= 0.0)
            e = e[m]
            s = s[m]
            if e.size < 128:
                return 0.0
            best_log = -20.0
            best_val = float("inf")
            for log_f in np.linspace(-12.0, -2.0, 81):
                f = float(math.exp(float(log_f)))
                val = objective(e, np.sqrt(s * s + f * f), 1.0)
                if val < best_val:
                    best_val = val
                    best_log = float(log_f)
            for log_f in np.linspace(best_log - 0.3, best_log + 0.3, 121):
                f = float(math.exp(float(log_f)))
                val = objective(e, np.sqrt(s * s + f * f), 1.0)
                if val < best_val:
                    best_val = val
                    best_log = float(log_f)
            return float(math.exp(best_log))

        def solve_floor_robust(err: np.ndarray, std: np.ndarray) -> float:
            e = np.abs(np.asarray(err, dtype=float).ravel())
            s = np.asarray(std, dtype=float).ravel()
            m = np.isfinite(e) & np.isfinite(s) & (s >= 0.0)
            e = e[m]
            s = s[m]
            if e.size < 64:
                return 0.0
            best_log = -20.0
            best_val = float("inf")
            for log_f in np.linspace(-10.0, -1.0, 91):
                f = float(math.exp(float(log_f)))
                tot = np.sqrt(s * s + f * f)
                r = e / np.maximum(tot, 1e-12)
                q95 = float(np.quantile(r, 0.95))
                q99 = float(np.quantile(r, 0.99))
                val = float((q95 - 2.0) * (q95 - 2.0) + (q99 - 3.0) * (q99 - 3.0))
                if val < best_val:
                    best_val = val
                    best_log = float(log_f)
            for log_f in np.linspace(best_log - 0.3, best_log + 0.3, 121):
                f = float(math.exp(float(log_f)))
                tot = np.sqrt(s * s + f * f)
                r = e / np.maximum(tot, 1e-12)
                q95 = float(np.quantile(r, 0.95))
                q99 = float(np.quantile(r, 0.99))
                val = float((q95 - 2.0) * (q95 - 2.0) + (q99 - 3.0) * (q99 - 3.0))
                if val < best_val:
                    best_val = val
                    best_log = float(log_f)
            return float(math.exp(best_log))

        if z_sigma is not None:
            zsig = np.asarray(z_sigma, dtype=float)
            mid_mask = (zsig >= 5.0) & (zsig <= float(emu.calib_z_hi))
            high_mask = zsig > float(emu.calib_z_hi)
            emu.calib_scale_high = 1.0
            emu.calib_scale_mid = 1.0

            calib_dir = Path("test_data/xHI")
            calib_sobol = Path("sobol_inner.txt")
            calib_files = sorted(calib_dir.glob("xH_concat_emu_*"), key=_extract_index) if calib_dir.exists() else []
            if not calib_files or not calib_sobol.exists():
                calib_dir = xhi_dir
                calib_sobol = sobol_path
                calib_files = xhi_paths

            params_cal = _parse_sobol_txt(calib_sobol)
            errs_mid: list[np.ndarray] = []
            stds_mid: list[np.ndarray] = []
            errs_high: list[np.ndarray] = []
            stds_high: list[np.ndarray] = []
            errs_low: list[np.ndarray] = []
            stds_low: list[np.ndarray] = []
            edge = float(getattr(emu, "calib_mean_edge", 0.05))
            for fp in calib_files:
                idx = _extract_index(fp)
                if idx >= len(params_cal):
                    continue
                zc, xc = _load_xhi_curve(fp)
                if zsig.size != zc.size or float(np.max(np.abs(np.asarray(zc, dtype=float) - zsig))) > 1e-12:
                    xc = np.interp(zsig, np.asarray(zc, dtype=float), np.asarray(xc, dtype=float))
                mean, std = emu.predict_many(params=params_cal[idx], z_values=zsig)
                err = np.asarray(mean, dtype=float) - np.asarray(xc, dtype=float)
                if bool(np.any(mid_mask)):
                    errs_mid.append(err[mid_mask])
                    stds_mid.append(np.asarray(std, dtype=float)[mid_mask])
                    if math.isfinite(edge) and edge > 0.0:
                        low_mask = mid_mask & (np.asarray(mean, dtype=float) < edge)
                        if bool(np.any(low_mask)):
                            errs_low.append(err[low_mask])
                            stds_low.append(np.asarray(std, dtype=float)[low_mask])
                if bool(np.any(high_mask)):
                    errs_high.append(err[high_mask])
                    stds_high.append(np.asarray(std, dtype=float)[high_mask])

            if errs_mid:
                emu.calib_scale_mid = solve_scale(np.concatenate(errs_mid), np.concatenate(stds_mid))
            if errs_high:
                emu.calib_scale_high = 1.0
                emu.calib_floor_high = solve_floor(np.concatenate(errs_high), np.concatenate(stds_high))

            if errs_low:
                e = np.abs(np.concatenate(errs_low))
                emu.calib_floor_low = solve_floor_robust(np.concatenate(errs_low), np.concatenate(stds_low))
                if e.size >= 10:
                    q = float(np.quantile(e, 0.999))
                    if math.isfinite(q) and q > 0.0:
                        emu.calib_floor_low = float(max(emu.calib_floor_low, q / 3.0))
            else:
                if math.isfinite(edge) and edge > 0.0:
                    errs_low_tr: list[np.ndarray] = []
                    stds_low_tr: list[np.ndarray] = []
                    for fp in xhi_paths:
                        idx = _extract_index(fp)
                        if idx >= len(params_all):
                            continue
                        zc, xc = _load_xhi_curve(fp)
                        if zsig.size != zc.size or float(np.max(np.abs(np.asarray(zc, dtype=float) - zsig))) > 1e-12:
                            xc = np.interp(zsig, np.asarray(zc, dtype=float), np.asarray(xc, dtype=float))
                        mean, std = emu.predict_many(params=params_all[idx], z_values=zsig)
                        low_mask = mid_mask & (np.asarray(mean, dtype=float) < edge)
                        if bool(np.any(low_mask)):
                            errs_low_tr.append((np.asarray(mean, dtype=float) - np.asarray(xc, dtype=float))[low_mask])
                            stds_low_tr.append(np.asarray(std, dtype=float)[low_mask])
                    if errs_low_tr:
                        e = np.abs(np.concatenate(errs_low_tr))
                        emu.calib_floor_low = solve_floor_robust(np.concatenate(errs_low_tr), np.concatenate(stds_low_tr))
                        if e.size >= 10:
                            q = float(np.quantile(e, 0.999))
                            if math.isfinite(q) and q > 0.0:
                                emu.calib_floor_low = float(max(emu.calib_floor_low, q / 3.0))

        return emu

    def _sigma_floor(self, z: np.ndarray, *, mean: np.ndarray | None = None) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z_sigma = getattr(self, "sigma_z", None)
        s_xhi = getattr(self, "sigma_xhi", None)
        if z_sigma is None or s_xhi is None:
            return np.zeros_like(z, dtype=float)
        z_sigma = np.asarray(z_sigma, dtype=float).ravel()
        if z_sigma.size < 2:
            return np.zeros_like(z, dtype=float)
        s_xhi = np.asarray(s_xhi, dtype=float).ravel()
        if s_xhi.size != z_sigma.size:
            return np.zeros_like(z, dtype=float)
        return np.interp(z, z_sigma, s_xhi, left=float(s_xhi[0]), right=float(s_xhi[-1])).astype(float)

    def predict_z05(self, params: Sequence[float]) -> tuple[float, float]:
        gp = getattr(self, "gp_z05", None)
        if gp is None:
            return float("nan"), float("nan")
        params_arr = np.asarray(params, dtype=float).reshape(1, -1)
        Xs = self.param_scaler.transform(params_arr)
        z05_mu, z05_std = gp.predict(Xs, return_std=True)
        return float(z05_mu[0]), float(z05_std[0])

    def predict_z25(self, params: Sequence[float]) -> tuple[float, float]:
        params_arr = np.asarray(params, dtype=float).reshape(1, -1)
        Xs = self.param_scaler.transform(params_arr)
        coef = getattr(self, "z25_trend_coef", None)
        trend = float(np.dot(_poly_features_2d(Xs, degree=int(getattr(self, "z25_trend_degree", 2)))[0], coef)) if coef is not None else 0.0
        gp = getattr(self, "gp_z25", None)
        if gp is None:
            if coef is None:
                return float("nan"), float("nan")
            return float(trend), 0.0
        mu_resid, std_resid = gp.predict(Xs, return_std=True)
        train_Xs = self.param_scaler.transform(np.asarray(self.train_params, dtype=float))
        d = float(np.sqrt(((train_Xs - Xs[0]) ** 2).sum(axis=1)).min())
        d0 = float(getattr(self, "d0", 0.5))
        w = float(1.0 / (1.0 + (d / d0) * (d / d0))) if d0 > 0 else 1.0
        mu = float(trend + w * float(mu_resid[0]))
        std = float(std_resid[0])
        return mu, std

    def predict_z75(self, params: Sequence[float]) -> tuple[float, float]:
        params_arr = np.asarray(params, dtype=float).reshape(1, -1)
        Xs = self.param_scaler.transform(params_arr)
        coef = getattr(self, "z75_trend_coef", None)
        trend = float(np.dot(_poly_features_2d(Xs, degree=int(getattr(self, "z75_trend_degree", 2)))[0], coef)) if coef is not None else 0.0
        gp = getattr(self, "gp_z75", None)
        if gp is None:
            if coef is None:
                return float("nan"), float("nan")
            return float(trend), 0.0
        mu_resid, std_resid = gp.predict(Xs, return_std=True)
        train_Xs = self.param_scaler.transform(np.asarray(self.train_params, dtype=float))
        d = float(np.sqrt(((train_Xs - Xs[0]) ** 2).sum(axis=1)).min())
        d0 = float(getattr(self, "d0", 0.5))
        w = float(1.0 / (1.0 + (d / d0) * (d / d0))) if d0 > 0 else 1.0
        mu = float(trend + w * float(mu_resid[0]))
        std = float(std_resid[0])
        return mu, std

    def predict_shape(self, params: Sequence[float]) -> tuple[float, float, float, float, float, float]:
        params_arr = np.asarray(params, dtype=float).reshape(1, -1)
        Xs = self.param_scaler.transform(params_arr)

        z50_trend = 0.0
        coef = getattr(self, "z50_trend_coef", None)
        if coef is not None:
            deg = int(getattr(self, "z50_trend_degree", 2))
            F = _poly_features_2d(Xs, degree=deg)
            z50_trend = float(np.dot(F[0], coef))
        z50_resid_mu, z50_resid_std = self.gp_z50.predict(Xs, return_std=True)
        z50_mu = float(z50_trend + float(z50_resid_mu[0]))
        z50_std = float(z50_resid_std[0])
        log_dzl_trend = 0.0
        coef_dzl = getattr(self, "log_dzl_trend_coef", None)
        if coef_dzl is not None:
            deg_dzl = int(getattr(self, "log_dzl_trend_degree", 2))
            F = _poly_features_2d(Xs, degree=deg_dzl)
            log_dzl_trend = float(np.dot(F[0], coef_dzl))
        log_dzl_resid_mu, log_dzl_resid_std = self.gp_log_dzl.predict(Xs, return_std=True)
        train_Xs = self.param_scaler.transform(np.asarray(self.train_params, dtype=float))
        d = float(np.sqrt(((train_Xs - Xs[0]) ** 2).sum(axis=1)).min())
        d0 = float(getattr(self, "d0", 0.5))
        w = float(1.0 / (1.0 + (d / d0) * (d / d0))) if d0 > 0 else 1.0
        log_dzl_mu = float(log_dzl_trend + w * float(log_dzl_resid_mu[0]))
        log_dzl_std = float(log_dzl_resid_std[0])
        log_dzr_trend = 0.0
        coef_dzr = getattr(self, "log_dzr_trend_coef", None)
        if coef_dzr is not None:
            deg_dzr = int(getattr(self, "log_dzr_trend_degree", 2))
            F = _poly_features_2d(Xs, degree=deg_dzr)
            log_dzr_trend = float(np.dot(F[0], coef_dzr))
        log_dzr_resid_mu, log_dzr_resid_std = self.gp_log_dzr.predict(Xs, return_std=True)
        train_Xs = self.param_scaler.transform(np.asarray(self.train_params, dtype=float))
        d = float(np.sqrt(((train_Xs - Xs[0]) ** 2).sum(axis=1)).min())
        d0 = float(getattr(self, "d0", 0.5))
        w = float(1.0 / (1.0 + (d / d0) * (d / d0))) if d0 > 0 else 1.0
        log_dzr_mu = float(log_dzr_trend + w * float(log_dzr_resid_mu[0]))
        log_dzr_std = float(log_dzr_resid_std[0])
        return (
            float(z50_mu),
            float(z50_std),
            float(log_dzl_mu),
            float(log_dzl_std),
            float(log_dzr_mu),
            float(log_dzr_std),
        )

    def _mean_curve(
        self,
        z: np.ndarray,
        *,
        z05: float,
        z25: float,
        z50: float,
        z75: float,
        log_dzl: float,
        log_dzr: float,
    ) -> np.ndarray:
        dzl = float(math.exp(float(log_dzl)))
        dzr = float(math.exp(float(log_dzr)))
        z10 = float(z50 - dzl)
        z90 = float(z50 + dzr)
        if not (math.isfinite(z10) and math.isfinite(z90) and z10 < z50 < z90):
            return _mean_curve_skew(z, z50, log_dzl, log_dzr, self.tau_frac)
        if not math.isfinite(z05):
            z05 = float(z10 - 0.5 * dzl)
        if not math.isfinite(z25):
            z25 = float(0.5 * (z10 + z50))
        if not math.isfinite(z75):
            z75 = float(0.5 * (z50 + z90))
        return _mean_curve_from_anchors(
            z,
            z05=z05,
            z25=z25,
            z50=z50,
            z75=z75,
            z10=z10,
            z90=z90,
            clip_eps=self.clip_eps,
        )

    def predict(self, params: Sequence[float], z: float) -> tuple[float, float]:
        mean, std = self.predict_many(params=params, z_values=[float(z)])
        return float(mean[0]), float(std[0])
        z50_mu, z50_std, log_dzl_mu, log_dzl_std, log_dzr_mu, log_dzr_std = self.predict_shape(params=params)
        z_arr = np.asarray([float(z)], dtype=float)
        z05_mu, z05_std = self.predict_z05(params=params)
        z25_mu, z25_std = self.predict_z25(params=params)
        z75_mu, z75_std = self.predict_z75(params=params)
        mean = float(
            self._mean_curve(
                z_arr,
                z05=z05_mu,
                z25=z25_mu,
                z50=z50_mu,
                z75=z75_mu,
                log_dzl=log_dzl_mu,
                log_dzr=log_dzr_mu,
            )[0]
        )

        var = 0.0
        if z50_std > 0:
            m_p = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu,
                    z50=z50_mu + z50_std,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            m_m = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu,
                    z50=z50_mu - z50_std,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            d = (m_p - m_m) / (2.0 * z50_std)
            var += float((d * d) * (z50_std * z50_std))
        if log_dzl_std > 0:
            m_p = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu,
                    z50=z50_mu,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu + log_dzl_std,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            m_m = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu,
                    z50=z50_mu,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu - log_dzl_std,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            d = (m_p - m_m) / (2.0 * log_dzl_std)
            var += float((d * d) * (log_dzl_std * log_dzl_std))
        if log_dzr_std > 0:
            m_p = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu,
                    z50=z50_mu,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu + log_dzr_std,
                )[0]
            )
            m_m = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu,
                    z50=z50_mu,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu - log_dzr_std,
                )[0]
            )
            d = (m_p - m_m) / (2.0 * log_dzr_std)
            var += float((d * d) * (log_dzr_std * log_dzr_std))
        if z05_std > 0 and math.isfinite(z05_mu):
            m_p = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu + z05_std,
                    z25=z25_mu,
                    z50=z50_mu,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            m_m = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu - z05_std,
                    z25=z25_mu,
                    z50=z50_mu,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            d = (m_p - m_m) / (2.0 * z05_std)
            var += float((d * d) * (z05_std * z05_std))
        if z25_std > 0 and math.isfinite(z25_mu):
            m_p = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu + z25_std,
                    z50=z50_mu,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            m_m = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu - z25_std,
                    z50=z50_mu,
                    z75=z75_mu,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            d = (m_p - m_m) / (2.0 * z25_std)
            var += float((d * d) * (z25_std * z25_std))
        if z75_std > 0 and math.isfinite(z75_mu):
            m_p = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu,
                    z50=z50_mu,
                    z75=z75_mu + z75_std,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            m_m = float(
                self._mean_curve(
                    z_arr,
                    z05=z05_mu,
                    z25=z25_mu,
                    z50=z50_mu,
                    z75=z75_mu - z75_std,
                    log_dzl=log_dzl_mu,
                    log_dzr=log_dzr_mu,
                )[0]
            )
            d = (m_p - m_m) / (2.0 * z75_std)
            var += float((d * d) * (z75_std * z75_std))

        if float(min(1.0, max(0.0, mean))) > 0.9:
            var_local = 0.0
            if z50_std > 0:
                dz = float(min(z50_std, 0.05))
                m_p = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu,
                        z50=z50_mu + dz,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                m_m = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu,
                        z50=z50_mu - dz,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                d = (m_p - m_m) / (2.0 * dz)
                var_local += float((d * d) * (z50_std * z50_std))
            if log_dzl_std > 0:
                dl = float(min(log_dzl_std, 0.02))
                m_p = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu,
                        z50=z50_mu,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu + dl,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                m_m = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu,
                        z50=z50_mu,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu - dl,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                d = (m_p - m_m) / (2.0 * dl)
                var_local += float((d * d) * (log_dzl_std * log_dzl_std))
            if log_dzr_std > 0:
                dr = float(min(log_dzr_std, 0.02))
                m_p = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu,
                        z50=z50_mu,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu + dr,
                    )[0]
                )
                m_m = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu,
                        z50=z50_mu,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu - dr,
                    )[0]
                )
                d = (m_p - m_m) / (2.0 * dr)
                var_local += float((d * d) * (log_dzr_std * log_dzr_std))
            if z05_std > 0 and math.isfinite(z05_mu):
                dz = float(min(z05_std, 0.05))
                m_p = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu + dz,
                        z25=z25_mu,
                        z50=z50_mu,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                m_m = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu - dz,
                        z25=z25_mu,
                        z50=z50_mu,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                d = (m_p - m_m) / (2.0 * dz)
                var_local += float((d * d) * (z05_std * z05_std))
            if z25_std > 0 and math.isfinite(z25_mu):
                dz = float(min(z25_std, 0.05))
                m_p = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu + dz,
                        z50=z50_mu,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                m_m = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu - dz,
                        z50=z50_mu,
                        z75=z75_mu,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                d = (m_p - m_m) / (2.0 * dz)
                var_local += float((d * d) * (z25_std * z25_std))
            if z75_std > 0 and math.isfinite(z75_mu):
                dz = float(min(z75_std, 0.05))
                m_p = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu,
                        z50=z50_mu,
                        z75=z75_mu + dz,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                m_m = float(
                    self._mean_curve(
                        z_arr,
                        z05=z05_mu,
                        z25=z25_mu,
                        z50=z50_mu,
                        z75=z75_mu - dz,
                        log_dzl=log_dzl_mu,
                        log_dzr=log_dzr_mu,
                    )[0]
                )
                d = (m_p - m_m) / (2.0 * dz)
                var_local += float((d * d) * (z75_std * z75_std))
            var = float(min(var, var_local))

            corr = getattr(self, "corr_shape", None)
            if corr is not None:
                C = np.asarray(corr, dtype=float)
                if C.shape == (6, 6):
                    g = [0.0] * 6
                    if z50_std > 0:
                        dz = float(min(z50_std, 0.05))
                        mp = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu + dz,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        mm = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu - dz,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        g[0] = float((mp - mm) / (2.0 * dz))
                    if log_dzl_std > 0:
                        dl = float(min(log_dzl_std, 0.02))
                        mp = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu + dl,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        mm = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu - dl,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        g[1] = float((mp - mm) / (2.0 * dl))
                    if log_dzr_std > 0:
                        dr = float(min(log_dzr_std, 0.02))
                        mp = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu + dr,
                            )[0]
                        )
                        mm = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu - dr,
                            )[0]
                        )
                        g[2] = float((mp - mm) / (2.0 * dr))
                    if z05_std > 0 and math.isfinite(z05_mu):
                        dz = float(min(z05_std, 0.05))
                        mp = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu + dz,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        mm = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu - dz,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        g[3] = float((mp - mm) / (2.0 * dz))
                    if z25_std > 0 and math.isfinite(z25_mu):
                        dz = float(min(z25_std, 0.05))
                        mp = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu + dz,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        mm = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu - dz,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        g[4] = float((mp - mm) / (2.0 * dz))
                    if z75_std > 0 and math.isfinite(z75_mu):
                        dz = float(min(z75_std, 0.05))
                        mp = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu + dz,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        mm = float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu - dz,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                        g[5] = float((mp - mm) / (2.0 * dz))

                    s = np.asarray([z50_std, log_dzl_std, log_dzr_std, z05_std, z25_std, z75_std], dtype=float)
                    w = s * np.asarray(g, dtype=float)
                    v_cov = float(w @ C @ w)
                    if math.isfinite(v_cov) and v_cov >= 0.0:
                        var = float(min(var, v_cov))

        mean_clip = float(min(1.0, max(0.0, mean)))
        if mean_clip > 0.9:
            x_left = float(max(1e-4, 10.0 * float(self.clip_eps)))
            mc = float(np.clip(mean_clip, x_left, 1.0 - x_left))
            jac = float(mc * (1.0 - mc))
            if jac > 0.0:
                var_logit = 0.0

                def y_of(x: float) -> float:
                    return float(_logit(np.asarray([float(np.clip(x, x_left, 1.0 - x_left))], dtype=float))[0])

                if z50_std > 0:
                    dz = float(min(z50_std, 0.05))
                    yp = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu + dz,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    ym = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu - dz,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    d = (yp - ym) / (2.0 * dz)
                    var_logit += float((d * d) * (z50_std * z50_std))
                if log_dzl_std > 0:
                    dl = float(min(log_dzl_std, 0.02))
                    yp = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu + dl,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    ym = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu - dl,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    d = (yp - ym) / (2.0 * dl)
                    var_logit += float((d * d) * (log_dzl_std * log_dzl_std))
                if log_dzr_std > 0:
                    dr = float(min(log_dzr_std, 0.02))
                    yp = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu + dr,
                            )[0]
                        )
                    )
                    ym = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu - dr,
                            )[0]
                        )
                    )
                    d = (yp - ym) / (2.0 * dr)
                    var_logit += float((d * d) * (log_dzr_std * log_dzr_std))
                if z05_std > 0 and math.isfinite(z05_mu):
                    dz = float(min(z05_std, 0.05))
                    yp = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu + dz,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    ym = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu - dz,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    d = (yp - ym) / (2.0 * dz)
                    var_logit += float((d * d) * (z05_std * z05_std))
                if z25_std > 0 and math.isfinite(z25_mu):
                    dz = float(min(z25_std, 0.05))
                    yp = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu + dz,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    ym = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu - dz,
                                z50=z50_mu,
                                z75=z75_mu,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    d = (yp - ym) / (2.0 * dz)
                    var_logit += float((d * d) * (z25_std * z25_std))
                if z75_std > 0 and math.isfinite(z75_mu):
                    dz = float(min(z75_std, 0.05))
                    yp = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu + dz,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    ym = y_of(
                        float(
                            self._mean_curve(
                                z_arr,
                                z05=z05_mu,
                                z25=z25_mu,
                                z50=z50_mu,
                                z75=z75_mu - dz,
                                log_dzl=log_dzl_mu,
                                log_dzr=log_dzr_mu,
                            )[0]
                        )
                    )
                    d = (yp - ym) / (2.0 * dz)
                    var_logit += float((d * d) * (z75_std * z75_std))

                var = float(min(var, (jac * jac) * var_logit))

        std = float(math.sqrt(max(0.0, var)))
        mean = float(min(1.0, max(0.0, mean)))
        floor = float(self._sigma_floor(np.asarray([float(z)], dtype=float), mean=np.asarray([mean], dtype=float))[0])
        std = float(math.sqrt(std * std + floor * floor))
        mean = float(min(1.0, max(0.0, mean)))
        return mean, std

    def predict_many(self, params: Sequence[float], z_values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        z_arr = np.asarray(list(z_values), dtype=float)
        z50_mu, z50_std, log_dzl_mu, log_dzl_std, log_dzr_mu, log_dzr_std = self.predict_shape(params=params)
        z05_mu, z05_std = self.predict_z05(params=params)
        z25_mu, z25_std = self.predict_z25(params=params)
        z75_mu, z75_std = self.predict_z75(params=params)

        mean = self._mean_curve(
            z_arr,
            z05=z05_mu,
            z25=z25_mu,
            z50=z50_mu,
            z75=z75_mu,
            log_dzl=log_dzl_mu,
            log_dzr=log_dzr_mu,
        )

        def grad_for(delta: float, plus_kwargs: dict[str, float], minus_kwargs: dict[str, float]) -> np.ndarray:
            dlt = float(delta)
            if not (math.isfinite(dlt) and dlt > 0.0):
                return np.zeros_like(mean, dtype=float)
            mp = self._mean_curve(z_arr, **plus_kwargs)
            mm = self._mean_curve(z_arr, **minus_kwargs)
            return (mp - mm) / (2.0 * dlt)

        caps = {
            "z": 0.05,
            "log": 0.02,
        }

        g0 = grad_for(
            min(z50_std, caps["z"]),
            dict(z05=z05_mu, z25=z25_mu, z50=z50_mu + min(z50_std, caps["z"]), z75=z75_mu, log_dzl=log_dzl_mu, log_dzr=log_dzr_mu),
            dict(z05=z05_mu, z25=z25_mu, z50=z50_mu - min(z50_std, caps["z"]), z75=z75_mu, log_dzl=log_dzl_mu, log_dzr=log_dzr_mu),
        )
        g1 = grad_for(
            min(log_dzl_std, caps["log"]),
            dict(z05=z05_mu, z25=z25_mu, z50=z50_mu, z75=z75_mu, log_dzl=log_dzl_mu + min(log_dzl_std, caps["log"]), log_dzr=log_dzr_mu),
            dict(z05=z05_mu, z25=z25_mu, z50=z50_mu, z75=z75_mu, log_dzl=log_dzl_mu - min(log_dzl_std, caps["log"]), log_dzr=log_dzr_mu),
        )
        g2 = grad_for(
            min(log_dzr_std, caps["log"]),
            dict(z05=z05_mu, z25=z25_mu, z50=z50_mu, z75=z75_mu, log_dzl=log_dzl_mu, log_dzr=log_dzr_mu + min(log_dzr_std, caps["log"])),
            dict(z05=z05_mu, z25=z25_mu, z50=z50_mu, z75=z75_mu, log_dzl=log_dzl_mu, log_dzr=log_dzr_mu - min(log_dzr_std, caps["log"])),
        )
        g3 = grad_for(
            min(z05_std, caps["z"]) if math.isfinite(z05_mu) else 0.0,
            dict(z05=z05_mu + min(z05_std, caps["z"]), z25=z25_mu, z50=z50_mu, z75=z75_mu, log_dzl=log_dzl_mu, log_dzr=log_dzr_mu),
            dict(z05=z05_mu - min(z05_std, caps["z"]), z25=z25_mu, z50=z50_mu, z75=z75_mu, log_dzl=log_dzl_mu, log_dzr=log_dzr_mu),
        )
        g4 = grad_for(
            min(z25_std, caps["z"]) if math.isfinite(z25_mu) else 0.0,
            dict(z05=z05_mu, z25=z25_mu + min(z25_std, caps["z"]), z50=z50_mu, z75=z75_mu, log_dzl=log_dzl_mu, log_dzr=log_dzr_mu),
            dict(z05=z05_mu, z25=z25_mu - min(z25_std, caps["z"]), z50=z50_mu, z75=z75_mu, log_dzl=log_dzl_mu, log_dzr=log_dzr_mu),
        )
        g5 = grad_for(
            min(z75_std, caps["z"]) if math.isfinite(z75_mu) else 0.0,
            dict(z05=z05_mu, z25=z25_mu, z50=z50_mu, z75=z75_mu + min(z75_std, caps["z"]), log_dzl=log_dzl_mu, log_dzr=log_dzr_mu),
            dict(z05=z05_mu, z25=z25_mu, z50=z50_mu, z75=z75_mu - min(z75_std, caps["z"]), log_dzl=log_dzl_mu, log_dzr=log_dzr_mu),
        )

        G = np.vstack([g0, g1, g2, g3, g4, g5]).astype(float)
        stds = np.asarray([z50_std, log_dzl_std, log_dzr_std, z05_std, z25_std, z75_std], dtype=float).reshape(6, 1)
        corr = getattr(self, "corr_shape", None)
        if corr is None:
            C = np.eye(6, dtype=float)
        else:
            C = np.asarray(corr, dtype=float)
            if C.shape != (6, 6):
                C = np.eye(6, dtype=float)
        W = G * stds
        var = np.einsum("in,ij,jn->n", W, C, W).astype(float)

        mean = np.clip(mean, 0.0, 1.0)
        std = np.sqrt(np.maximum(var, 0.0))
        floor = self._sigma_floor(z_arr, mean=mean)
        std = np.sqrt(std * std + floor * floor)
        z_hi = float(getattr(self, "calib_z_hi", 12.0))
        w = float(getattr(self, "calib_blend_width", 0.5))
        s_mid = float(getattr(self, "calib_scale_mid", 1.0))
        s_high = float(getattr(self, "calib_scale_high", 1.0))
        if math.isfinite(z_hi) and math.isfinite(w) and w > 0.0 and math.isfinite(s_mid) and s_mid > 0.0 and math.isfinite(s_high) and s_high > 0.0:
            t = (z_arr - (z_hi - w)) / (2.0 * w)
            t = np.clip(t, 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)
            std = std * (s_mid + (s_high - s_mid) * t)
        fhi = float(getattr(self, "calib_floor_high", 0.0))
        if math.isfinite(fhi) and fhi > 0.0 and math.isfinite(z_hi) and math.isfinite(w) and w > 0.0:
            t = (z_arr - (z_hi - w)) / (2.0 * w)
            t = np.clip(t, 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)
            std = np.sqrt(std * std + (t * fhi) * (t * fhi))

        flo = float(getattr(self, "calib_floor_low", 0.0))
        edge = float(getattr(self, "calib_mean_edge", 0.05))
        bw = float(getattr(self, "calib_mean_blend", 0.02))
        if math.isfinite(flo) and flo > 0.0 and math.isfinite(edge) and edge > 0.0 and math.isfinite(bw) and bw > 0.0:
            m = np.asarray(mean, dtype=float)
            t = (edge + bw - m) / (2.0 * bw)
            t = np.clip(t, 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)
            std = np.sqrt(std * std + (t * flo) * (t * flo))
        return mean.astype(float), std.astype(float)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "XHIShapeEmulator":
        import __main__ as main_mod

        if not hasattr(main_mod, cls.__name__):
            setattr(main_mod, cls.__name__, cls)
        obj = joblib.load(str(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Not an {cls.__name__} object: {path}")
        return obj


def _cmd_train(args: argparse.Namespace) -> None:
    emu = XHIShapeEmulator.train_from_files(
        sobol_path=args.sobol,
        xhi_dir=args.xhi_dir,
        file_glob=args.file_glob,
        clip_eps=args.clip_eps,
        optimize_kernel=args.optimize_kernel,
        random_state=args.random_state,
        min_dz=args.min_dz,
        tau_frac=args.tau_frac,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emu.save(out)
    print(f"saved: {out}")
    print(f"nsims: {emu.train_params.shape[0]} params_dim: {emu.train_params.shape[1]}")


def _cmd_predict(args: argparse.Namespace) -> None:
    emu = XHIShapeEmulator.load(args.model)
    params = [float(x) for x in args.params]
    mean, std = emu.predict(params=params, z=float(args.z))
    z50_mu, z50_std, log_dzl_mu, log_dzl_std, log_dzr_mu, log_dzr_std = emu.predict_shape(params=params)
    z05_mu, z05_std = emu.predict_z05(params=params)
    z25_mu, z25_std = emu.predict_z25(params=params)
    z75_mu, z75_std = emu.predict_z75(params=params)
    print(f"xHI_mean {mean:.8e}")
    print(f"xHI_std  {std:.8e}")
    if math.isfinite(z05_mu):
        print(f"z05_mean {z05_mu:.8e}")
        print(f"z05_std  {z05_std:.8e}")
    if math.isfinite(z25_mu):
        print(f"z25_mean {z25_mu:.8e}")
        print(f"z25_std  {z25_std:.8e}")
    print(f"z50_mean {z50_mu:.8e}")
    print(f"z50_std  {z50_std:.8e}")
    if math.isfinite(z75_mu):
        print(f"z75_mean {z75_mu:.8e}")
        print(f"z75_std  {z75_std:.8e}")
    print(f"log_dzl_mean {log_dzl_mu:.8e}")
    print(f"log_dzl_std  {log_dzl_std:.8e}")
    print(f"log_dzr_mean {log_dzr_mu:.8e}")
    print(f"log_dzr_std  {log_dzr_std:.8e}")


def _cmd_predict_curve(args: argparse.Namespace) -> None:
    emu = XHIShapeEmulator.load(args.model)
    params = [float(x) for x in args.params]

    if args.z_file is not None:
        z, _ = _load_xhi_curve(Path(args.z_file))
        z_values = z
    else:
        z_values = np.linspace(float(args.z_min), float(args.z_max), int(args.nz), dtype=float)

    mean, std = emu.predict_many(params=params, z_values=z_values)
    out_lines = np.column_stack([np.asarray(z_values, dtype=float), mean, std])

    if args.out is None:
        for row in out_lines:
            print(f"{row[0]:.8e} {row[1]:.8e} {row[2]:.8e}")
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path, out_lines, fmt="%.10e")
        print(f"saved: {out_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xhi_shape_emulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--sobol", required=True)
    t.add_argument("--xhi-dir", required=True)
    t.add_argument("--file-glob", default="xH_concat_emu_*")
    t.add_argument("--out", required=True)
    t.add_argument("--clip-eps", type=float, default=1e-6)
    t.add_argument("--min-dz", type=float, default=0.05)
    t.add_argument("--tau-frac", type=float, default=0.08)
    t.add_argument("--optimize-kernel", action="store_true", default=False)
    t.add_argument("--random-state", type=int, default=0)
    t.set_defaults(func=_cmd_train)

    q = sub.add_parser("predict")
    q.add_argument("--model", required=True)
    q.add_argument("--z", required=True, type=float)
    q.add_argument("--params", required=True, nargs="+")
    q.set_defaults(func=_cmd_predict)

    c = sub.add_parser("predict-curve")
    c.add_argument("--model", required=True)
    c.add_argument("--params", required=True, nargs="+")
    c.add_argument("--z-file")
    c.add_argument("--z-min", type=float, default=5.0)
    c.add_argument("--z-max", type=float, default=35.0)
    c.add_argument("--nz", type=int, default=200)
    c.add_argument("--out")
    c.set_defaults(func=_cmd_predict_curve)

    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
