from __future__ import annotations

import argparse
import dataclasses
import math
import re
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from xhi_shape_emulator import _parse_sobol_txt, _pchip_eval, _pchip_tangents, _poly_features_2d


def _extract_index(p: Path) -> int:
    m = re.search(r"(\d+)$", p.name)
    if m is None:
        raise ValueError(f"Cannot infer index from filename: {p.name}")
    return int(m.group(1))


def _load_cross_pmxh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=float, usecols=(0, 1, 2))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Bad cross file format: {path}")
    z = arr[:, 0]
    k = arr[:, 1]
    pmxh = arr[:, 2]
    return z, k, pmxh


def _infer_grids(z_flat: np.ndarray, k_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_vals, first_idx = np.unique(z_flat, return_index=True)
    order = np.argsort(first_idx)
    z_grid = z_vals[order]

    mask0 = z_flat == z_grid[0]
    k_grid = k_flat[mask0]

    nz = len(z_grid)
    nk = len(k_grid)
    if nz * nk != len(z_flat):
        raise ValueError("Cannot infer a regular (z,k) grid from file contents.")

    for i, z in enumerate(z_grid):
        sl = slice(i * nk, (i + 1) * nk)
        if not np.all(z_flat[sl] == z):
            raise ValueError("Expected z blocks to be contiguous and ordered.")
        if np.max(np.abs(k_flat[sl] - k_grid)) > 1e-12:
            raise ValueError("k grid is not identical across redshift blocks.")

    return z_grid, k_grid


def _distance_weight(*, Xs: np.ndarray, train_Xs: np.ndarray, d0: float) -> float:
    d = float(np.sqrt(((train_Xs - Xs[0]) ** 2).sum(axis=1)).min())
    return float(1.0 / (1.0 + (d / d0) * (d / d0)))


_SHAPE_KEYS = (
    "zmin",
    "log_amp",
    "log_dL02",
    "log_dL10",
    "log_dL25",
    "log_dL50",
    "log_dL75",
    "log_dL90",
    "log_dR01",
    "log_dR02",
    "log_dR10",
    "log_dR25",
    "log_dR50",
    "log_dR75",
    "log_dR90",
)


def _pchip_eval_with_tangents(x: np.ndarray, y: np.ndarray, m: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=float)
    xq = np.asarray(xq, dtype=float)

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


def _extract_u_shape(
    z_grid: np.ndarray,
    pmxh: np.ndarray,
    *,
    min_amp: float = 1e-14,
    eps_z: float = 1e-6,
) -> tuple[float, float, list[float], list[float]] | None:
    z = np.asarray(z_grid, dtype=float)
    y = np.asarray(pmxh, dtype=float)
    if z.ndim != 1 or y.ndim != 1 or z.size != y.size or z.size < 10:
        return None

    i0 = int(np.argmin(y))
    if i0 <= 1 or i0 >= z.size - 2:
        return None

    zmin = float(z[i0])
    ymin = float(y[i0])
    if not (math.isfinite(zmin) and math.isfinite(ymin)):
        return None

    amp_ymin = float(ymin)
    w = 4
    if i0 - w >= 0 and i0 + w < z.size:
        zw = np.asarray(z[i0 - w : i0 + w + 1], dtype=float)
        yw = np.asarray(y[i0 - w : i0 + w + 1], dtype=float)
        if np.all(np.isfinite(zw)) and np.all(np.isfinite(yw)) and np.min(np.diff(zw)) > 0:
            try:
                a, b, c = (float(v) for v in np.polyfit(zw, yw, deg=2))
                if math.isfinite(a) and a > 0.0 and math.isfinite(b) and math.isfinite(c):
                    z0 = float(-b / (2.0 * a))
                    z0 = float(np.clip(z0, float(zw[0]), float(zw[-1])))
                    y0 = float(a * z0 * z0 + b * z0 + c)
                    y0 = float(max(y0, float(np.min(yw))))
                    if math.isfinite(y0):
                        amp_ymin = float(y0)
            except Exception:
                pass

    amp = float(-amp_ymin)
    if not (math.isfinite(amp) and amp > float(min_amp)):
        return None

    z_left = z[: i0 + 1]
    y_left = y[: i0 + 1]
    z_right = z[i0:]
    y_right = y[i0:]

    d_left = np.clip((-y_left) / amp, 0.0, 1.0)
    d_left = np.maximum.accumulate(d_left)
    d_left = d_left + 1e-12 * np.arange(d_left.size, dtype=float)

    def z_at_left(frac: float) -> float:
        return float(np.interp(float(frac), d_left, z_left))

    zL02 = z_at_left(0.02)
    zL10 = z_at_left(0.10)
    zL25 = z_at_left(0.25)
    zL50 = z_at_left(0.50)
    zL75 = z_at_left(0.75)
    zL90 = z_at_left(0.90)
    if not (z_left[0] + eps_z < zL02 < zL10 < zL25 < zL50 < zL75 < zL90 < zmin - eps_z):
        return None

    d_right = np.clip((-y_right) / amp, 0.0, 1.0)
    z_rev = z_right[::-1]
    d_rev = d_right[::-1]
    d_rev = np.maximum.accumulate(d_rev)
    d_rev = d_rev + 1e-12 * np.arange(d_rev.size, dtype=float)

    def z_at_right(frac: float) -> float:
        return float(np.interp(float(frac), d_rev, z_rev))

    zR01 = z_at_right(0.01)
    zR02 = z_at_right(0.02)
    zR10 = z_at_right(0.10)
    zR25 = z_at_right(0.25)
    zR50 = z_at_right(0.50)
    zR75 = z_at_right(0.75)
    zR90 = z_at_right(0.90)
    if not (zmin + eps_z < zR90 < zR75 < zR50 < zR25 < zR10 < zR02 < zR01 < z_right[-1] - eps_z):
        return None

    dL = [
        float(max(1e-8, zmin - zL02)),
        float(max(1e-8, zmin - zL10)),
        float(max(1e-8, zmin - zL25)),
        float(max(1e-8, zmin - zL50)),
        float(max(1e-8, zmin - zL75)),
        float(max(1e-8, zmin - zL90)),
    ]
    dR = [
        float(max(1e-8, zR01 - zmin)),
        float(max(1e-8, zR02 - zmin)),
        float(max(1e-8, zR10 - zmin)),
        float(max(1e-8, zR25 - zmin)),
        float(max(1e-8, zR50 - zmin)),
        float(max(1e-8, zR75 - zmin)),
        float(max(1e-8, zR90 - zmin)),
    ]

    for arr in (dL, dR):
        for j in range(1, len(arr)):
            if not (arr[j - 1] > arr[j] > 0.0):
                return None

    log_amp = float(math.log(amp))
    log_dL = [float(math.log(v)) for v in dL]
    log_dR = [float(math.log(v)) for v in dR]
    return zmin, log_amp, log_dL, log_dR


def _enforce_decreasing(arr: np.ndarray, *, eps: float, upper: float) -> np.ndarray:
    a = np.asarray(arr, dtype=float).copy()
    upper = float(upper)
    if not math.isfinite(upper):
        upper = float(np.nanmax(a))
    if not math.isfinite(upper):
        upper = 1.0
    upper = max(upper, float(eps))

    a[0] = float(np.clip(a[0], float(eps), upper))
    for i in range(1, len(a)):
        allowed_max = float(a[i - 1] - float(eps))
        a[i] = float(np.clip(a[i], float(eps), allowed_max))
    return a


def _u_depth_curve(
    z_values: np.ndarray,
    *,
    zmin: float,
    z_low: float,
    z_high: float,
    zL02: float,
    zL10: float,
    zL25: float,
    zL50: float,
    zL75: float,
    zL90: float,
    zR02: float,
    zR10: float,
    zR25: float,
    zR50: float,
    zR75: float,
    zR90: float,
    zR01: float,
) -> np.ndarray:
    z_arr = np.asarray(z_values, dtype=float)
    out = np.zeros_like(z_arr, dtype=float)

    left_mask = z_arr <= float(zmin)
    right_mask = ~left_mask

    xL = np.asarray([z_low, zL02, zL10, zL25, zL50, zL75, zL90, zmin], dtype=float)
    yL = np.asarray([0.0, 0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0], dtype=float)
    mL = _pchip_tangents(xL, yL)
    mL[0] = 0.0
    mL[-1] = 0.0

    xR = np.asarray([zmin, zR90, zR75, zR50, zR25, zR10, zR02, zR01, z_high], dtype=float)
    yR = np.asarray([1.0, 0.90, 0.75, 0.50, 0.25, 0.10, 0.02, 0.01, 0.0], dtype=float)
    mR = _pchip_tangents(xR, yR)
    mR[0] = 0.0
    mR[-1] = 0.0

    if np.any(left_mask):
        zLq = np.clip(z_arr[left_mask], xL[0], xL[-1])
        out[left_mask] = _pchip_eval_with_tangents(xL, yL, mL, zLq)
    if np.any(right_mask):
        zRq = np.clip(z_arr[right_mask], xR[0], xR[-1])
        out[right_mask] = _pchip_eval_with_tangents(xR, yR, mR, zRq)

    out[z_arr < z_low] = 0.0
    out[z_arr > z_high] = 0.0
    return np.clip(out, 0.0, 1.0)


@dataclasses.dataclass
class _KShapeModel:
    train_Xs: np.ndarray
    gps: dict[str, GaussianProcessRegressor]
    trend_coef: dict[str, np.ndarray]
    trend_degree: int = 2
    corr_shape: np.ndarray | None = None


@dataclasses.dataclass
class PMXHShapeEmulator:
    z_grid: np.ndarray
    k_grid: np.ndarray
    param_scaler: StandardScaler
    models_by_k: list[_KShapeModel]
    train_params: np.ndarray
    clip_eps: float = 1e-6
    d0: float = 0.5
    sigma_floor: np.ndarray | None = None
    calib_scale_mid: float = 1.0
    calib_scale_high: float = 1.0
    calib_floor_high: float = 0.0
    calib_z_hi: float = 12.0
    calib_z_blend: float = 0.5

    @classmethod
    def train_from_files(
        cls,
        sobol_path: str | Path,
        cross_dir: str | Path,
        file_glob: str = "cross_power_concat_emu_*",
        optimize_kernel: bool = False,
        random_state: int = 0,
        clip_eps: float = 1e-6,
        d0: float = 0.2,
        min_amp: float = 1e-14,
        calibrate: bool = True,
        calib_holdout_frac: float = 0.2,
        calib_min_holdout: int = 8,
    ) -> "PMXHShapeEmulator":
        sobol_path = Path(sobol_path)
        cross_dir = Path(cross_dir)
        params_all = _parse_sobol_txt(sobol_path)

        cross_paths = sorted(cross_dir.glob(file_glob), key=_extract_index)
        if not cross_paths:
            raise ValueError(f"No cross files found under {cross_dir} with glob {file_glob}")

        valid_pairs: list[tuple[int, Path]] = []
        for p in cross_paths:
            idx = _extract_index(p)
            if 0 <= idx < len(params_all):
                valid_pairs.append((idx, p))
        if not valid_pairs:
            raise ValueError("No usable cross curves were loaded.")

        def _fit(paths: list[Path]) -> "PMXHShapeEmulator":
            indices: list[int] = []
            z_grid: np.ndarray | None = None
            k_grid: np.ndarray | None = None
            z_ref: np.ndarray | None = None
            k_ref: np.ndarray | None = None

            per_k_valid: list[list[bool]] | None = None
            per_k_out: dict[str, list[list[float]]] | None = None
            sumsq_floor: np.ndarray | None = None
            cnt_floor: np.ndarray | None = None

            for p in paths:
                idx = _extract_index(p)
                if idx >= len(params_all):
                    continue
                z, k, pmxh = _load_cross_pmxh(p)
                if z_grid is None or k_grid is None:
                    z_grid, k_grid = _infer_grids(z, k)
                    z_ref = z
                    k_ref = k
                    nk = int(len(k_grid))
                    per_k_valid = [[] for _ in range(nk)]
                    per_k_out = {k: [[] for _ in range(nk)] for k in _SHAPE_KEYS}
                else:
                    if len(z) != len(z_ref) or np.max(np.abs(z - z_ref)) > 1e-12:
                        raise ValueError(f"Inconsistent (z,k) grid in {p}")
                    if len(k) != len(k_ref) or np.max(np.abs(k - k_ref)) > 1e-12:
                        raise ValueError(f"Inconsistent (z,k) grid in {p}")

                nz = int(len(z_grid))
                nk = int(len(k_grid))
                Y = pmxh.reshape(nz, nk)

                indices.append(idx)
                if sumsq_floor is None:
                    sumsq_floor = np.zeros((nz, nk), dtype=float)
                    cnt_floor = np.zeros((nz, nk), dtype=float)
                for ik in range(nk):
                    extracted = _extract_u_shape(z_grid, Y[:, ik], min_amp=float(min_amp))
                    if extracted is None:
                        per_k_valid[ik].append(False)
                        for key in _SHAPE_KEYS:
                            per_k_out[key][ik].append(float("nan"))
                        continue

                    zmin, log_amp, log_dL, log_dR = extracted
                    per_k_valid[ik].append(True)
                    per_k_out["zmin"][ik].append(float(zmin))
                    per_k_out["log_amp"][ik].append(float(log_amp))
                    for frac_key, v in zip(
                        ["log_dL02", "log_dL10", "log_dL25", "log_dL50", "log_dL75", "log_dL90"], log_dL
                    ):
                        per_k_out[frac_key][ik].append(float(v))
                    for frac_key, v in zip(
                        ["log_dR01", "log_dR02", "log_dR10", "log_dR25", "log_dR50", "log_dR75", "log_dR90"], log_dR
                    ):
                        per_k_out[frac_key][ik].append(float(v))

                    if sumsq_floor is not None and cnt_floor is not None:
                        z_low = float(z_grid[0])
                        z_high = float(z_grid[-1])
                        eps = 1e-4
                        zmin_c = float(np.clip(float(zmin), z_low + 10.0 * eps, z_high - 10.0 * eps))
                        amp = float(math.exp(float(log_amp)))
                        dL = np.exp(np.asarray(log_dL, dtype=float)).astype(float)
                        dR = np.exp(np.asarray(log_dR, dtype=float)).astype(float)
                        max_dL = max(eps, zmin_c - z_low - eps)
                        max_dR = max(eps, z_high - zmin_c - eps)
                        dL = np.clip(dL, eps, max_dL)
                        dR = np.clip(dR, eps, max_dR)
                        dL = _enforce_decreasing(dL, eps=eps, upper=max_dL)
                        dR = _enforce_decreasing(dR, eps=eps, upper=max_dR)
                        zL02, zL10, zL25, zL50, zL75, zL90 = (zmin_c - float(x) for x in dL)
                        zR01, zR02, zR10, zR25, zR50, zR75, zR90 = (zmin_c + float(x) for x in dR)
                        depth = _u_depth_curve(
                            np.asarray(z_grid, dtype=float),
                            zmin=zmin_c,
                            z_low=z_low,
                            z_high=z_high,
                            zL02=zL02,
                            zL10=zL10,
                            zL25=zL25,
                            zL50=zL50,
                            zL75=zL75,
                            zL90=zL90,
                            zR02=zR02,
                            zR10=zR10,
                            zR25=zR25,
                            zR50=zR50,
                            zR75=zR75,
                            zR90=zR90,
                            zR01=zR01,
                        )
                        recon = (-amp * depth).astype(float)
                        res = recon - np.asarray(Y[:, ik], dtype=float)
                        sumsq_floor[:, ik] += res * res
                        cnt_floor[:, ik] += 1.0

            if z_grid is None or k_grid is None or not indices:
                raise ValueError("No usable cross curves were loaded.")

            idx_arr = np.asarray(indices, dtype=int)
            X = params_all[idx_arr]
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * RBF(length_scale=np.ones(Xs.shape[1]), length_scale_bounds=(1e-2, 1e5))
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1.0))
            )

            models_by_k: list[_KShapeModel] = []
            nk = int(len(k_grid))
            F_poly = _poly_features_2d(Xs, degree=2)

            for ik in range(nk):
                valid = np.asarray(per_k_valid[ik], dtype=bool)
                if valid.size != X.shape[0]:
                    raise ValueError("Internal shape mismatch in training lists")
                if int(np.count_nonzero(valid)) < 12:
                    raise ValueError(f"Too few valid curves for k index {ik}: {int(np.count_nonzero(valid))}")

                Xs_k = Xs[valid]
                F_k = F_poly[valid]
                gps: dict[str, GaussianProcessRegressor] = {}
                trend_coef: dict[str, np.ndarray] = {}
                for key in _SHAPE_KEYS:
                    y = np.asarray(per_k_out[key][ik], dtype=float)[valid]
                    coef, *_ = np.linalg.lstsq(F_k, y, rcond=None)
                    resid = y - F_k @ coef
                    gp = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=0.0,
                        normalize_y=True,
                        optimizer=None if not optimize_kernel else "fmin_l_bfgs_b",
                        n_restarts_optimizer=0 if not optimize_kernel else 2,
                        random_state=random_state,
                    )
                    gp.fit(Xs_k, resid)
                    gps[key] = gp
                    trend_coef[key] = np.asarray(coef, dtype=float)

                n_keys = len(_SHAPE_KEYS)
                resid_matrix = np.zeros((int(np.sum(valid)), n_keys), dtype=float)
                for i, key in enumerate(_SHAPE_KEYS):
                    y = np.asarray(per_k_out[key][ik], dtype=float)[valid]
                    coef = trend_coef[key]
                    resid_matrix[:, i] = y - F_k @ coef
                cov = np.cov(resid_matrix, rowvar=False)
                stds = np.sqrt(np.diag(cov))
                stds[stds == 0] = 1.0
                corr = cov / np.outer(stds, stds)
                corr = np.clip(corr, -0.99, 0.99)
                np.fill_diagonal(corr, 1.0)

                models_by_k.append(
                    _KShapeModel(
                        train_Xs=np.asarray(Xs_k, dtype=float),
                        gps=gps,
                        trend_coef=trend_coef,
                        trend_degree=2,
                        corr_shape=corr,
                    )
                )

            sigma_floor = None
            if sumsq_floor is not None and cnt_floor is not None:
                denom = np.maximum(cnt_floor, 1.0)
                sigma_floor = np.sqrt(np.maximum(sumsq_floor / denom, 0.0)).astype(float)

            return cls(
                z_grid=np.asarray(z_grid, dtype=float),
                k_grid=np.asarray(k_grid, dtype=float),
                param_scaler=scaler,
                models_by_k=models_by_k,
                train_params=X,
                clip_eps=float(clip_eps),
                d0=float(d0),
                sigma_floor=sigma_floor,
            )

        emu = _fit([p for _, p in valid_pairs])

        if bool(calibrate):
            n_total = len(valid_pairs)
            frac = float(calib_holdout_frac)
            n_hold = int(round(frac * n_total))
            n_hold = max(int(calib_min_holdout), n_hold)
            n_hold = min(n_hold, n_total - int(calib_min_holdout))
            if n_hold > 0 and (n_total - n_hold) >= 12:
                rng = np.random.default_rng(int(random_state))
                perm = np.arange(n_total)
                rng.shuffle(perm)
                hold_pairs = [valid_pairs[i] for i in perm[:n_hold]]
                fit_pairs = [valid_pairs[i] for i in perm[n_hold:]]

                emu_cal = _fit([p for _, p in fit_pairs])
                z = np.asarray(emu_cal.z_grid, dtype=float)
                mid_mask = (z >= 5.0) & (z <= float(emu.calib_z_hi))
                high_mask = z > float(emu.calib_z_hi)

                err_mid: list[np.ndarray] = []
                std_mid: list[np.ndarray] = []
                err_high: list[np.ndarray] = []
                std_high: list[np.ndarray] = []

                for idx, p in hold_pairs:
                    z_flat, k_flat, pmxh = _load_cross_pmxh(p)
                    z_grid2, k_grid2 = _infer_grids(z_flat, k_flat)
                    if (
                        len(z_grid2) != len(emu_cal.z_grid)
                        or np.max(np.abs(z_grid2 - emu_cal.z_grid)) > 1e-10
                        or len(k_grid2) != len(emu_cal.k_grid)
                        or np.max(np.abs(k_grid2 - emu_cal.k_grid)) > 1e-10
                    ):
                        raise ValueError(f"Inconsistent z/k grid in holdout {p.name}")

                    pred, std = emu_cal.predict_grid(params=params_all[int(idx)])
                    true = pmxh.reshape(len(z_grid2), len(k_grid2))
                    err = np.abs(pred - true)
                    if bool(np.any(mid_mask)):
                        err_mid.append(err[mid_mask, :].ravel())
                        std_mid.append(std[mid_mask, :].ravel())
                    if bool(np.any(high_mask)):
                        err_high.append(err[high_mask, :].ravel())
                        std_high.append(std[high_mask, :].ravel())

                def solve_scale(e: np.ndarray, s: np.ndarray) -> float:
                    e = np.asarray(e, dtype=float).ravel()
                    s = np.asarray(s, dtype=float).ravel()
                    m = np.isfinite(e) & np.isfinite(s) & (s > 0.0)
                    e = e[m]
                    s = s[m]
                    if e.size < 256:
                        return 1.0
                    r = e / np.maximum(s, 1e-12)
                    q68 = float(np.quantile(r, 0.68))
                    q95 = float(np.quantile(r, 0.95)) / 2.0
                    return float(np.clip(0.5 * (q68 + q95), 0.05, 5.0))

                if err_mid:
                    emu.calib_scale_mid = solve_scale(np.concatenate(err_mid), np.concatenate(std_mid))
                if err_high:
                    emu.calib_scale_high = solve_scale(np.concatenate(err_high), np.concatenate(std_high))
                emu.calib_floor_high = 0.0

        return emu

    def _predict_shape_for_k(self, *, ik: int, params_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = self.models_by_k[int(ik)]
        Xs = self.param_scaler.transform(params_arr)
        F = _poly_features_2d(Xs, degree=int(m.trend_degree))

        w = _distance_weight(Xs=Xs, train_Xs=np.asarray(m.train_Xs, dtype=float), d0=float(self.d0))

        def pred(key: str) -> tuple[float, float]:
            gp = m.gps[key]
            coef = m.trend_coef[key]
            mu_resid, sd_resid = gp.predict(Xs, return_std=True)
            w_eff = 1.0 if key in ("zmin", "log_amp") else w
            mu = float(np.dot(F[0], coef) + w_eff * float(mu_resid[0]))
            sd = float(sd_resid[0])
            return mu, sd

        means = np.empty(len(_SHAPE_KEYS), dtype=float)
        stds = np.empty(len(_SHAPE_KEYS), dtype=float)
        for i, key in enumerate(_SHAPE_KEYS):
            mu, sd = pred(key)
            means[i] = float(mu)
            stds[i] = float(sd)
        return means, stds

    def _curve_from_shape_mu(self, *, z_arr: np.ndarray, mu: np.ndarray) -> np.ndarray:
        md = {k: float(v) for k, v in zip(_SHAPE_KEYS, mu)}
        z_low = float(self.z_grid[0])
        z_high = float(self.z_grid[-1])
        eps = 1e-4

        zmin = float(np.clip(md["zmin"], z_low + 10.0 * eps, z_high - 10.0 * eps))
        amp = float(math.exp(md["log_amp"]))

        dL = np.exp([md["log_dL02"], md["log_dL10"], md["log_dL25"], md["log_dL50"], md["log_dL75"], md["log_dL90"]]).astype(float)
        dR = np.exp([md["log_dR01"], md["log_dR02"], md["log_dR10"], md["log_dR25"], md["log_dR50"], md["log_dR75"], md["log_dR90"]]).astype(float)

        max_dL = max(eps, zmin - z_low - eps)
        max_dR = max(eps, z_high - zmin - eps)
        dL = np.clip(dL, eps, max_dL)
        dR = np.clip(dR, eps, max_dR)
        dL = _enforce_decreasing(dL, eps=eps, upper=max_dL)
        dR = _enforce_decreasing(dR, eps=eps, upper=max_dR)

        zL02, zL10, zL25, zL50, zL75, zL90 = (zmin - float(x) for x in dL)
        zR01, zR02, zR10, zR25, zR50, zR75, zR90 = (zmin + float(x) for x in dR)

        depth = _u_depth_curve(
            z_arr,
            zmin=zmin,
            z_low=z_low,
            z_high=z_high,
            zL02=zL02,
            zL10=zL10,
            zL25=zL25,
            zL50=zL50,
            zL75=zL75,
            zL90=zL90,
            zR02=zR02,
            zR10=zR10,
            zR25=zR25,
            zR50=zR50,
            zR75=zR75,
            zR90=zR90,
            zR01=zR01,
        )
        return (-amp * depth).astype(float)

    def predict_grid(self, *, params: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        params_arr = np.asarray(params, dtype=float).reshape(1, -1)
        z_arr = np.asarray(self.z_grid, dtype=float)
        nk = int(len(self.k_grid))

        mean = np.empty((z_arr.size, nk), dtype=float)
        std = np.empty((z_arr.size, nk), dtype=float)
        n_keys = len(_SHAPE_KEYS)
        deltas = np.full(n_keys, 0.02, dtype=float)

        for ik in range(nk):
            mu, sd = self._predict_shape_for_k(ik=ik, params_arr=params_arr)
            curve = self._curve_from_shape_mu(z_arr=z_arr, mu=mu)
            
            m = self.models_by_k[ik]
            corr = m.corr_shape if m.corr_shape is not None else np.eye(n_keys, dtype=float)
            grads = np.zeros((n_keys, z_arr.size), dtype=float)
            
            for j in range(n_keys):
                dj = float(deltas[j])
                mu_p = mu.copy()
                mu_m = mu.copy()
                mu_p[j] += dj
                mu_m[j] -= dj
                f_p = self._curve_from_shape_mu(z_arr=z_arr, mu=mu_p)
                f_m = self._curve_from_shape_mu(z_arr=z_arr, mu=mu_m)
                grads[j] = (f_p - f_m) / (2.0 * dj)

            cov = corr * np.outer(sd, sd)
            var = np.einsum("in,ij,jn->n", grads, cov, grads)
            curve_sd = np.sqrt(np.maximum(var, 0.0))
            
            mean[:, ik] = curve
            std[:, ik] = curve_sd

        if self.sigma_floor is not None:
            sf = np.asarray(self.sigma_floor, dtype=float)
            if sf.shape == std.shape:
                std = np.sqrt(std * std + sf * sf)
                
        def smoothstep(edge0: float, edge1: float, x: float) -> float:
            x = float(np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)

        z_blend = float(self.calib_z_blend)
        z_hi = float(self.calib_z_hi)

        for i in range(z_arr.size):
            zi = float(z_arr[i])
            w_mid = 1.0 - smoothstep(z_hi - z_blend, z_hi + z_blend, zi)
            w_high = smoothstep(z_hi - z_blend, z_hi + z_blend, zi)
            
            scale = w_mid * self.calib_scale_mid + w_high * self.calib_scale_high
            std[i, :] *= scale
            
            if w_high > 0 and self.calib_floor_high > 0:
                fl_high = w_high * self.calib_floor_high
                std[i, :] = np.sqrt(std[i, :]**2 + fl_high**2)

        return mean, std

    def predict_many(self, *, params: Sequence[float], z_values: Sequence[float], k: float) -> tuple[np.ndarray, np.ndarray]:
        params_arr = np.asarray(params, dtype=float).reshape(1, -1)
        z_arr = np.asarray(z_values, dtype=float)
        xk = np.log10(np.asarray(self.k_grid, dtype=float))
        xq = float(np.log10(float(k)))
        xq = float(np.clip(xq, float(xk[0]), float(xk[-1])))

        nk = int(len(self.k_grid))
        means_k = np.empty((nk, z_arr.size), dtype=float)
        stds_k = np.empty((nk, z_arr.size), dtype=float)

        for ik in range(nk):
            mu, sd = self._predict_shape_for_k(ik=ik, params_arr=params_arr)
            base = self._curve_from_shape_mu(z_arr=z_arr, mu=mu)

            m = self.models_by_k[ik]
            if m.corr_shape is not None:
                corr = m.corr_shape
            else:
                corr = np.eye(len(_SHAPE_KEYS), dtype=float)

            n_keys = len(_SHAPE_KEYS)
            grads = np.zeros((n_keys, z_arr.size), dtype=float)
            deltas = np.full(n_keys, 0.02, dtype=float)
            for j in range(n_keys):
                dj = float(deltas[j])
                mu_p = mu.copy()
                mu_m = mu.copy()
                mu_p[j] += dj
                mu_m[j] -= dj
                f_p = self._curve_from_shape_mu(z_arr=z_arr, mu=mu_p)
                f_m = self._curve_from_shape_mu(z_arr=z_arr, mu=mu_m)
                grads[j] = (f_p - f_m) / (2.0 * dj)

            cov = corr * np.outer(sd, sd)
            var = np.einsum("in,ij,jn->n", grads, cov, grads)

            means_k[ik] = base
            stds_k[ik] = np.sqrt(np.maximum(var, 0.0))

        mean = np.empty(z_arr.size, dtype=float)
        std = np.empty(z_arr.size, dtype=float)
        floor = np.zeros_like(std, dtype=float)
        for i in range(z_arr.size):
            mean[i] = float(_pchip_eval(xk, means_k[:, i], np.asarray([xq]))[0])
            std[i] = float(abs(_pchip_eval(xk, stds_k[:, i], np.asarray([xq]))[0]))
            if self.sigma_floor is not None:
                sf = np.asarray(self.sigma_floor, dtype=float)
                if sf.ndim == 2 and sf.shape[1] == nk and sf.shape[0] == self.z_grid.size:
                    sf_z = np.asarray([float(np.interp(float(z_arr[i]), self.z_grid, sf[:, ik])) for ik in range(nk)], dtype=float)
                    floor[i] = float(abs(_pchip_eval(xk, sf_z, np.asarray([xq]))[0]))
        if np.any(floor > 0):
            std = np.sqrt(std * std + floor * floor)
            
        def smoothstep(edge0: float, edge1: float, x: float) -> float:
            x = float(np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)

        z_blend = float(self.calib_z_blend)
        z_hi = float(self.calib_z_hi)

        for i in range(z_arr.size):
            zi = float(z_arr[i])
            w_mid = 1.0 - smoothstep(z_hi - z_blend, z_hi + z_blend, zi)
            w_high = smoothstep(z_hi - z_blend, z_hi + z_blend, zi)
            
            scale = w_mid * self.calib_scale_mid + w_high * self.calib_scale_high
            std[i] *= scale
            
            if w_high > 0 and self.calib_floor_high > 0:
                fl_high = w_high * self.calib_floor_high
                std[i] = float(math.sqrt(std[i]**2 + fl_high**2))

        return mean, std

    def predict(self, *, params: Sequence[float], z: float, k: float) -> tuple[float, float]:
        mean, std = self.predict_many(params=params, z_values=[float(z)], k=float(k))
        return float(mean[0]), float(std[0])

    def save(self, path: str | Path) -> None:
        payload = {
            "format": "PMXHShapeEmulator:v5",
            "z_grid": np.asarray(self.z_grid, dtype=float),
            "k_grid": np.asarray(self.k_grid, dtype=float),
            "param_scaler": self.param_scaler,
            "train_params": np.asarray(self.train_params, dtype=float),
            "clip_eps": float(self.clip_eps),
            "d0": float(self.d0),
            "calib_scale_mid": float(self.calib_scale_mid),
            "calib_scale_high": float(self.calib_scale_high),
            "calib_floor_high": float(self.calib_floor_high),
            "calib_z_hi": float(self.calib_z_hi),
            "calib_z_blend": float(self.calib_z_blend),
            "sigma_floor": None if self.sigma_floor is None else np.asarray(self.sigma_floor, dtype=float),
            "models_by_k": [
                {
                    "train_Xs": np.asarray(m.train_Xs, dtype=float),
                    "gps": m.gps,
                    "trend_coef": {k: np.asarray(v, dtype=float) for k, v in m.trend_coef.items()},
                    "trend_degree": int(m.trend_degree),
                    "corr_shape": None if m.corr_shape is None else np.asarray(m.corr_shape, dtype=float),
                }
                for m in self.models_by_k
            ],
        }
        joblib.dump(payload, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "PMXHShapeEmulator":
        obj = joblib.load(str(path))
        if isinstance(obj, cls):
            return obj
        if not isinstance(obj, dict):
            raise TypeError(f"Not an {cls.__name__} payload: {path}")
        fmt = obj.get("format")
        if fmt == "PMXHShapeEmulator:v1":
            raise TypeError(f"Deprecated PMXHShapeEmulator:v1 model: {path}. Retrain with the current pmxh_shape_emulator.py")
        if fmt == "PMXHShapeEmulator:v2":
            raise TypeError(f"Deprecated PMXHShapeEmulator:v2 model: {path}. Retrain with the current pmxh_shape_emulator.py")
        if fmt == "PMXHShapeEmulator:v3":
            raise TypeError(f"Deprecated PMXHShapeEmulator:v3 model: {path}. Retrain with the current pmxh_shape_emulator.py")
        if fmt == "PMXHShapeEmulator:v4":
            raise TypeError(f"Deprecated PMXHShapeEmulator:v4 model: {path}. Retrain with the current pmxh_shape_emulator.py")
        if fmt != "PMXHShapeEmulator:v5":
            raise TypeError(f"Not an {cls.__name__} payload: {path}")
        models: list[_KShapeModel] = []
        for md in obj["models_by_k"]:
            models.append(
                _KShapeModel(
                    train_Xs=np.asarray(md["train_Xs"], dtype=float),
                    gps=dict(md["gps"]),
                    trend_coef={k: np.asarray(v, dtype=float) for k, v in dict(md["trend_coef"]).items()},
                    trend_degree=int(md.get("trend_degree", 2)),
                    corr_shape=md.get("corr_shape"),
                )
            )
        return cls(
            z_grid=np.asarray(obj["z_grid"], dtype=float),
            k_grid=np.asarray(obj["k_grid"], dtype=float),
            param_scaler=obj["param_scaler"],
            models_by_k=models,
            train_params=np.asarray(obj["train_params"], dtype=float),
            clip_eps=float(obj["clip_eps"]),
            d0=float(obj["d0"]),
            sigma_floor=None if obj.get("sigma_floor") is None else np.asarray(obj.get("sigma_floor"), dtype=float),
            calib_scale_mid=float(obj.get("calib_scale_mid", 1.0)),
            calib_scale_high=float(obj.get("calib_scale_high", 1.0)),
            calib_floor_high=float(obj.get("calib_floor_high", 0.0)),
            calib_z_hi=float(obj.get("calib_z_hi", 12.0)),
            calib_z_blend=float(obj.get("calib_z_blend", 0.5)),
        )


def _cmd_train(args: argparse.Namespace) -> None:
    emu = PMXHShapeEmulator.train_from_files(
        sobol_path=args.sobol,
        cross_dir=args.cross_dir,
        file_glob=args.file_glob,
        optimize_kernel=args.optimize_kernel,
        random_state=args.random_state,
        clip_eps=args.clip_eps,
        d0=args.d0,
        min_amp=args.min_amp,
        calibrate=not bool(args.no_calibrate),
        calib_holdout_frac=float(args.calib_holdout_frac),
        calib_min_holdout=int(args.calib_min_holdout),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    emu.save(out)
    print(f"saved: {out}")
    print(f"nsims: {emu.train_params.shape[0]} params_dim: {emu.train_params.shape[1]}")
    print(f"nz: {len(emu.z_grid)} nk: {len(emu.k_grid)}")


def _cmd_predict(args: argparse.Namespace) -> None:
    emu = PMXHShapeEmulator.load(args.model)
    params = [float(x) for x in args.params]
    mean, std = emu.predict(params=params, z=float(args.z), k=float(args.k))
    print(f"pmxh_mean {mean:.8e}")
    print(f"pmxh_std  {std:.8e}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pmxh_shape_emulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--sobol", required=True)
    t.add_argument("--cross-dir", required=True)
    t.add_argument("--file-glob", default="cross_power_concat_emu_*")
    t.add_argument("--out", required=True)
    t.add_argument("--optimize-kernel", action="store_true", default=False)
    t.add_argument("--random-state", type=int, default=0)
    t.add_argument("--clip-eps", type=float, default=1e-6)
    t.add_argument("--d0", type=float, default=0.5)
    t.add_argument("--min-amp", type=float, default=1e-14)
    t.add_argument("--no-calibrate", action="store_true", default=False)
    t.add_argument("--calib-holdout-frac", type=float, default=0.2)
    t.add_argument("--calib-min-holdout", type=int, default=8)
    t.set_defaults(func=_cmd_train)

    q = sub.add_parser("predict")
    q.add_argument("--model", required=True)
    q.add_argument("--z", required=True, type=float)
    q.add_argument("--k", required=True, type=float)
    q.add_argument("--params", required=True, nargs="+")
    q.set_defaults(func=_cmd_predict)

    return p


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
