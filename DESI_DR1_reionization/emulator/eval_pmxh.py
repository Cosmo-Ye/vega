from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pmxh_shape_emulator import PMXHShapeEmulator
from xhi_shape_emulator import _parse_sobol_txt


def _extract_index_from_name(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    if m is None:
        raise ValueError(f"Cannot infer index from filename: {name}")
    return int(m.group(1))


def _iter_files(*, d: Path, glob_pat: str) -> list[Path]:
    files = sorted((p for p in d.glob(glob_pat) if p.is_file()), key=lambda p: _extract_index_from_name(p.name))
    return files


def _load_cross_pmxh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=float, usecols=(0, 1, 2))
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Bad cross file format: {path}")
    return arr[:, 0], arr[:, 1], arr[:, 2]


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


def _weights_z(z: np.ndarray, *, z1: float, z2: float, factor: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    w = np.ones_like(z, dtype=float)
    w[(z >= float(z1)) & (z <= float(z2))] = float(factor)
    return w


def _w_rmse(err: np.ndarray, w: np.ndarray) -> float:
    err = np.asarray(err, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    if err.size == 0:
        return float("nan")
    s = float(np.sum(w))
    if not (math.isfinite(s) and s > 0.0):
        return float("nan")
    return float(math.sqrt(float(np.sum(w * err * err)) / s))


def _q99_abs(err: np.ndarray) -> float:
    err = np.asarray(err, dtype=float).ravel()
    if err.size == 0:
        return float("nan")
    return float(np.quantile(np.abs(err), 0.99))


def _coverage(err: np.ndarray, std: np.ndarray, *, n_sigma: float, min_std: float) -> float:
    err = np.asarray(err, dtype=float).ravel()
    std = np.asarray(std, dtype=float).ravel()
    if err.size == 0:
        return float("nan")
    s = np.maximum(np.abs(std), float(min_std))
    return float(np.mean(np.abs(err) <= float(n_sigma) * s))


def _nll(err: np.ndarray, std: np.ndarray, *, min_std: float) -> float:
    err = np.asarray(err, dtype=float).ravel()
    std = np.asarray(std, dtype=float).ravel()
    if err.size == 0:
        return float("nan")
    s2 = np.maximum(std * std, float(min_std) * float(min_std))
    return float(np.mean(0.5 * np.log(2.0 * math.pi * s2) + 0.5 * (err * err) / s2))


def _bin_masks_z(z: np.ndarray, *, z_lo: float, z_hi: float) -> dict[str, np.ndarray]:
    z = np.asarray(z, dtype=float).ravel()
    m2 = (z >= float(z_lo)) & (z <= float(z_hi))
    m3 = z > float(z_hi)
    return {
        "all": np.ones_like(z, dtype=bool),
        f"{z_lo:g}<=z<={z_hi:g}": m2,
        f"z>{z_hi:g}": m3,
    }


def _k_bin_edges(k_grid: np.ndarray, *, nbins: int) -> np.ndarray:
    k = np.asarray(k_grid, dtype=float)
    lk = np.log10(k)
    qs = np.linspace(0.0, 1.0, int(nbins) + 1)
    edges = np.quantile(lk, qs)
    edges[0] = float(np.min(lk))
    edges[-1] = float(np.max(lk))
    return np.asarray(10.0 ** edges, dtype=float)


def _bin_masks_k(k: np.ndarray, *, edges: np.ndarray) -> dict[str, np.ndarray]:
    k = np.asarray(k, dtype=float).ravel()
    edges = np.asarray(edges, dtype=float).ravel()
    out: dict[str, np.ndarray] = {"all": np.ones_like(k, dtype=bool)}
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            m = (k >= lo) & (k <= hi)
        else:
            m = (k >= lo) & (k < hi)
        out[f"{lo:.6g}<=k<={hi:.6g}"] = m
    return out


def _mae_vals(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x)))


def _shape_features_from_curve(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if z.size == 0:
        return float("nan"), float("nan")
    i0 = int(np.argmin(y))
    return float(z[i0]), float(y[i0])


def _eval_dataset(
    *,
    label: str,
    emu: PMXHShapeEmulator,
    sobol_params: np.ndarray,
    files: list[Path],
    z_weight_lo: float,
    z_weight_hi: float,
    z_weight_factor: float,
    z_bin_lo: float,
    z_bin_hi: float,
    k_nbins: int,
    min_std: float,
) -> list[dict[str, object]]:
    errs: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    ks: list[np.ndarray] = []

    dzmin_all: list[float] = []
    dpmin_all: list[float] = []
    k_shape: list[float] = []

    for p in files:
        idx = _extract_index_from_name(p.name)
        if idx >= len(sobol_params):
            continue
        params = sobol_params[idx]
        z_flat, k_flat, pmxh = _load_cross_pmxh(p)
        z_grid, k_grid = _infer_grids(z_flat, k_flat)

        if len(z_grid) != len(emu.z_grid) or np.max(np.abs(z_grid - emu.z_grid)) > 1e-10:
            raise ValueError(f"Inconsistent z grid between model and {p.name}")
        if len(k_grid) != len(emu.k_grid) or np.max(np.abs(k_grid - emu.k_grid)) > 1e-10:
            raise ValueError(f"Inconsistent k grid between model and {p.name}")

        pred_grid, std_grid = emu.predict_grid(params=params)
        true_grid = pmxh.reshape(len(z_grid), len(k_grid))
        err_grid = np.asarray(pred_grid, dtype=float) - np.asarray(true_grid, dtype=float)

        errs.append(err_grid.reshape(-1))
        stds.append(np.asarray(std_grid, dtype=float).reshape(-1))
        zs.append(np.repeat(np.asarray(z_grid, dtype=float), len(k_grid)))
        ks.append(np.tile(np.asarray(k_grid, dtype=float), len(z_grid)))

        for ik, kval in enumerate(np.asarray(k_grid, dtype=float)):
            zmin_t, pmin_t = _shape_features_from_curve(z_grid, true_grid[:, ik])
            zmin_p, pmin_p = _shape_features_from_curve(z_grid, np.asarray(pred_grid, dtype=float)[:, ik])
            if math.isfinite(zmin_t) and math.isfinite(zmin_p) and math.isfinite(pmin_t) and math.isfinite(pmin_p):
                dzmin_all.append(float(zmin_p - zmin_t))
                dpmin_all.append(float(pmin_p - pmin_t))
                k_shape.append(float(kval))

    if not errs:
        return []

    err_all = np.concatenate(errs)
    std_all = np.concatenate(stds)
    z_all = np.concatenate(zs)
    k_all = np.concatenate(ks)

    edges = _k_bin_edges(emu.k_grid, nbins=int(k_nbins))
    zmasks = _bin_masks_z(z_all, z_lo=z_bin_lo, z_hi=z_bin_hi)
    kmasks = _bin_masks_k(k_all, edges=edges)

    rows: list[dict[str, object]] = []
    base_row = {
        "dataset": label,
        "n_samples": int(len(files)),
    }

    def metrics_for_mask(mask: np.ndarray) -> dict[str, float]:
        e = err_all[mask]
        s = std_all[mask]
        zsel = z_all[mask]
        wsel = _weights_z(zsel, z1=z_weight_lo, z2=z_weight_hi, factor=z_weight_factor)
        return {
            "n_points": int(e.size),
            "wRMSE": _w_rmse(e, wsel),
            "Q99": _q99_abs(e),
            "Coverage68": _coverage(e, s, n_sigma=1.0, min_std=min_std),
            "Coverage95": _coverage(e, s, n_sigma=2.0, min_std=min_std),
            "NLL": _nll(e, s, min_std=min_std),
        }

    overall = dict(base_row)
    overall.update({"z_bin": "all", "k_bin": "all"})
    overall.update(metrics_for_mask(np.ones_like(err_all, dtype=bool)))
    overall.update(
        {
            "mae_zmin": _mae_vals(np.asarray(dzmin_all, dtype=float)),
            "mae_pmin": _mae_vals(np.asarray(dpmin_all, dtype=float)),
        }
    )
    rows.append(overall)

    for zbin, m in zmasks.items():
        if zbin == "all":
            continue
        r = dict(base_row)
        r.update({"z_bin": zbin, "k_bin": "all"})
        r.update(metrics_for_mask(m))
        rows.append(r)

    dzmin_arr = np.asarray(dzmin_all, dtype=float)
    dpmin_arr = np.asarray(dpmin_all, dtype=float)
    k_shape_arr = np.asarray(k_shape, dtype=float)

    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            m_points = (k_all >= lo) & (k_all <= hi)
            m_shape = (k_shape_arr >= lo) & (k_shape_arr <= hi)
        else:
            m_points = (k_all >= lo) & (k_all < hi)
            m_shape = (k_shape_arr >= lo) & (k_shape_arr < hi)
        r = dict(base_row)
        r.update({"z_bin": "all", "k_bin": f"{lo:.6g}<=k<={hi:.6g}"})
        r.update(metrics_for_mask(m_points))
        r["mae_zmin"] = _mae_vals(dzmin_arr[m_shape])
        r["mae_pmin"] = _mae_vals(dpmin_arr[m_shape])
        rows.append(r)

    return rows


def _print_tsv(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    print("\t".join(keys))
    for r in rows:
        vals: list[str] = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("nan")
                else:
                    vals.append(f"{v:.8e}")
            else:
                vals.append(str(v))
        print("\t".join(vals))

def _format_tsv(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    lines: list[str] = ["\t".join(keys)]
    for r in rows:
        vals: list[str] = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("nan")
                else:
                    vals.append(f"{v:.8e}")
            else:
                vals.append(str(v))
        lines.append("\t".join(vals))
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval_pmxh")
    p.add_argument("--model", default="pmxh_shape_emulator.joblib")
    p.add_argument("--train-dir", default="train_data/cross")
    p.add_argument("--train-glob", default="cross_power_concat_emu_*")
    p.add_argument("--train-sobol", default="sobol.txt")
    p.add_argument("--test-dir", default="test_data/cross")
    p.add_argument("--test-glob", default="cross_power_concat_emu_*")
    p.add_argument("--test-sobol", default="sobol_inner.txt")
    p.add_argument("--z-weight-lo", type=float, default=5.0)
    p.add_argument("--z-weight-hi", type=float, default=12.0)
    p.add_argument("--z-weight-factor", type=float, default=3.0)
    p.add_argument("--z-bin-lo", type=float, default=5.0)
    p.add_argument("--z-bin-hi", type=float, default=12.0)
    p.add_argument("--k-nbins", type=int, default=4)
    p.add_argument("--min-std", type=float, default=1e-8)
    p.add_argument("--out", default="eval_pmxh.tsv")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    emu = PMXHShapeEmulator.load(model_path)

    all_rows: list[dict[str, object]] = []

    train_dir = Path(args.train_dir)
    train_files = _iter_files(d=train_dir, glob_pat=str(args.train_glob)) if train_dir.exists() else []
    if train_files:
        sobol_path = Path(args.train_sobol)
        if sobol_path.exists():
            sobol_params = _parse_sobol_txt(sobol_path)
            all_rows.extend(
                _eval_dataset(
                    label="train(train_data)",
                    emu=emu,
                    sobol_params=sobol_params,
                    files=train_files,
                    z_weight_lo=float(args.z_weight_lo),
                    z_weight_hi=float(args.z_weight_hi),
                    z_weight_factor=float(args.z_weight_factor),
                    z_bin_lo=float(args.z_bin_lo),
                    z_bin_hi=float(args.z_bin_hi),
                    k_nbins=int(args.k_nbins),
                    min_std=float(args.min_std),
                )
            )

    test_dir = Path(args.test_dir)
    test_files = _iter_files(d=test_dir, glob_pat=str(args.test_glob)) if test_dir.exists() else []
    if test_files:
        sobol_path = Path(args.test_sobol)
        if sobol_path.exists():
            sobol_params = _parse_sobol_txt(sobol_path)
            all_rows.extend(
                _eval_dataset(
                    label="test(test_data)",
                    emu=emu,
                    sobol_params=sobol_params,
                    files=test_files,
                    z_weight_lo=float(args.z_weight_lo),
                    z_weight_hi=float(args.z_weight_hi),
                    z_weight_factor=float(args.z_weight_factor),
                    z_bin_lo=float(args.z_bin_lo),
                    z_bin_hi=float(args.z_bin_hi),
                    k_nbins=int(args.k_nbins),
                    min_std=float(args.min_std),
                )
            )

    if not all_rows:
        raise FileNotFoundError("No usable files found under train/test dirs")
    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_format_tsv(all_rows), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
