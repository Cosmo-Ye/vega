from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from xhi_shape_emulator import XHIShapeEmulator, _parse_sobol_txt, _z_at_x


def _extract_index_from_name(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    if m is None:
        raise ValueError(f"Cannot infer index from filename: {name}")
    return int(m.group(1))


def _iter_files(*, d: Path, glob_pat: str) -> list[Path]:
    files = sorted((p for p in d.glob(glob_pat) if p.is_file()), key=lambda p: _extract_index_from_name(p.name))
    return files


def _load_xhi_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xHI file format: {path}")
    return arr[:, 0], arr[:, 1]


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


def _bin_masks(z: np.ndarray, *, z_lo: float, z_hi: float) -> dict[str, np.ndarray]:
    z = np.asarray(z, dtype=float).ravel()
    m2 = (z >= float(z_lo)) & (z <= float(z_hi))
    m3 = z > float(z_hi)
    return {
        "all": np.ones_like(z, dtype=bool),
        f"{z_lo:g}<=z<={z_hi:g}": m2,
        f"z>{z_hi:g}": m3,
    }


def _mae(a: Iterable[float]) -> float:
    x = np.asarray(list(a), dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x)))


def _finite_pairs(true_v: float, pred_v: float) -> float | None:
    if math.isfinite(true_v) and math.isfinite(pred_v):
        return float(pred_v - true_v)
    return None


def _eval_dataset(
    *,
    label: str,
    emu: XHIShapeEmulator,
    sobol_params: np.ndarray,
    files: list[Path],
    z_weight_lo: float,
    z_weight_hi: float,
    z_weight_factor: float,
    z_bin_lo: float,
    z_bin_hi: float,
    min_std: float,
) -> list[dict[str, object]]:
    errs: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    zs: list[np.ndarray] = []

    dz_z05: list[float] = []
    dz_z25: list[float] = []
    dz_z50: list[float] = []
    dz_z75: list[float] = []
    dlog_dzl: list[float] = []
    dlog_dzr: list[float] = []

    for p in files:
        idx = _extract_index_from_name(p.name)
        if idx >= len(sobol_params):
            continue
        params = sobol_params[idx]
        z, x_true = _load_xhi_file(p)
        x_pred, x_std = emu.predict_many(params=params, z_values=z)
        err = np.asarray(x_pred, dtype=float) - np.asarray(x_true, dtype=float)
        errs.append(err)
        stds.append(np.asarray(x_std, dtype=float))
        zs.append(np.asarray(z, dtype=float))

        z05_t = _z_at_x(z, x_true, 0.05)
        z25_t = _z_at_x(z, x_true, 0.25)
        z50_t = _z_at_x(z, x_true, 0.50)
        z75_t = _z_at_x(z, x_true, 0.75)
        z10_t = _z_at_x(z, x_true, 0.10)
        z90_t = _z_at_x(z, x_true, 0.90)
        if math.isfinite(z10_t) and math.isfinite(z50_t) and z10_t < z50_t:
            log_dzl_t = float(math.log(max(1e-12, z50_t - z10_t)))
        else:
            log_dzl_t = float("nan")
        if math.isfinite(z90_t) and math.isfinite(z50_t) and z50_t < z90_t:
            log_dzr_t = float(math.log(max(1e-12, z90_t - z50_t)))
        else:
            log_dzr_t = float("nan")

        z50_mu, _, log_dzl_mu, _, log_dzr_mu, _ = emu.predict_shape(params=params)
        z05_mu, _ = emu.predict_z05(params=params)
        z25_mu, _ = emu.predict_z25(params=params)
        z75_mu, _ = emu.predict_z75(params=params)

        for lst, t, m in (
            (dz_z05, z05_t, z05_mu),
            (dz_z25, z25_t, z25_mu),
            (dz_z50, z50_t, z50_mu),
            (dz_z75, z75_t, z75_mu),
            (dlog_dzl, log_dzl_t, log_dzl_mu),
            (dlog_dzr, log_dzr_t, log_dzr_mu),
        ):
            d = _finite_pairs(t, m)
            if d is not None:
                lst.append(float(d))

    if not errs:
        return []

    err_all = np.concatenate(errs)
    std_all = np.concatenate(stds)
    z_all = np.concatenate(zs)
    w_all = _weights_z(z_all, z1=z_weight_lo, z2=z_weight_hi, factor=z_weight_factor)
    masks = _bin_masks(z_all, z_lo=z_bin_lo, z_hi=z_bin_hi)

    shape_mae = {
        "mae_z05": _mae(dz_z05),
        "mae_z25": _mae(dz_z25),
        "mae_z50": _mae(dz_z50),
        "mae_z75": _mae(dz_z75),
        "mae_log_dzl": _mae(dlog_dzl),
        "mae_log_dzr": _mae(dlog_dzr),
        "n_shape": int(
            max(
                len(dz_z05),
                len(dz_z25),
                len(dz_z50),
                len(dz_z75),
                len(dlog_dzl),
                len(dlog_dzr),
            )
        ),
    }

    rows: list[dict[str, object]] = []
    for zbin, m in masks.items():
        e = err_all[m]
        s = std_all[m]
        zsel = z_all[m]
        wsel = _weights_z(zsel, z1=z_weight_lo, z2=z_weight_hi, factor=z_weight_factor)
        row: dict[str, object] = {
            "dataset": label,
            "z_bin": zbin,
            "n_samples": int(len(files)),
            "n_points": int(e.size),
            "wRMSE": _w_rmse(e, wsel),
            "Q99": _q99_abs(e),
            "Coverage68": _coverage(e, s, n_sigma=1.0, min_std=min_std),
            "Coverage95": _coverage(e, s, n_sigma=2.0, min_std=min_std),
            "NLL": _nll(e, s, min_std=min_std),
        }
        if zbin == "all":
            row.update(shape_mae)
        rows.append(row)
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
    p = argparse.ArgumentParser(prog="eval_xhi")
    p.add_argument("--model", default="xhi_shape_emulator.joblib")
    p.add_argument("--train-dir", default="train_data/xHI")
    p.add_argument("--train-glob", default="xH_concat_emu_*")
    p.add_argument("--train-sobol", default="sobol.txt")
    p.add_argument("--test-dir", default="test_data/xHI")
    p.add_argument("--test-glob", default="xH_concat_emu_*")
    p.add_argument("--test-sobol", default="sobol_inner.txt")
    p.add_argument("--z-weight-lo", type=float, default=5.0)
    p.add_argument("--z-weight-hi", type=float, default=12.0)
    p.add_argument("--z-weight-factor", type=float, default=3.0)
    p.add_argument("--z-bin-lo", type=float, default=5.0)
    p.add_argument("--z-bin-hi", type=float, default=12.0)
    p.add_argument("--min-std", type=float, default=1e-6)
    p.add_argument("--out", default="eval_xhi.tsv")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    emu = XHIShapeEmulator.load(model_path)

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
