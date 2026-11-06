from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from pmxh_shape_emulator import PMXHShapeEmulator
from xhi_shape_emulator import _parse_sobol_txt

style_path = os.path.join(os.path.dirname(__file__), "5par.mplstyle")
if os.path.exists(style_path):
    plt.style.use(style_path)
else:
    print(f"Warning: Style file {style_path} not found. Using default style.")


def _extract_index_from_name(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    if m is None:
        raise ValueError(f"Cannot infer index from filename: {name}")
    return int(m.group(1))


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


def _resolve_out_path(*, out_dir: Path, out_arg: str | None, default_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_arg is None:
        return out_dir / default_name

    out_arg_path = Path(out_arg)
    if out_arg_path.suffix:
        if out_arg_path.parent == Path("."):
            return out_dir / out_arg_path.name
        out_arg_path.parent.mkdir(parents=True, exist_ok=True)
        return out_arg_path

    out_arg_path.mkdir(parents=True, exist_ok=True)
    return out_arg_path / default_name


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _iter_test_files(*, test_dir: Path, test_glob: str) -> list[Path]:
    files = sorted((p for p in test_dir.glob(test_glob) if p.is_file()), key=lambda p: _extract_index_from_name(p.name))
    if not files:
        raise FileNotFoundError(f"No test files matched: dir={test_dir}, glob={test_glob}")
    return files


def _predict_grid(emu: PMXHShapeEmulator, *, params: np.ndarray) -> np.ndarray:
    pred_grid, _ = emu.predict_grid(params=params)
    return np.asarray(pred_grid, dtype=float)


def _predict_all_points(
    *,
    emu: PMXHShapeEmulator,
    sobol_params: np.ndarray,
    files: Iterable[Path],
) -> tuple[np.ndarray, np.ndarray]:
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for p in files:
        idx = _extract_index_from_name(p.name)
        if idx >= len(sobol_params):
            raise IndexError(f"Test index {idx} is out of range for sobol rows {len(sobol_params)} (file {p.name})")
        params = sobol_params[idx]

        z_flat, k_flat, pmxh = _load_cross_pmxh(p)
        z_grid, k_grid = _infer_grids(z_flat, k_flat)

        if len(z_grid) != len(emu.z_grid) or np.max(np.abs(z_grid - emu.z_grid)) > 1e-10:
            raise ValueError(f"Inconsistent z grid between model and {p.name}")
        if len(k_grid) != len(emu.k_grid) or np.max(np.abs(k_grid - emu.k_grid)) > 1e-10:
            raise ValueError(f"Inconsistent k grid between model and {p.name}")

        pred_grid = _predict_grid(emu, params=params)
        true_grid = pmxh.reshape(len(z_grid), len(k_grid))
        y_true_all.append(true_grid.reshape(-1))
        y_pred_all.append(pred_grid.reshape(-1))

    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="error_pmxh")
    p.add_argument("--model", default="pmxh_shape_emulator.joblib")
    p.add_argument("--test-dir", default="test_data/cross")
    p.add_argument("--test-glob", default="cross_power_concat_emu_*")
    p.add_argument("--test-sobol", default="sobol_inner.txt")
    p.add_argument("--out-dir", default="error_plot")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--max-points", type=int, default=0)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    emu = PMXHShapeEmulator.load(model_path)

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")
    files = _iter_test_files(test_dir=test_dir, test_glob=str(args.test_glob))

    sobol_path = Path(args.test_sobol)
    if not sobol_path.exists():
        raise FileNotFoundError(f"Test sobol not found: {sobol_path}")
    sobol_params = _parse_sobol_txt(sobol_path)

    y_true, y_pred = _predict_all_points(emu=emu, sobol_params=sobol_params, files=files)
    r2 = _r2_score(y_true, y_pred)

    if int(args.max_points) > 0 and y_true.size > int(args.max_points):
        rng = np.random.default_rng(0)
        keep = rng.choice(y_true.size, size=int(args.max_points), replace=False)
        y_true_plot = y_true[keep]
        y_pred_plot = y_pred[keep]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    plt.figure(figsize=(6.2, 6.0))
    plt.scatter(y_pred_plot, y_true_plot, c="black", s=6.0, alpha=0.25, edgecolors="none")
    plt.xlabel("Prediction", fontsize=16)
    plt.ylabel("Truth", fontsize=16)
    # if args.title is None:
    #     plt.title(f"pmxh scatter (truth vs prediction), R^2={r2:.6f}")
    # else:
    #     plt.title(str(args.title))
    plt.tight_layout()

    out = _resolve_out_path(out_dir=Path(args.out_dir), out_arg=args.out, default_name="pmxh_truth_vs_pred.png")
    plt.savefig(out, dpi=220)
    plt.close()

    print(f"n_samples {len(files)}")
    print(f"n_points {int(y_true.size)}")
    print(f"R2 {r2:.10f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
