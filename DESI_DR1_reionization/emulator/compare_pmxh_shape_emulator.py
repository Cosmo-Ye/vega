from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from pmxh_shape_emulator import PMXHShapeEmulator
from xhi_shape_emulator import _parse_sobol_txt, _pchip_eval


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


def _iter_test_files(*, test_file: str | None, test_dir: str | None, test_glob: str) -> list[Path]:
    if test_file is not None:
        p = Path(test_file)
        if not p.exists():
            raise FileNotFoundError(f"Test file not found: {p}")
        return [p]

    if test_dir is None:
        raise ValueError("Provide either --test-file or --test-dir")
    d = Path(test_dir)
    if not d.exists():
        raise FileNotFoundError(f"Test dir not found: {d}")
    files = sorted((p for p in d.glob(test_glob) if p.is_file()), key=lambda p: _extract_index_from_name(p.name))
    if not files:
        raise FileNotFoundError(f"No test files matched: dir={d}, glob={test_glob}")
    return files


def _truth_curve_at_k(*, z_grid: np.ndarray, k_grid: np.ndarray, pmxh_grid: np.ndarray, k: float) -> np.ndarray:
    k = float(k)
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"Invalid k: {k}")

    idx = int(np.argmin(np.abs(k_grid - k)))
    if float(abs(k_grid[idx] - k)) <= 1e-12:
        return np.asarray(pmxh_grid[:, idx], dtype=float)

    xk = np.log10(np.asarray(k_grid, dtype=float))
    xq = float(np.log10(k))
    xq = float(np.clip(xq, float(xk[0]), float(xk[-1])))
    out = np.empty(len(z_grid), dtype=float)
    for iz in range(len(z_grid)):
        out[iz] = float(_pchip_eval(xk, pmxh_grid[iz], np.asarray([xq]))[0])
    return out


def _plot_one(
    *,
    emu: PMXHShapeEmulator,
    sobol_params: np.ndarray,
    test_path: Path,
    k: float,
    out_dir: Path,
    out: str | None,
    title: str | None,
    show_std: bool,
) -> Path:
    idx = _extract_index_from_name(test_path.name)
    if idx >= len(sobol_params):
        raise IndexError(f"Test index {idx} is out of range for sobol rows {len(sobol_params)} (file {test_path.name})")
    params = sobol_params[idx]

    z_flat, k_flat, pmxh_flat = _load_cross_pmxh(test_path)
    z_grid, k_grid = _infer_grids(z_flat, k_flat)
    pmxh_grid = pmxh_flat.reshape(len(z_grid), len(k_grid))

    truth = _truth_curve_at_k(z_grid=z_grid, k_grid=k_grid, pmxh_grid=pmxh_grid, k=float(k))
    mean, std = emu.predict_many(params=params, z_values=z_grid, k=float(k))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.6, 4.2))
    plt.plot(z_grid, truth, lw=2.0, label=f"truth (idx={idx})")
    # plt.plot(z_grid, mean, lw=2.0, label="prediction")
    # if show_std:
        # plt.fill_between(z_grid, mean - std, mean + std, alpha=0.25, label="prediction ±1σ")
    plt.xlabel("z")
    plt.ylabel("pmxh")
    if title is None:
        plt.title(f"pmxh(z, k={float(k):.6g}): truth vs prediction ({test_path.name})")
    else:
        plt.title(str(title))
    plt.legend()
    plt.tight_layout()

    default_name = f"{test_path.name}_k{float(k):.6g}_pmxh_compare.png"
    out_path = _resolve_out_path(out_dir=out_dir, out_arg=out, default_name=default_name)
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="compare_pmxh_shape_emulator")
    p.add_argument("--sobol", default="sobol_inner.txt")
    p.add_argument("--model", default="pmxh_shape_emulator.joblib")
    p.add_argument("--k", required=True, type=float)
    p.add_argument("--test-file", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--test-glob", default="cross_power_concat_emu_*")
    p.add_argument("--out-dir", default="test_data/compare_cross")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--no-std", action="store_true", default=False)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    emu = PMXHShapeEmulator.load(model_path)

    sobol_path = Path(args.sobol)
    if not sobol_path.exists():
        raise FileNotFoundError(f"sobol not found: {sobol_path}")
    sobol_params = _parse_sobol_txt(sobol_path)

    files = _iter_test_files(test_file=args.test_file, test_dir=args.test_dir, test_glob=str(args.test_glob))
    out_dir = Path(args.out_dir)

    saved: list[Path] = []
    for p in files:
        saved.append(
            _plot_one(
                emu=emu,
                sobol_params=sobol_params,
                test_path=p,
                k=float(args.k),
                out_dir=out_dir,
                out=args.out if len(files) == 1 else None,
                title=args.title,
                show_std=not bool(args.no_std),
            )
        )

    for p in saved:
        print(f"saved {p}")


if __name__ == "__main__":
    main()

