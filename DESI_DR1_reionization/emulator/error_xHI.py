from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from xhi_shape_emulator import XHIShapeEmulator, _parse_sobol_txt

'''python error_xHI.py \
  --model xhi_shape_emulator.joblib \
  --test-dir test_data/xHI \
  --test-glob 'xH_concat_emu_*' \
  --test-sobol sobol_inner.txt \
  --out-dir error_plot'''

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


def _load_xhi_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Bad xHI file format: {path}")
    return arr[:, 0], arr[:, 1]


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


def _predict_all_points(
    *,
    emu: XHIShapeEmulator,
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
        z, x_true = _load_xhi_file(p)
        x_pred, _ = emu.predict_many(params=params, z_values=z)
        y_true_all.append(np.asarray(x_true, dtype=float))
        y_pred_all.append(np.asarray(x_pred, dtype=float))

    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="error_xHI")
    p.add_argument("--model", default="xhi_shape_emulator.joblib")
    p.add_argument("--test-dir", default="test_data/xHI")
    p.add_argument("--test-glob", default="xH_concat_emu_*")
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
    emu = XHIShapeEmulator.load(model_path)

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
        y_true = y_true[keep]
        y_pred = y_pred[keep]

    plt.figure(figsize=(6.2, 6.0))
    plt.scatter(y_pred, y_true, s=10.0, alpha=0.25, edgecolors="none", color="black")
    plt.xlabel("Prediction", fontsize=16)
    plt.ylabel("Truth", fontsize=16)
    # if args.title is None:
    #     plt.title(f"xHI scatter (truth vs prediction), R^2={r2:.6f}")
    # else:
    #     plt.title(str(args.title))
    plt.tight_layout()

    out = _resolve_out_path(out_dir=Path(args.out_dir), out_arg=args.out, default_name="xHI_truth_vs_pred.png")
    plt.savefig(out, dpi=220)
    plt.close()

    print(f"n_samples {len(files)}")
    print(f"n_points {int(np.asarray(y_true).size)}")
    print(f"R2 {r2:.10f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
