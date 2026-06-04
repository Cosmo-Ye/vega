from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from xhi_shape_emulator import XHIShapeEmulator, _parse_sobol_txt


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


def _plot_one(
    *,
    emu: XHIShapeEmulator,
    sobol_params: np.ndarray,
    test_path: Path,
    out_dir: Path,
    out: str | None,
    title: str | None,
    show_std: bool,
) -> Path:
    idx = _extract_index_from_name(test_path.name)
    if idx >= len(sobol_params):
        raise IndexError(f"Test index {idx} is out of range for sobol rows {len(sobol_params)} (file {test_path.name})")
    params = sobol_params[idx]

    z_true, xhi_true = _load_xhi_file(test_path)
    mean, std = emu.predict_many(params=params, z_values=z_true)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(z_true, xhi_true, lw=2.0, label=f"truth (idx={idx})")
    plt.plot(z_true, mean, lw=2.0, label="prediction")
    if show_std:
        plt.fill_between(z_true, mean - std, mean + std, alpha=0.25, label="prediction ±1σ")
    plt.xlabel("z")
    plt.ylabel("xHI")
    if title is None:
        plt.title(f"xHI(z): truth vs prediction ({test_path.name})")
    else:
        plt.title(str(title))
    plt.legend()
    plt.tight_layout()

    default_name = f"{test_path.name}_shape_compare.png"
    out_path = _resolve_out_path(out_dir=out_dir, out_arg=out, default_name=default_name)
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="compare_xhi_shape_emulator")
    p.add_argument("--sobol", default="sobol.txt")
    p.add_argument("--model", default="xhi_shape_emulator.joblib")
    p.add_argument("--test-file", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--test-glob", default="xH_concat_emu_*")
    p.add_argument("--out-dir", default="compare_plot_shape")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--no-std", action="store_true", default=False)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    emu = XHIShapeEmulator.load(model_path)

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

