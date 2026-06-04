from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def _load_three_col(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(str(path), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected >=3 columns in {path} (k, pmpsi, err), got {arr.shape}")
    k = np.asarray(arr[:, 0], dtype=float)
    y = np.asarray(arr[:, 1], dtype=float)
    yerr = np.asarray(arr[:, 2], dtype=float)
    return k, y, yerr


def _plot(*, k1: np.ndarray, y1: np.ndarray, e1: np.ndarray, k2: np.ndarray, y2: np.ndarray, e2: np.ndarray, out: Path, labels: tuple[str, str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    ax.plot(k1, y1, lw=2.0, label=labels[0])
    ax.fill_between(k1, y1 - e1, y1 + e1, alpha=0.20, linewidth=0.0)
    ax.plot(k2, y2, lw=2.0, label=labels[1])
    ax.fill_between(k2, y2 - e2, y2 + e2, alpha=0.20, linewidth=0.0)
    ax.set_xscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("P_mpsi")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="compare_pmpsi_txt")
    p.add_argument("--a", default="mcmc/pmpsi_manual.txt")
    p.add_argument("--b", default="mcmc/pmpsi_manual2.txt")
    p.add_argument("--out", default="mcmc/pmpsi_manual_compare.png")
    p.add_argument("--label-a", default="pmpsi_manual")
    p.add_argument("--label-b", default="pmpsi_manual2")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    a = Path(str(args.a))
    b = Path(str(args.b))
    if not a.exists():
        raise SystemExit(f"Not found: {a}")
    if not b.exists():
        raise SystemExit(f"Not found: {b}")

    k1, y1, e1 = _load_three_col(a)
    k2, y2, e2 = _load_three_col(b)

    out = Path(str(args.out))
    _plot(k1=k1, y1=y1, e1=e1, k2=k2, y2=y2, e2=e2, out=out, labels=(str(args.label_a), str(args.label_b)))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

