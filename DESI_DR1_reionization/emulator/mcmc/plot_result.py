from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from xhi_shape_emulator import XHIShapeEmulator

# python mcmc/plot_result.py \
#   --kind xhi-emulator \
#   --zeta 25 --tvir 30000 \
#   --xhi-model xhi_shape_emulator.joblib \
#   --z-min 5 --z-max 12 --nz 100 \
#   --out mcmc/xhi_manual.png \
#   --out-txt mcmc/xhi_manual.txt \
#   --title "xHI(z)"

# python mcmc/plot_result.py \
#   --kind xhi-band \
#   --in mcmc/xhi_extwhole.txt \
#   --out mcmc/xhi_extwhole.png

def _load_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(str(path), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got shape {data.shape}")
    return data


def _plot_txt_x_y_band(*, data: np.ndarray, out: Path, x_label: str, y_label: str, title: str | None, ylim: tuple[float, float] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = data[:, 0]
    y = data[:, 1]
    band = data[:, 2] if data.shape[1] >= 3 else None

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    ax.plot(x, y, lw=2.0)
    if band is not None and np.all(np.isfinite(band)):
        ax.fill_between(x, y - band, y + band, alpha=0.25, linewidth=0.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x.size:
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))


def _infer_q16_q50_q84(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Expected 4 columns [z, q16, q84, q50] (or permutation), got shape {data.shape}")

    z = data[:, 0]
    c1 = data[:, 1]
    c2 = data[:, 2]
    c3 = data[:, 3]

    candidates = [
        (c1, c3, c2),
        (c1, c2, c3),
        (c2, c3, c1),
        (c2, c1, c3),
        (c3, c2, c1),
        (c3, c1, c2),
    ]

    best = None
    best_score = -1
    for q16, q50, q84 in candidates:
        m = np.isfinite(q16) & np.isfinite(q50) & np.isfinite(q84)
        if not np.any(m):
            score = -1
        else:
            score = int(np.sum((q16[m] <= q50[m]) & (q50[m] <= q84[m])))
        if score > best_score:
            best_score = score
            best = (q16, q50, q84)

    if best is None or best_score <= 0:
        raise ValueError("Could not infer q16/q50/q84 column order")
    q16, q50, q84 = best
    return np.asarray(z, dtype=float), np.asarray(q16, dtype=float), np.asarray(q50, dtype=float), np.asarray(q84, dtype=float)


def _plot_quantile_band(*, z: np.ndarray, q16: np.ndarray, q50: np.ndarray, q84: np.ndarray, out: Path, title: str | None, y_label: str, ylim: tuple[float, float] | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.asarray(z, dtype=float)
    q16 = np.asarray(q16, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q84 = np.asarray(q84, dtype=float)

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    ax.fill_between(z, q16, q84, alpha=0.25, linewidth=0.0)
    ax.plot(z, q50, lw=2.0)
    ax.set_xlabel("z")
    ax.set_ylabel(y_label)
    if z.size:
        ax.set_xlim(float(np.min(z)), float(np.max(z)))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_xlim(5.0, 12.0)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def _plot_chain_npz(*, payload: dict[str, np.ndarray], out: Path, labels: tuple[str, str], title: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "chain" not in payload:
        raise ValueError("NPZ does not contain 'chain'")
    chain = np.asarray(payload["chain"], dtype=float)
    if chain.ndim != 2 or chain.shape[1] < 2:
        raise ValueError(f"Expected chain to have shape (N,>=2), got {chain.shape}")

    x0 = chain[:, 0]
    x1 = chain[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5), dpi=150)
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    ax00.plot(x0, lw=1.0)
    ax00.set_ylabel(labels[0])
    ax00.set_xlabel("step")
    ax00.grid(True, alpha=0.3)

    ax10.plot(x1, lw=1.0)
    ax10.set_ylabel(labels[1])
    ax10.set_xlabel("step")
    ax10.grid(True, alpha=0.3)

    ax01.hist(x0, bins=40, density=True)
    ax01.set_xlabel(labels[0])
    ax01.set_ylabel("pdf")
    ax01.grid(True, alpha=0.3)

    ax11.hist(x1, bins=40, density=True)
    ax11.set_xlabel(labels[1])
    ax11.set_ylabel("pdf")
    ax11.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plot_result")
    p.add_argument("--in", dest="in_path", default=None)
    p.add_argument("--out", dest="out_path", default=None)
    p.add_argument("--kind", choices=["auto", "xhi-z", "xhi-band", "chain", "xhi-emulator"], default="auto")
    p.add_argument("--title", default=None)
    p.add_argument("--x-label", default=None)
    p.add_argument("--y-label", default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--labels", nargs=2, default=["zeta_UV", "Tvir"])
    p.add_argument("--zeta", type=float, default=None)
    p.add_argument("--tvir", type=float, default=None)
    p.add_argument("--xhi-model", default="xhi_shape_emulator.joblib")
    p.add_argument("--z-min", type=float, default=5.0)
    p.add_argument("--z-max", type=float, default=12.0)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--out-txt", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    in_path = Path(str(args.in_path)) if args.in_path else None

    kind = str(args.kind)
    if kind == "auto":
        if in_path is None:
            raise SystemExit("--in is required when --kind=auto")
        if in_path.suffix.lower() == ".npz":
            kind = "chain"
        else:
            kind = "xhi-z"

    if args.out_path:
        out_path = Path(str(args.out_path))
    else:
        if in_path is None:
            raise SystemExit("--out is required when --in is not provided")
        out_path = in_path.with_suffix(".png")

    if kind == "xhi-z":
        if in_path is None:
            raise SystemExit("--in is required for kind='xhi-z'")
        if not in_path.exists():
            raise SystemExit(f"Not found: {in_path}")
        data = _load_txt(in_path)
        x_label = args.x_label or "z"
        y_label = args.y_label or "xHI"
        ylim = None
        if args.ymin is not None or args.ymax is not None:
            ymin = -0.02 if args.ymin is None else float(args.ymin)
            ymax = 1.02 if args.ymax is None else float(args.ymax)
            ylim = (ymin, ymax)
        _plot_txt_x_y_band(data=data, out=out_path, x_label=x_label, y_label=y_label, title=args.title, ylim=ylim)
        print(f"saved: {out_path}")
        return

    if kind == "xhi-band":
        if in_path is None:
            raise SystemExit("--in is required for kind='xhi-band'")
        if not in_path.exists():
            raise SystemExit(f"Not found: {in_path}")
        data = _load_txt(in_path)
        z, q16, q50, q84 = _infer_q16_q50_q84(data)
        ylim = None
        if args.ymin is not None or args.ymax is not None:
            ymin = -0.02 if args.ymin is None else float(args.ymin)
            ymax = 1.02 if args.ymax is None else float(args.ymax)
            ylim = (ymin, ymax)
        y_label = args.y_label or "xHI"
        _plot_quantile_band(z=z, q16=q16, q50=q50, q84=q84, out=out_path, title=args.title, y_label=y_label, ylim=ylim)
        print(f"saved: {out_path}")
        return

    if kind == "chain":
        if in_path is None:
            raise SystemExit("--in is required for kind='chain'")
        if not in_path.exists():
            raise SystemExit(f"Not found: {in_path}")
        payload = _load_npz(in_path)
        _plot_chain_npz(payload=payload, out=out_path, labels=(str(args.labels[0]), str(args.labels[1])), title=args.title)
        print(f"saved: {out_path}")
        return

    if kind == "xhi-emulator":
        if args.zeta is None or args.tvir is None:
            raise SystemExit("--zeta and --tvir are required for kind='xhi-emulator'")
        emu = XHIShapeEmulator.load(str(args.xhi_model))
        zmin = float(args.z_min)
        zmax = float(args.z_max)
        nz = int(args.nz)
        if nz <= 1:
            raise SystemExit("--nz must be > 1")
        z = np.linspace(zmin, zmax, nz, dtype=float)
        mean, std = emu.predict_many(params=[float(args.zeta), float(args.tvir)], z_values=z)
        data = np.column_stack([z, np.asarray(mean, dtype=float), np.asarray(std, dtype=float)])
        if args.out_txt:
            out_txt = Path(str(args.out_txt))
            out_txt.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(str(out_txt), data, fmt="%.10e")
        x_label = "z"
        y_label = args.y_label or "xHI"
        ylim = None
        if args.ymin is not None or args.ymax is not None:
            ymin = -0.02 if args.ymin is None else float(args.ymin)
            ymax = 1.02 if args.ymax is None else float(args.ymax)
            ylim = (ymin, ymax)
        _plot_txt_x_y_band(data=data, out=out_path, x_label=x_label, y_label=y_label, title=args.title, ylim=ylim)
        print(f"saved: {out_path}")
        return

    raise SystemExit(f"Unsupported kind: {kind}")


if __name__ == "__main__":
    main()
