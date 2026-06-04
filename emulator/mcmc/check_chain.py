from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def _lag1_rho(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 3:
        return float("nan")
    xc = x - float(np.mean(x))
    a = xc[:-1]
    b = xc[1:]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ess_from_rho1(n: int, rho1: float) -> float:
    if not np.isfinite(rho1) or abs(1.0 + rho1) < 1e-12:
        return float("nan")
    return float(n * (1.0 - rho1) / (1.0 + rho1))


def _autocorr_rho_fft(x: np.ndarray, *, max_lag: int | None) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(x.size)
    if n < 3 or not np.all(np.isfinite(x)):
        return np.asarray([np.nan], dtype=float)
    xc = x - float(np.mean(x))
    v = float(np.dot(xc, xc) / float(n))
    if not (v > 0):
        return np.asarray([np.nan], dtype=float)

    if max_lag is None:
        m = n - 1
    else:
        m = int(max(0, min(int(max_lag), n - 1)))

    nfft = 1 << int((2 * n - 1).bit_length())
    f = np.fft.rfft(xc, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[: m + 1].real
    denom = np.arange(n, n - (m + 1), -1, dtype=float)
    acov = acf / denom
    rho = acov / float(acov[0])
    return np.asarray(rho, dtype=float)


def _iat_geyer_from_rho(rho: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=float).reshape(-1)
    if rho.size < 3 or not np.isfinite(rho[0]):
        return float("nan")
    if rho[0] <= 0:
        return float("nan")

    max_pairs = (rho.size - 1) // 2
    if max_pairs <= 0:
        return 1.0

    gammas: list[float] = []
    prev = float("inf")
    for k in range(max_pairs):
        i1 = 2 * k + 1
        i2 = 2 * k + 2
        g = float(rho[i1] + rho[i2])
        if not np.isfinite(g) or g <= 0:
            break
        if g > prev:
            g = prev
        gammas.append(g)
        prev = g

    tau = 1.0 + 2.0 * float(np.sum(gammas)) if len(gammas) else 1.0
    return float(max(tau, 1.0))


def _ess_from_tau(n: int, tau: float) -> float:
    if not np.isfinite(tau) or tau <= 0:
        return float("nan")
    return float(n / float(tau))


def _chunk_means(x: np.ndarray, chunks: int) -> list[float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if chunks <= 0:
        return []
    m = n // chunks
    if m <= 0:
        return []
    out: list[float] = []
    for i in range(chunks):
        sl = slice(i * m, (i + 1) * m)
        if (i + 1) * m <= n:
            out.append(float(np.mean(x[sl])))
    return out


def _contour_levels_from_hist(H: np.ndarray, probs: Sequence[float]) -> list[float]:
    hf = np.asarray(H, dtype=float).ravel()
    order = np.argsort(hf)[::-1]
    s = hf[order]
    cs = np.cumsum(s)
    if cs.size == 0 or cs[-1] <= 0:
        return [float("nan") for _ in probs]
    cs = cs / cs[-1]
    levels: list[float] = []
    for p in probs:
        idx = int(np.searchsorted(cs, float(p)))
        idx = min(max(idx, 0), s.size - 1)
        levels.append(float(s[idx]))
    return levels


def _plot_contour(*, x: np.ndarray, y: np.ndarray, out: Path, xlabel: str, ylabel: str, title: str, bins: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, xe, ye = np.histogram2d(np.asarray(x, dtype=float), np.asarray(y, dtype=float), bins=int(bins), density=True)
    lvl68, lvl95 = _contour_levels_from_hist(H, probs=[0.68, 0.95])
    maxH = float(np.nanmax(H)) if np.size(H) else float("nan")
    if np.isfinite(maxH) and maxH > 0:
        cap = maxH * (1.0 - 1e-12)
        if np.isfinite(lvl68):
            lvl68 = float(min(lvl68, cap))
        if np.isfinite(lvl95):
            lvl95 = float(min(lvl95, cap))
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=170)
    if np.isfinite(lvl95) and np.isfinite(maxH) and lvl95 < maxH:
        ax.contourf(X, Y, H, levels=[lvl95, maxH], alpha=0.20)
    if np.isfinite(lvl68) and np.isfinite(maxH) and lvl68 < maxH:
        ax.contourf(X, Y, H, levels=[lvl68, maxH], alpha=0.35)
    if np.isfinite(lvl95) and np.isfinite(lvl68):
        lvls = sorted({float(lvl95), float(lvl68)})
        if len(lvls) >= 2 and lvls[0] < lvls[1]:
            ax.contour(X, Y, H, levels=lvls, linewidths=1.4)
    ax.scatter(float(np.median(x)), float(np.median(y)), marker="x", s=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))


def _plot_trace(*, x: np.ndarray, y: np.ndarray, out: Path, xlabel: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.8), dpi=150, sharex=True)
    axes[0].plot(np.asarray(x, dtype=float), lw=0.8)
    axes[0].set_ylabel(xlabel)
    axes[0].grid(alpha=0.25)
    axes[1].plot(np.asarray(y, dtype=float), lw=0.8)
    axes[1].set_ylabel(ylabel)
    axes[1].set_xlabel("sample index")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="check_chain")
    p.add_argument("--in", dest="in_paths", nargs="+", default=["mcmc/mh_zeta_tvir_chain.npz"])
    p.add_argument("--out-dir", default="mcmc")
    p.add_argument("--prefix", default=None)
    p.add_argument("--bins", type=int, default=70)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--labels", nargs=2, default=["zeta_UV", "log10Tvir"])
    p.add_argument("--no-iat", action="store_true", default=False)
    p.add_argument("--iat-max-lag", type=int, default=200000)
    p.add_argument("--no-trace", action="store_true", default=False)
    p.add_argument("--no-contour", action="store_true", default=False)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    in_paths = [Path(str(p)) for p in (args.in_paths or [])]
    if len(in_paths) == 0:
        raise SystemExit("--in requires at least 1 path")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ns: list[int] = []
    accs: list[float] = []

    print("files:")
    for p in in_paths:
        if not p.exists():
            raise SystemExit(f"Not found: {p}")
        payload = _load_npz(p)
        if "chain" not in payload:
            raise SystemExit(f"NPZ missing 'chain': {p}")
        chain = np.asarray(payload["chain"], dtype=float)
        if chain.ndim != 2 or chain.shape[1] < 2:
            raise SystemExit(f"Bad chain shape in {p}: {chain.shape}")
        n = int(chain.shape[0])
        acc_rate = float(np.asarray(payload["acc_rate"])) if "acc_rate" in payload else float("nan")
        print(f"- {p}  N_saved={n}  acc_rate={acc_rate}")
        ns.append(n)
        accs.append(acc_rate)
        if n > 0:
            xs.append(chain[:, 0])
            ys.append(chain[:, 1])
        else:
            if "last_x" in payload:
                last_x = np.asarray(payload["last_x"], dtype=float).reshape(-1)
                if last_x.size >= 2:
                    print(f"  last_x: {last_x[:2]}")
            if "last_logp" in payload:
                print(f"  last_logp: {float(np.asarray(payload['last_logp']).reshape(()))}")

    if len(xs) == 0:
        raise SystemExit("All chains are empty.")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    n = int(x.size)
    rho1_x = _lag1_rho(x)
    rho1_y = _lag1_rho(y)
    ess_x = _ess_from_rho1(n, rho1_x)
    ess_y = _ess_from_rho1(n, rho1_y)
    tau_x = float("nan")
    tau_y = float("nan")
    ess_tau_x = float("nan")
    ess_tau_y = float("nan")
    if not bool(args.no_iat):
        rho_x = _autocorr_rho_fft(x, max_lag=int(args.iat_max_lag))
        rho_y = _autocorr_rho_fft(y, max_lag=int(args.iat_max_lag))
        tau_x = _iat_geyer_from_rho(rho_x)
        tau_y = _iat_geyer_from_rho(rho_y)
        ess_tau_x = _ess_from_tau(n, tau_x)
        ess_tau_y = _ess_from_tau(n, tau_y)

    qx = np.quantile(x, [0.16, 0.5, 0.84])
    qy = np.quantile(y, [0.16, 0.5, 0.84])
    mx = _chunk_means(x, int(args.chunks))
    my = _chunk_means(y, int(args.chunks))

    print("N_saved_total:", n)
    print("q16/50/84", args.labels[0] + ":", qx)
    print("q16/50/84", args.labels[1] + ":", qy)
    print("chunk means", args.labels[0] + ":", mx)
    print("chunk means", args.labels[1] + ":", my)
    print("lag1 rho", args.labels[0] + ":", rho1_x)
    print("lag1 rho", args.labels[1] + ":", rho1_y)
    print("approx ESS", args.labels[0] + ":", ess_x)
    print("approx ESS", args.labels[1] + ":", ess_y)
    if not bool(args.no_iat):
        print("tau_int", args.labels[0] + ":", tau_x)
        print("tau_int", args.labels[1] + ":", tau_y)
        print("ESS(tau)", args.labels[0] + ":", ess_tau_x)
        print("ESS(tau)", args.labels[1] + ":", ess_tau_y)

    out_dir = Path(str(args.out_dir))
    if args.prefix:
        prefix = str(args.prefix)
    else:
        prefix = "combined" if len(in_paths) > 1 else in_paths[0].stem
    if not bool(args.no_trace):
        out_trace = out_dir / f"{prefix}_trace.png"
        _plot_trace(x=x, y=y, out=out_trace, xlabel=str(args.labels[0]), ylabel=str(args.labels[1]))
        print("saved:", out_trace)
    if not bool(args.no_contour):
        out_contour = out_dir / f"{prefix}_contour.png"
        _plot_contour(
            x=x,
            y=y,
            out=out_contour,
            xlabel=str(args.labels[0]),
            ylabel=str(args.labels[1]),
            title="Posterior Contours (68% / 95%)",
            bins=int(args.bins),
        )
        print("saved:", out_contour)


if __name__ == "__main__":
    main()
