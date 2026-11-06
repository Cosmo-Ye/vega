from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from xhi_shape_emulator import XHIShapeEmulator


def _npz_scalar(d: np.lib.npyio.NpzFile, key: str) -> object:
    v = d[key]
    if np.ndim(v) == 0:
        return v.item()
    v = np.asarray(v)
    if v.size == 1:
        return v.reshape(()).item()
    return v.reshape(-1)[0].item()


def _load_chain_npz(path: Path) -> np.ndarray:
    d = np.load(str(path), allow_pickle=False)
    if "chain" not in d.files:
        raise ValueError(f"{path} missing 'chain'")
    chain = np.asarray(d["chain"], dtype=float)
    if chain.ndim != 2 or chain.shape[1] < 2:
        raise ValueError(f"{path} has bad chain shape {chain.shape}")
    chain = np.asarray(chain[:, :2], dtype=float)
    if "tvir_sample_log10" in d.files:
        if bool(_npz_scalar(d, "tvir_sample_log10")):
            chain[:, 1] = 10.0 ** chain[:, 1]
    return chain


def _select_samples(
    chain: np.ndarray,
    *,
    burn: int,
    thin: int,
    last: int | None,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    chain = np.asarray(chain, dtype=float)
    burn = int(max(0, burn))
    thin = int(max(1, thin))
    x = chain[burn::thin, :]
    if last is not None and int(last) > 0 and x.shape[0] > int(last):
        x = x[-int(last) :, :]
    if max_samples is not None and int(max_samples) > 0 and x.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(x.shape[0], size=int(max_samples), replace=False)
        idx.sort()
        x = x[idx, :]
    return np.asarray(x, dtype=float)


def _hist_quantiles_from_counts(
    *,
    counts: np.ndarray,
    edges: np.ndarray,
    probs: Sequence[float],
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    edges = np.asarray(edges, dtype=float)
    if counts.ndim != 2:
        raise ValueError("counts must be 2D (nz, nbins)")
    if edges.ndim != 1 or edges.size != counts.shape[1] + 1:
        raise ValueError("edges must have length nbins+1")

    nz, nbins = counts.shape
    out = np.full((nz, len(list(probs))), np.nan, dtype=float)
    w = float(edges[1] - edges[0])
    if not (w > 0):
        return out

    cs = np.cumsum(counts, axis=1)
    n_tot = cs[:, -1].astype(np.int64)
    for j, p in enumerate(probs):
        p = float(p)
        if not (0.0 < p < 1.0):
            continue
        targets = (p * n_tot.astype(float)).astype(float)
        for i in range(nz):
            nt = int(n_tot[i])
            if nt <= 0:
                continue
            t = float(targets[i])
            k = int(np.searchsorted(cs[i, :], t, side="left"))
            k = min(max(k, 0), nbins - 1)
            c0 = int(cs[i, k - 1]) if k > 0 else 0
            ck = int(counts[i, k])
            if ck <= 0:
                out[i, j] = float(edges[k])
                continue
            frac = (t - float(c0)) / float(ck)
            frac = float(min(max(frac, 0.0), 1.0))
            out[i, j] = float(edges[k] + frac * w)
    return out


_EMU = None
_Z_GRID = None
_NBINS = None
_XMIN = None
_XMAX = None


def _init_worker(model_path: str, z_grid: np.ndarray, nbins: int, xmin: float, xmax: float) -> None:
    global _EMU, _Z_GRID, _NBINS, _XMIN, _XMAX
    _EMU = XHIShapeEmulator.load(str(model_path))
    _Z_GRID = np.asarray(z_grid, dtype=float)
    _NBINS = int(nbins)
    _XMIN = float(xmin)
    _XMAX = float(xmax)


def _work_hist_chunk(params_chunk: np.ndarray) -> np.ndarray:
    global _EMU, _Z_GRID, _NBINS, _XMIN, _XMAX
    emu = _EMU
    z_grid = _Z_GRID
    nbins = int(_NBINS)
    xmin = float(_XMIN)
    xmax = float(_XMAX)
    if emu is None or z_grid is None:
        raise RuntimeError("Worker not initialized")

    params_chunk = np.asarray(params_chunk, dtype=float)
    nz = int(z_grid.size)
    counts = np.zeros((nz, nbins), dtype=np.int64)
    scale = (nbins - 1) / (xmax - xmin)

    for p in params_chunk:
        mean, _std = emu.predict_many(params=[float(p[0]), float(p[1])], z_values=z_grid)
        x = np.asarray(mean, dtype=float)
        x = np.clip(x, xmin, xmax)
        idx = np.floor((x - xmin) * scale + 1e-12).astype(np.int64)
        idx = np.clip(idx, 0, nbins - 1)
        flat = (np.arange(nz, dtype=np.int64) * nbins + idx).astype(np.int64)
        np.add.at(counts.reshape(-1), flat, 1)

    return counts


def _compute_band(
    *,
    chains: list[np.ndarray],
    model_path: str,
    z_grid: np.ndarray,
    nbins: int,
    xmin: float,
    xmax: float,
    nprocs: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(chains) == 0:
        raise ValueError("No chains provided")
    params = np.concatenate([np.asarray(c, dtype=float) for c in chains], axis=0)
    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("params must have shape (N, 2)")
    if params.shape[0] == 0:
        raise ValueError("No samples after selection")

    z_grid = np.asarray(z_grid, dtype=float)
    nbins = int(nbins)
    xmin = float(xmin)
    xmax = float(xmax)
    if not (nbins >= 16):
        raise ValueError("--nbins must be >= 16")
    if not (xmax > xmin):
        raise ValueError("xmax must be > xmin")

    edges = np.linspace(xmin, xmax, nbins + 1, dtype=float)
    nz = int(z_grid.size)

    if int(nprocs) <= 1:
        _init_worker(model_path=str(model_path), z_grid=z_grid, nbins=nbins, xmin=xmin, xmax=xmax)
        counts = np.zeros((nz, nbins), dtype=np.int64)
        for start in range(0, params.shape[0], int(chunk_size)):
            sl = slice(start, min(params.shape[0], start + int(chunk_size)))
            counts += _work_hist_chunk(params[sl, :])
        qs = _hist_quantiles_from_counts(counts=counts, edges=edges, probs=[0.16, 0.5, 0.84])
        return z_grid, qs

    import multiprocessing as mp

    rng = np.random.default_rng(int(seed))
    order = np.arange(params.shape[0])
    rng.shuffle(order)
    params = params[order, :]

    chunks: list[np.ndarray] = []
    for start in range(0, params.shape[0], int(chunk_size)):
        sl = slice(start, min(params.shape[0], start + int(chunk_size)))
        chunks.append(np.asarray(params[sl, :], dtype=float))

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    with ctx.Pool(
        processes=int(nprocs),
        initializer=_init_worker,
        initargs=(str(model_path), np.asarray(z_grid, dtype=float), int(nbins), float(xmin), float(xmax)),
    ) as pool:
        counts = np.zeros((nz, nbins), dtype=np.int64)
        for c in pool.imap_unordered(_work_hist_chunk, chunks, chunksize=1):
            counts += np.asarray(c, dtype=np.int64)

    qs = _hist_quantiles_from_counts(counts=counts, edges=edges, probs=[0.16, 0.5, 0.84])
    return z_grid, qs


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xhi_band_from_chain")
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--xhi-model", default="xhi_shape_emulator.joblib")
    p.add_argument("--z-min", type=float, default=5.0)
    p.add_argument("--z-max", type=float, default=20.0)
    p.add_argument("--nz", type=int, default=120)

    p.add_argument("--burn", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--last", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--nbins", type=int, default=2000)
    p.add_argument("--xmin", type=float, default=0.0)
    p.add_argument("--xmax", type=float, default=1.0)

    p.add_argument("--nprocs", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=256)

    p.add_argument("--out-txt", default="mcmc/xhi_wholeband.txt")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    zmin = float(args.z_min)
    zmax = float(args.z_max)
    nz = int(args.nz)
    if not (nz >= 2):
        raise SystemExit("--nz must be >= 2")
    if not (zmax > zmin):
        raise SystemExit("--z-max must be > --z-min")
    z_grid = np.linspace(zmin, zmax, nz, dtype=float)

    chains: list[np.ndarray] = []
    for s in args.chains:
        p = Path(str(s))
        if not p.exists():
            raise SystemExit(f"Not found: {p}")
        raw = _load_chain_npz(p)
        sel = _select_samples(
            raw,
            burn=int(args.burn),
            thin=int(args.thin),
            last=args.last,
            max_samples=args.max_samples,
            seed=int(args.seed),
        )
        chains.append(sel)
        print(f"loaded: {p}  n_raw={raw.shape[0]}  n_used={sel.shape[0]}")

    out_txt = Path(str(args.out_txt)) if args.out_txt else None

    z, qs = _compute_band(
        chains=chains,
        model_path=str(args.xhi_model),
        z_grid=z_grid,
        nbins=int(args.nbins),
        xmin=float(args.xmin),
        xmax=float(args.xmax),
        nprocs=int(args.nprocs),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
    )

    q16 = np.asarray(qs[:, 0], dtype=float)
    q50 = np.asarray(qs[:, 1], dtype=float)
    q84 = np.asarray(qs[:, 2], dtype=float)
    if out_txt is not None:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(str(out_txt), np.column_stack([z, q16, q84, q50]), fmt="%.10e")
        print(f"saved: {out_txt}")

    if np.any(np.isfinite(q16)) and np.any(np.isfinite(q84)):
        frac = float(np.mean((q84 - q16)[np.isfinite(q84 - q16)]))
        if math.isfinite(frac):
            print(f"mean_band_width: {frac:.6g}")


if __name__ == "__main__":
    main()
