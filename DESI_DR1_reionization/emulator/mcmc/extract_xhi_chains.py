from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Sequence

import numpy as np

from xhi_shape_emulator import XHIShapeEmulator


def _load_chain_npz(path: Path) -> np.ndarray:
    d = np.load(str(path), allow_pickle=False)
    if "chain" not in d.files:
        raise ValueError(f"{path} missing 'chain'")
    chain = np.asarray(d["chain"], dtype=float)
    if chain.ndim != 2 or chain.shape[1] < 2:
        raise ValueError(f"{path} has bad chain shape {chain.shape}")
    return chain[:, :2]


def _load_band(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(str(path), dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected >=3 columns in {path} (z, q16, q84), got {data.shape}")
    z = np.asarray(data[:, 0], dtype=float)
    q16 = np.asarray(data[:, 1], dtype=float)
    q84 = np.asarray(data[:, 2], dtype=float)
    center = 0.5 * (q16 + q84)
    order = np.argsort(z)
    return z[order], center[order]


def _invert_center_to_z(*, z: np.ndarray, center: np.ndarray, targets: Sequence[float]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    center = np.asarray(center, dtype=float)
    if z.ndim != 1 or center.ndim != 1 or z.size != center.size:
        raise ValueError("z/center must be 1D arrays with the same length")
    if z.size < 2:
        raise ValueError("Need at least 2 points to invert")
    out = np.full(len(list(targets)), np.nan, dtype=float)
    for j, t in enumerate(targets):
        t = float(t)
        found = False
        for i in range(z.size - 1):
            a = float(center[i])
            b = float(center[i + 1])
            if (t - a) * (t - b) > 0:
                continue
            if a == b:
                out[j] = float(z[i])
                found = True
                break
            frac = (t - a) / (b - a)
            frac = float(min(max(frac, 0.0), 1.0))
            out[j] = float(z[i] + frac * (z[i + 1] - z[i]))
            found = True
            break
        if not found:
            out[j] = float("nan")
    return out


_EMU = None
_Z_TARGETS = None


def _init_worker(model_path: str, z_targets: np.ndarray) -> None:
    global _EMU, _Z_TARGETS
    _EMU = XHIShapeEmulator.load(str(model_path))
    _Z_TARGETS = np.asarray(z_targets, dtype=float)


def _work_chunk(params_chunk: np.ndarray) -> np.ndarray:
    global _EMU, _Z_TARGETS
    emu = _EMU
    z_targets = _Z_TARGETS
    if emu is None or z_targets is None:
        raise RuntimeError("Worker not initialized")
    params_chunk = np.asarray(params_chunk, dtype=float)
    out = np.empty((params_chunk.shape[0], z_targets.size), dtype=float)
    for i, p in enumerate(params_chunk):
        mean, _std = emu.predict_many(params=[float(p[0]), float(p[1])], z_values=z_targets)
        out[i, :] = np.asarray(mean, dtype=float)
    return out


def _write_column(*, f, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=float).reshape(-1)
    np.savetxt(f, x, fmt="%.10e")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="extract_xhi_chains")
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--xhi-model", default="xhi_shape_emulator.joblib")
    p.add_argument("--band", default="mcmc/xhi_band.txt")
    p.add_argument("--targets", nargs="+", type=float, default=[25, 50, 75])
    p.add_argument("--burn", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--out-dir", default="mcmc")
    p.add_argument("--prefix", default="xhi")
    p.add_argument("--nprocs", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=512)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    chains = [Path(str(p)) for p in args.chains]
    for p in chains:
        if not p.exists():
            raise SystemExit(f"Not found: {p}")

    z_band, center = _load_band(Path(str(args.band)))
    raw_targets = [float(x) for x in args.targets]
    if len(raw_targets) == 0:
        raise SystemExit("--targets must provide at least one value")
    targets: list[float] = []
    labels: list[int] = []
    for v in raw_targets:
        x = float(v)
        if x > 1.0:
            x = x / 100.0
        if not (0.0 < x < 1.0):
            raise SystemExit(f"Bad target value {v}. Use either fractions in (0,1) or percents in (0,100).")
        targets.append(float(x))
        labels.append(int(round(100.0 * float(x))))

    z_targets = _invert_center_to_z(z=z_band, center=center, targets=targets)
    if np.any(~np.isfinite(z_targets)):
        raise SystemExit(f"Cannot invert z for targets {targets} from band {args.band}")

    burn = int(max(0, args.burn))
    thin = int(max(1, args.thin))
    params_all: list[np.ndarray] = []
    n_total = 0
    for p in chains:
        raw = _load_chain_npz(p)
        sel = np.asarray(raw[burn::thin, :], dtype=float)
        params_all.append(sel)
        n_total += int(sel.shape[0])
        print(f"loaded: {p}  n_raw={raw.shape[0]}  n_used={sel.shape[0]}")

    params = np.concatenate(params_all, axis=0)
    if params.shape[0] == 0:
        raise SystemExit("No samples after burn/thin.")

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)

    out_files = [out_dir / f"{prefix}{lab:02d}_chain.txt" for lab in labels]

    chunks: list[np.ndarray] = []
    chunk_size = int(max(1, args.chunk_size))
    for start in range(0, params.shape[0], chunk_size):
        sl = slice(start, min(params.shape[0], start + chunk_size))
        chunks.append(np.asarray(params[sl, :], dtype=float))

    nprocs = int(args.nprocs)
    if nprocs <= 1:
        _init_worker(str(args.xhi_model), z_targets)
        with contextlib.ExitStack() as stack:
            fs = [stack.enter_context(open(p, "w")) for p in out_files]
            for c in chunks:
                vals = np.asarray(_work_chunk(c), dtype=float)
                for j, f in enumerate(fs):
                    _write_column(f=f, x=vals[:, j])
        for p in out_files:
            print("saved:", p)
        print("z_targets:", z_targets)
        return

    import multiprocessing as mp

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)
    with ctx.Pool(processes=nprocs, initializer=_init_worker, initargs=(str(args.xhi_model), z_targets)) as pool:
        with contextlib.ExitStack() as stack:
            fs = [stack.enter_context(open(p, "w")) for p in out_files]
            for vals in pool.imap(_work_chunk, chunks, chunksize=1):
                vals = np.asarray(vals, dtype=float)
                for j, f in enumerate(fs):
                    _write_column(f=f, x=vals[:, j])

    for p in out_files:
        print("saved:", p)
    print("z_targets:", z_targets)


if __name__ == "__main__":
    main()
