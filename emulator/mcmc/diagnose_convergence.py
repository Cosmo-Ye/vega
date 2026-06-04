from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(str(path), allow_pickle=False)
    return {k: d[k] for k in d.files}


def _as_chain_2d(payload: dict[str, np.ndarray]) -> np.ndarray:
    if "chain" not in payload:
        raise ValueError("NPZ missing 'chain'")
    chain = np.asarray(payload["chain"], dtype=float)
    if chain.ndim != 2 or chain.shape[1] < 1:
        raise ValueError(f"Bad chain shape: {chain.shape}")
    return chain


def _trim_chain(chain: np.ndarray, *, burn: int, thin: int) -> np.ndarray:
    burn = int(max(0, burn))
    thin = int(max(1, thin))
    if chain.shape[0] <= burn:
        return chain[:0, :]
    return np.asarray(chain[burn::thin, :], dtype=float)


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


def _rhat_from_chains(chains: list[np.ndarray]) -> np.ndarray:
    if len(chains) < 2:
        raise ValueError("Need at least 2 chains for multi-chain R-hat")
    dims = int(chains[0].shape[1])
    if any(c.ndim != 2 or c.shape[1] != dims for c in chains):
        raise ValueError("All chains must be 2D and have the same parameter dimension")

    n = min(int(c.shape[0]) for c in chains)
    if n < 4:
        return np.full(dims, np.nan, dtype=float)
    X = np.stack([np.asarray(c[:n, :], dtype=float) for c in chains], axis=0)
    m = int(X.shape[0])

    chain_means = np.mean(X, axis=1)
    chain_vars = np.var(X, axis=1, ddof=1)
    mean_of_means = np.mean(chain_means, axis=0)

    B = float(n) * np.sum((chain_means - mean_of_means) ** 2, axis=0) / float(m - 1)
    W = np.mean(chain_vars, axis=0)
    var_hat = ((float(n) - 1.0) / float(n)) * W + B / float(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / W)
    rhat[~np.isfinite(rhat)] = np.nan
    return np.asarray(rhat, dtype=float)


def _split_chains(chains: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for c in chains:
        n = int(c.shape[0])
        n2 = n // 2
        if n2 <= 1:
            continue
        a = np.asarray(c[:n2, :], dtype=float)
        b = np.asarray(c[n2 : 2 * n2, :], dtype=float)
        out.append(a)
        out.append(b)
    return out


def _block_chains(chain: np.ndarray, blocks: int) -> list[np.ndarray]:
    blocks = int(max(2, blocks))
    n = int(chain.shape[0])
    m = n // blocks
    if m <= 1:
        return []
    cut = blocks * m
    X = np.asarray(chain[:cut, :], dtype=float).reshape(blocks, m, -1)
    return [np.asarray(X[i, :, :], dtype=float) for i in range(blocks)]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="diagnose_convergence")
    p.add_argument("--in", dest="in_paths", nargs="+", required=True)
    p.add_argument("--burn", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--labels", nargs="*", default=["zeta_UV", "Tvir"])
    p.add_argument("--blocks", type=int, default=2)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    paths = [Path(str(p)) for p in args.in_paths]

    payloads = []
    chains = []
    for p in paths:
        payload = _load_npz(p)
        chain = _as_chain_2d(payload)
        chain = _trim_chain(chain, burn=int(args.burn), thin=int(args.thin))
        payloads.append(payload)
        chains.append(chain)

    dims = int(chains[0].shape[1])
    labels = list(args.labels)
    if len(labels) < dims:
        labels = labels + [f"p{i}" for i in range(len(labels), dims)]
    labels = labels[:dims]

    print("n_chains:", len(chains))
    for p, c in zip(paths, chains):
        n = int(c.shape[0])
        print(f"chain: {p}  n: {n}")
        if n == 0:
            payload = payloads[paths.index(p)]
            if "last_x" in payload:
                last_x = np.asarray(payload["last_x"], dtype=float).reshape(-1)
                if last_x.size >= dims:
                    print("  last_x:", last_x[:dims])
            if "last_logp" in payload:
                print("  last_logp:", float(np.asarray(payload["last_logp"]).reshape(())))
            continue
        q16 = np.quantile(c, 0.16, axis=0)
        q50 = np.quantile(c, 0.50, axis=0)
        q84 = np.quantile(c, 0.84, axis=0)
        for j in range(dims):
            rho1 = _lag1_rho(c[:, j])
            ess = _ess_from_rho1(n, rho1)
            print(f"  {labels[j]} q16/50/84: {q16[j]:.6g} {q50[j]:.6g} {q84[j]:.6g}  rho1: {rho1:.4f}  ess~: {ess:.3g}")

    nonempty = [c for c in chains if int(c.shape[0]) > 0]
    if len(nonempty) == 0:
        raise SystemExit("No non-empty chains after burn/thin.")

    if len(chains) == 1:
        blocks = _block_chains(nonempty[0], blocks=int(args.blocks))
        if len(blocks) < 2:
            print("approx_rhat (single chain): nan (too short after blocking)")
        else:
            rhat = _rhat_from_chains(blocks)
            print("approx_rhat (single chain blocks):", " ".join(f"{labels[j]}={rhat[j]:.6g}" for j in range(dims)))
    else:
        if len(nonempty) >= 2:
            rhat = _rhat_from_chains(nonempty)
            print("rhat:", " ".join(f"{labels[j]}={rhat[j]:.6g}" for j in range(dims)))
            split = _split_chains(nonempty)
            if len(split) >= 2:
                rhat_s = _rhat_from_chains(split)
                print("split_rhat:", " ".join(f"{labels[j]}={rhat_s[j]:.6g}" for j in range(dims)))
        else:
            print("rhat: nan (need >=2 non-empty chains)")


if __name__ == "__main__":
    main()
