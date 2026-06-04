from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from mcmc.run_mcmc import pmpsi_from_pmxh
from pmxh_shape_emulator import PMXHShapeEmulator


def _load_chain_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def _accepted_indices(chain: np.ndarray) -> np.ndarray:
    chain = np.asarray(chain, dtype=float)
    if chain.ndim != 2 or chain.shape[0] < 1:
        raise ValueError("chain must be a 2D array with at least 1 row")
    if chain.shape[0] == 1:
        return np.asarray([0], dtype=int)
    moved = np.any(chain[1:] != chain[:-1], axis=1)
    idx = np.concatenate([np.asarray([0], dtype=int), np.nonzero(moved)[0] + 1])
    return idx.astype(int)


def _select_index(*, chain: np.ndarray, logp: np.ndarray | None, pick: str, rng: np.random.Generator) -> int:
    chain = np.asarray(chain, dtype=float)
    acc_idx = _accepted_indices(chain)
    if pick == "first":
        return int(acc_idx[0])
    if pick == "last":
        return int(acc_idx[-1])
    if pick == "random":
        return int(rng.choice(acc_idx))
    if pick == "best":
        if logp is None:
            raise ValueError("pick='best' requires logp in the npz file")
        logp = np.asarray(logp, dtype=float).reshape(-1)
        if logp.shape[0] != chain.shape[0]:
            raise ValueError("logp length does not match chain length")
        best_i = int(acc_idx[np.argmax(logp[acc_idx])])
        return best_i
    raise ValueError(f"Unknown pick mode: {pick}")


def _build_full_params(
    *,
    base_params: np.ndarray,
    zeta_uv: float,
    tvir: float,
    zeta_idx: int,
    tvir_idx: int,
    tvir_in_log10: bool,
) -> np.ndarray:
    p = np.asarray(base_params, dtype=float).copy()
    if p.ndim != 1:
        raise ValueError("base_params must be 1D")
    if not (0 <= int(zeta_idx) < p.size and 0 <= int(tvir_idx) < p.size):
        raise ValueError("zeta_idx/tvir_idx out of range for base_params")
    p[int(zeta_idx)] = float(zeta_uv)
    p[int(tvir_idx)] = float(np.log10(float(tvir))) if bool(tvir_in_log10) else float(tvir)
    return p


def _plot_k_curve(
    *,
    k: np.ndarray,
    models: list[tuple[str, np.ndarray, np.ndarray | None]],
    obs_k: np.ndarray | None,
    obs_y: np.ndarray | None,
    obs_yerr: np.ndarray | None,
    out: Path,
    title: str,
    y_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = np.asarray(k, dtype=float)
    models = [(label, np.asarray(y, dtype=float), None if yerr is None else np.asarray(yerr, dtype=float)) for (label, y, yerr) in models]
    obs_k = None if obs_k is None else np.asarray(obs_k, dtype=float)
    obs_y = None if obs_y is None else np.asarray(obs_y, dtype=float)
    obs_yerr = None if obs_yerr is None else np.asarray(obs_yerr, dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
    for label, y, yerr in models:
        ax.plot(k, y, lw=2.0, label=label)
        if yerr is not None and yerr.shape == y.shape and np.all(np.isfinite(yerr)):
            ax.fill_between(k, y - yerr, y + yerr, alpha=0.18, linewidth=0.0, label=f"{label} ±1σ")
    if obs_k is not None and obs_y is not None:
        if obs_yerr is not None and obs_yerr.shape == obs_y.shape and np.all(np.isfinite(obs_yerr)):
            ax.errorbar(obs_k, obs_y, yerr=obs_yerr, fmt="o", ms=4.0, capsize=2.5, elinewidth=1.0, label="obs")
        else:
            ax.plot(obs_k, obs_y, "o", ms=4.0, label="obs")
    ax.set_xscale("log")
    ax.set_xlim(0.2, 0.4)
    ax.set_ylim(-20, 0)
    ax.set_xlabel("k")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))


def _convert_spectrum(*, k: np.ndarray, y: np.ndarray, yerr: np.ndarray | None, spectrum: str) -> tuple[np.ndarray, np.ndarray | None]:
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = None if yerr is None else np.asarray(yerr, dtype=float)

    if spectrum == "delta2":
        return y, yerr
    if spectrum != "pk":
        raise ValueError(f"Unknown spectrum mode: {spectrum}")
    if np.any(k <= 0):
        raise ValueError("k must be positive to convert between P(k) and Delta^2(k)")

    factor = k**3 / (2.0 * np.pi**2)
    y2 = y / factor
    yerr2 = None if yerr is None else (yerr / np.abs(factor))
    return y2, yerr2


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inspect_pmpsi_from_chain")
    p.add_argument("--chain", default="mcmc/mh_zeta_tvir_chain.npz")
    p.add_argument("--zeta", type=float, default=None)
    p.add_argument("--tvir", type=float, default=None)
    p.add_argument("--zeta2", type=float, default=None)
    p.add_argument("--tvir2", type=float, default=None)
    p.add_argument("--emu-model", default="pmxh_shape_emulator.joblib")
    p.add_argument("--pick", choices=["last", "first", "random", "best"], default="last")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--zeta-idx", type=int, default=0)
    p.add_argument("--tvir-idx", type=int, default=1)
    p.add_argument("--tvir-in-log10", action="store_true", default=False)
    p.add_argument("--z-select", type=float, default=2.33)
    p.add_argument("--base-params", nargs="*", type=float, default=None)

    p.add_argument("--out-txt", default="mcmc/pmpsi_from_accepted.txt")
    p.add_argument("--out-plot", default="mcmc/pmpsi_from_accepted.png")
    p.add_argument("--spectrum", choices=["delta2", "pk"], default="delta2")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    rng = np.random.default_rng(int(args.seed))

    payload: dict[str, np.ndarray] = {}
    chain = None
    lp = None
    idx = None
    chain_path = Path(str(args.chain))
    if chain_path.exists():
        payload = _load_chain_npz(chain_path)
        if "chain" in payload:
            chain = np.asarray(payload["chain"], dtype=float)

    points: list[tuple[float, float]] = []
    if args.zeta is not None and args.tvir is not None:
        points.append((float(args.zeta), float(args.tvir)))
    if args.zeta2 is not None and args.tvir2 is not None:
        points.append((float(args.zeta2), float(args.tvir2)))

    if len(points) == 0:
        if chain is None:
            raise SystemExit("No --zeta/--tvir provided and --chain was not found or missing 'chain'")
        logp = payload.get("logp")
        if args.index is None:
            idx = _select_index(chain=chain, logp=logp, pick=str(args.pick), rng=rng)
        else:
            idx = int(args.index)
            if not (0 <= idx < chain.shape[0]):
                raise SystemExit(f"--index out of range: {idx} (chain has {chain.shape[0]} rows)")
        zeta_uv = float(chain[idx, 0])
        tvir = float(chain[idx, 1])
        if logp is not None:
            lp_arr = np.asarray(logp, dtype=float).reshape(-1)
            if lp_arr.shape[0] == chain.shape[0]:
                lp = float(lp_arr[idx])
        points.append((float(chain[idx, 0]), float(chain[idx, 1])))

    emu = PMXHShapeEmulator.load(str(args.emu_model))
    ndim = int(emu.train_params.shape[1])

    if args.base_params is None or len(args.base_params) == 0:
        base_params = np.median(np.asarray(emu.train_params, dtype=float), axis=0)
    else:
        base_params = np.asarray(args.base_params, dtype=float)
        if base_params.size != ndim:
            raise SystemExit(f"--base-params must have length {ndim} (emu param dimension)")

    k_grid_full = np.asarray(emu.k_grid, dtype=float)
    k_mask = (k_grid_full > 0.06) & (k_grid_full < 0.4)
    k_grid = k_grid_full[k_mask]
    spectrum = str(args.spectrum)
    models: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    for i, (zeta_uv, tvir) in enumerate(points):
        full_params = _build_full_params(
            base_params=base_params,
            zeta_uv=float(zeta_uv),
            tvir=float(tvir),
            zeta_idx=int(args.zeta_idx),
            tvir_idx=int(args.tvir_idx),
            tvir_in_log10=bool(args.tvir_in_log10),
        )
        pmxh_mean_grid, pmxh_std_grid = emu.predict_grid(params=full_params.tolist())
        pmpsi_mean_k, pmpsi_std_k = pmpsi_from_pmxh(
            pmxh_mean_grid=np.asarray(pmxh_mean_grid, dtype=float)[:, k_mask],
            pmxh_std_grid=np.asarray(pmxh_std_grid, dtype=float)[:, k_mask],
            z_grid=np.asarray(emu.z_grid, dtype=float),
            k_grid=k_grid,
            k_obs=k_grid,
            z_select=float(args.z_select),
        )
        pmpsi_mean_k, pmpsi_std_k = _convert_spectrum(k=k_grid, y=pmpsi_mean_k, yerr=pmpsi_std_k, spectrum=spectrum)
        label = f"model{i+1} (zeta={zeta_uv:.4g}, Tvir={tvir:.4g})"
        models.append((label, pmpsi_mean_k, pmpsi_std_k))

    out_txt = Path(str(args.out_txt))
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    if len(models) == 1:
        arr = np.column_stack([k_grid, np.asarray(models[0][1], dtype=float), np.asarray(models[0][2], dtype=float)])
    else:
        arr = np.column_stack(
            [
                k_grid,
                np.asarray(models[0][1], dtype=float),
                np.asarray(models[0][2], dtype=float),
                np.asarray(models[1][1], dtype=float),
                np.asarray(models[1][2], dtype=float),
            ]
        )
    np.savetxt(str(out_txt), arr, fmt="%.10e")

    title = f"z={float(args.z_select):.2f}"
    if lp is not None and idx is not None and len(points) == 1 and chain is not None:
        title = f"{title} logp={lp:.6g}"
    obs_k = payload.get("k_obs") if payload else None
    obs_y = payload.get("pmpsi_obs") if payload else None
    obs_yerr = payload.get("pmpsi_sigma") if payload else None
    if obs_k is not None and obs_y is not None:
        obs_y, obs_yerr = _convert_spectrum(k=np.asarray(obs_k, dtype=float), y=np.asarray(obs_y, dtype=float), yerr=obs_yerr, spectrum=spectrum)
    y_label = "Δ²_mψ" if spectrum == "delta2" else "P_mψ"
    _plot_k_curve(
        k=k_grid,
        models=models,
        obs_k=obs_k,
        obs_y=obs_y,
        obs_yerr=obs_yerr,
        out=Path(str(args.out_plot)),
        title=title,
        y_label=y_label,
    )

    if idx is not None and len(points) == 1:
        print(f"picked index: {idx}")
    elif len(points) == 1:
        print("picked index: manual")
    else:
        print("picked index: manual (2 points)")
    for i, (zeta_uv, tvir) in enumerate(points):
        print(f"point{i+1} zeta_uv: {zeta_uv}")
        print(f"point{i+1} tvir: {tvir}")
    if lp is not None and idx is not None and len(points) == 1:
        print(f"logp: {lp}")
    print(f"saved txt: {out_txt}")
    print(f"saved plot: {args.out_plot}")


if __name__ == "__main__":
    main()
