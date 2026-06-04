from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from mcmc.run_mcmc import pmpsi_from_pmxh
from pmxh_shape_emulator import PMXHShapeEmulator


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="21cm_chiby2")
    p.add_argument("--model", default="pmxh_shape_emulator.joblib")
    p.add_argument("--out", default="pmpsi_chiby2.pdf")
    p.add_argument("--z-select", type=float, default=2.33)
    p.add_argument("--nsamples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zeta-low", type=float, default=48.27599101)
    p.add_argument("--zeta-high", type=float, default=212.49486043)
    p.add_argument("--logtvir-low", type=float, default=5.01566378)
    p.add_argument("--logtvir-high", type=float, default=5.725232)
    p.add_argument("--spectrum", choices=["delta2", "pk"], default="pk")
    return p


def _model_expects_logtvir(emu: PMXHShapeEmulator) -> bool:
    train_params = np.asarray(emu.train_params, dtype=float)
    if train_params.ndim != 2 or train_params.shape[1] < 2:
        raise ValueError(f"Unexpected emu.train_params shape: {train_params.shape}")
    med = float(np.median(train_params[:, 1]))
    return med < 50.0


def _convert_spectrum(*, k: np.ndarray, y: np.ndarray, spectrum: str) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    spectrum = str(spectrum)
    if spectrum == "delta2":
        return y
    if spectrum != "pk":
        raise ValueError(f"Unknown spectrum mode: {spectrum}")
    if np.any(k <= 0):
        raise ValueError("k must be positive to convert between P(k) and Delta^2(k)")
    factor = k**3 / (2.0 * np.pi**2)
    return y / factor


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    plt.style.use("mcmc/perf.mplstyle")
    out = Path(str(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    n = int(args.nsamples)
    if n <= 0:
        raise SystemExit("--nsamples must be > 0")

    zeta_low = float(args.zeta_low)
    zeta_high = float(args.zeta_high)
    logtvir_low = float(args.logtvir_low)
    logtvir_high = float(args.logtvir_high)
    if not (zeta_high > zeta_low):
        raise SystemExit("--zeta-high must be > --zeta-low")
    if not (logtvir_high > logtvir_low):
        raise SystemExit("--logtvir-high must be > --logtvir-low")

    zeta = rng.uniform(zeta_low, zeta_high, size=n).astype(float)
    logtvir = rng.uniform(logtvir_low, logtvir_high, size=n).astype(float)

    emu = PMXHShapeEmulator.load(str(args.model))
    expects_logtvir = _model_expects_logtvir(emu)

    k_grid_full = np.asarray(emu.k_grid, dtype=float)
    k_mask = (k_grid_full > 0.06) & (k_grid_full < 0.2)
    if not np.any(k_mask):
        raise SystemExit("No k modes left after applying 0.06<k<0.2 cut to emu.k_grid")
    k_grid = k_grid_full[k_mask]

    fig = plt.figure(figsize=(4.7, 3))
    ax = fig.add_subplot(111)

    z_select = float(args.z_select)
    spectrum = str(args.spectrum)
    for i in range(n):
        if expects_logtvir:
            params = [float(zeta[i]), float(logtvir[i])]
        else:
            params = [float(zeta[i]), float(10.0 ** float(logtvir[i]))]

        pmxh_mean_grid, pmxh_std_grid = emu.predict_grid(params=params)
        pmxh_mean_grid = np.asarray(pmxh_mean_grid, dtype=float)[:, k_mask]
        pmxh_std_grid = np.asarray(pmxh_std_grid, dtype=float)[:, k_mask]

        pmpsi_mean, _pmpsi_std = pmpsi_from_pmxh(
            pmxh_mean_grid=pmxh_mean_grid,
            pmxh_std_grid=pmxh_std_grid,
            z_grid=np.asarray(emu.z_grid, dtype=float),
            k_grid=k_grid,
            k_obs=k_grid,
            z_select=z_select,
        )
        y = _convert_spectrum(k=k_grid, y=np.asarray(pmpsi_mean, dtype=float), spectrum=spectrum)
        m = np.isfinite(k_grid) & np.isfinite(y)
        if np.any(m):
            ax.plot(k_grid[m], y[m], "-", color="lightgray", lw=0.5)

    ax.set_xlim(float(np.min(k_grid)), float(np.max(k_grid)))
    ax.set_xlabel("k")
    ax.set_ylabel(rf"$P_{{m\\psi}}(z={z_select:g})$" if spectrum == "pk" else rf"$\Delta^2_{{m\\psi}}(z={z_select:g})$")
    ax.set_xscale("log")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(str(out))
    print(f"saved: {out}")
    print(f"nsamples: {n}  expects_log10Tvir: {expects_logtvir}")


if __name__ == "__main__":
    main()
