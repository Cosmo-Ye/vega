from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from astropy import constants as const
from scipy.integrate import quad
from scipy.interpolate import interp1d

# python tau_calculation.py --mode file --xhi-file mcmc/xhi_wholeband.txt --xhi-col center --z-max 35
# python tau_calculation.py --mode emulator --zeta 25 --tvir 3e4 --z-max 35
# python tau_calculation.py --mode file --xhi-file mcmc/xhi_wholeband.txt --z-max 35 --z-obs 20 --tau-out mcmc/tau_curve.png

c = const.c.value * 100
G = const.G.value * 1e3
sigmaT = const.sigma_T.value * 1e4
Y = 0.245
m_H = 1.008
m_He = 4.0026
m_p = const.m_p.value * 1e3
c1 = (m_H / m_He) * (Y / (1 - Y))
x_rec = 2.053700009128e-04
z_re_HeII = 3.5
dz_HeII = 0.5
z_max = 35

# 21cmFAST cosmology
h = 0.6774
H0 = h * 3.2407e-18
Om = 0.3088
OL = 0.7012
Or = 8.6e-5
Ob = 0.02230 / h**2
RHOcrit_cgs = (3.0 * H0 * H0) / (8.0 * np.pi * G)
n0 = RHOcrit_cgs * Ob * (1 - Y) / m_p

# # DESI DR1 cosmology
# h = 0.6736
# H0 = h * 3.2407e-18
# Om = 0.315094454725487
# OL = 0.684825913083526
# Or = 7.9638e-5
# Ob = 0.02237 / h**2
# RHOcrit_cgs = (3.0 * H0 * H0) / (8.0 * np.pi * G)
# n0 = RHOcrit_cgs * Ob * (1 - Y) / m_p


def _hubble_z(z: float | np.ndarray) -> float | np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    return H0 * np.sqrt(Om * (1.0 + z_arr) ** 3 + OL + Or * (1.0 + z_arr) ** 4)


def get_xHI(
    *,
    mode: str,
    z_max_integral: float,
    xhi_file: str | None,
    xhi_col: str,
    xhi_model: str,
    zeta: float | None,
    tvir: float | None,
    z_grid_min: float,
    z_grid_max: float,
    nz_grid: int,
) -> Callable[[float | np.ndarray], np.ndarray]:
    if mode == "file":
        if xhi_file is None:
            raise ValueError("--xhi-file is required when --mode file")
        band = np.loadtxt(str(xhi_file), dtype=float)
        if band.ndim != 2 or band.shape[1] < 2:
            raise ValueError(f"{xhi_file} must have at least 2 columns: z, xHI..., got {band.shape}")

        if xhi_col == "center":
            col = 3 if band.shape[1] >= 4 else 1
        elif xhi_col == "low":
            col = 1
        elif xhi_col == "high":
            col = 2 if band.shape[1] >= 3 else 1
        else:
            raise ValueError("--xhi-col must be one of: low, high, center")

        z_src = np.asarray(band[:, 0], dtype=float)
        x_src = np.asarray(band[:, col], dtype=float)
    elif mode == "emulator":
        if zeta is None or tvir is None:
            raise ValueError("--zeta and --tvir are required when --mode emulator")
        from xhi_shape_emulator import XHIShapeEmulator

        emu = XHIShapeEmulator.load(str(xhi_model))
        z_src = np.linspace(float(z_grid_min), float(z_grid_max), int(nz_grid), dtype=float)
        x_src, _ = emu.predict_many(params=[float(zeta), float(tvir)], z_values=z_src)
        x_src = np.asarray(x_src, dtype=float)
    else:
        raise ValueError("--mode must be one of: file, emulator")

    order = np.argsort(z_src)
    z_src = z_src[order]
    x_src = x_src[order]
    z_src = np.asarray(z_src, dtype=float)
    x_src = np.asarray(x_src, dtype=float)
    mask = np.isfinite(z_src) & np.isfinite(x_src)
    z_src = z_src[mask]
    x_src = x_src[mask]
    if z_src.size < 2:
        raise ValueError("xHI(z) input has too few finite points to interpolate")

    f = interp1d(
        z_src,
        np.clip(x_src, 0.0, 1.0),
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, 1.0),
        assume_sorted=True,
    )

    def xhi(z: float | np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        out = np.asarray(f(z_arr), dtype=float)
        out = np.where(z_arr < 0.0, 0.0, out)
        out = np.where(z_arr > float(z_max_integral), 1.0, out)
        return np.clip(out, 0.0, 1.0)

    return xhi


def _ne(z: float, xhi: Callable[[float], np.ndarray]) -> float:
    zz = float(z)
    x = float(np.asarray(xhi(zz), dtype=float).reshape(()))
    n_reio1 = float(n0 * (1.0 + c1) * (1.0 - x))
    n_reio2 = float(n0 * 0.5 * c1 * (np.tanh((z_re_HeII - zz) / dz_HeII) + 1.0))
    n_rec = float(n0 * x_rec)
    return float(n_reio1 + n_reio2 + n_rec)


def compute_tau(*, xhi: Callable[[float], np.ndarray], z_max_integral: float) -> float:
    def integrand(z: float) -> float:
        hz = float(_hubble_z(z))
        return float(_ne(z, xhi) * (1.0 + float(z)) ** 2 / hz)

    val = quad(lambda z: integrand(float(z)), 0.0, float(z_max_integral), epsabs=0.0, epsrel=1e-6, limit=200)[0]
    return float(c * sigmaT * val)


def compute_tau_curve(*, xhi: Callable[[float | np.ndarray], np.ndarray], z_grid: np.ndarray) -> np.ndarray:
    z = np.asarray(z_grid, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_grid must be a 1D array with at least 2 points")
    if np.any(~np.isfinite(z)):
        raise ValueError("z_grid contains non-finite values")
    if np.any(np.diff(z) <= 0):
        raise ValueError("z_grid must be strictly increasing")

    x = np.asarray(xhi(z), dtype=float)
    x = np.clip(x, 0.0, 1.0)
    n_reio1 = n0 * (1.0 + c1) * (1.0 - x)
    n_reio2 = n0 * 0.5 * c1 * (np.tanh((z_re_HeII - z) / dz_HeII) + 1.0)
    n_rec = n0 * x_rec
    ne = n_reio1 + n_reio2 + n_rec
    hz = np.asarray(_hubble_z(z), dtype=float)
    f = ne * (1.0 + z) ** 2 / hz

    dz = np.diff(z)
    area = 0.5 * (f[1:] + f[:-1]) * dz
    integ = np.concatenate([[0.0], np.cumsum(area)])
    return np.asarray(c * sigmaT * integ, dtype=float)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tau_calculation")
    p.add_argument("--mode", choices=["file", "emulator"], default="file")
    p.add_argument("--z-max", type=float, default=float(z_max))

    p.add_argument("--xhi-file", default="mcmc/xhi_wholeband.txt")
    p.add_argument("--xhi-col", choices=["low", "high", "center"], default="center")
    p.add_argument("--z-obs", type=float, default=None)
    p.add_argument("--tau-nz", type=int, default=600)
    p.add_argument("--tau-out", default="mcmc/tau_curve.png")

    p.add_argument("--xhi-model", default="xhi_shape_emulator.joblib")
    p.add_argument("--zeta", type=float, default=None)
    p.add_argument("--tvir", type=float, default=None)
    p.add_argument("--z-grid-min", type=float, default=5.0)
    p.add_argument("--z-grid-max", type=float, default=20.0)
    p.add_argument("--nz-grid", type=int, default=300)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    zmax = float(args.z_max)
    if zmax <= 0:
        raise SystemExit("--z-max must be > 0")

    xhi = get_xHI(
        mode=str(args.mode),
        z_max_integral=zmax,
        xhi_file=str(args.xhi_file) if args.xhi_file is not None else None,
        xhi_col=str(args.xhi_col),
        xhi_model=str(args.xhi_model),
        zeta=args.zeta,
        tvir=args.tvir,
        z_grid_min=float(args.z_grid_min),
        z_grid_max=float(args.z_grid_max),
        nz_grid=int(args.nz_grid),
    )
    tau = compute_tau(xhi=xhi, z_max_integral=zmax)
    print(tau)

    if args.z_obs is not None:
        if str(args.mode) != "file":
            raise SystemExit("--z-obs is only supported with --mode file")
        z_obs = float(args.z_obs)
        if z_obs <= 0:
            raise SystemExit("--z-obs must be > 0")

        nz = int(args.tau_nz)
        if nz < 10:
            raise SystemExit("--tau-nz must be >= 10")

        z_grid = np.linspace(0.0, z_obs, nz, dtype=float)
        xhi_low = get_xHI(
            mode="file",
            z_max_integral=z_obs,
            xhi_file=str(args.xhi_file),
            xhi_col="low",
            xhi_model=str(args.xhi_model),
            zeta=None,
            tvir=None,
            z_grid_min=float(args.z_grid_min),
            z_grid_max=float(args.z_grid_max),
            nz_grid=int(args.nz_grid),
        )
        xhi_high = get_xHI(
            mode="file",
            z_max_integral=z_obs,
            xhi_file=str(args.xhi_file),
            xhi_col="high",
            xhi_model=str(args.xhi_model),
            zeta=None,
            tvir=None,
            z_grid_min=float(args.z_grid_min),
            z_grid_max=float(args.z_grid_max),
            nz_grid=int(args.nz_grid),
        )
        xhi_center = get_xHI(
            mode="file",
            z_max_integral=z_obs,
            xhi_file=str(args.xhi_file),
            xhi_col="center",
            xhi_model=str(args.xhi_model),
            zeta=None,
            tvir=None,
            z_grid_min=float(args.z_grid_min),
            z_grid_max=float(args.z_grid_max),
            nz_grid=int(args.nz_grid),
        )

        tau_low = compute_tau_curve(xhi=xhi_low, z_grid=z_grid)
        tau_high = compute_tau_curve(xhi=xhi_high, z_grid=z_grid)
        tau_center = compute_tau_curve(xhi=xhi_center, z_grid=z_grid)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lo = np.minimum(tau_low, tau_high)
        hi = np.maximum(tau_low, tau_high)

        plt.style.use('mcmc/perf.mplstyle')
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
        # ax.fill_between(z_grid, np.full_like(z_grid, 0.053), np.full_like(z_grid, 0.065), color="olive", alpha=0.20, linewidth=0.0, label="Pagano et al. 2020")
        # ax.axhline(0.059, color='olive', ls="--", lw=1.6)
        # ax.fill_between(z_grid, np.full_like(z_grid, 0.0572), np.full_like(z_grid, 0.068), color="cyan", alpha=0.20, linewidth=0.0, label="de Belsunce et al. 2021")
        # ax.axhline(0.063, color='cyan', ls="--", lw=1.6)
        ax.fill_between(z_grid, np.full_like(z_grid, 0.052), np.full_like(z_grid, 0.064), color="red", alpha=0.20, linewidth=0.0, label="Planck PR4")
        ax.axhline(0.058, color='orange', ls="--", lw=1.6)
        ax.fill_between(z_grid, lo, hi, alpha=0.25, color="black", linewidth=0.0)
        ax.plot(z_grid, tau_center, lw=2.0, color="black", label="DESI DR1")
        ax.set_xlabel("z")
        ax.set_ylabel(r"$\tau(z)$")
        ax.set_xlim(0.0, z_obs)
        ax.set_ylim(0.0, 0.07)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()

        out = Path(str(args.tau_out))
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out))
        print(f"saved: {out}")

if __name__ == "__main__":
    main()
