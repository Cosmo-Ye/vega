from __future__ import annotations

import argparse
import math
from typing import Sequence

import numpy as np

from mcmc.run_mcmc import (
    _build_full_params,
    _log_likelihood_gaussian_diag,
    _log_prior,
    pmpsi_from_pmxh,
)
from mcmc.script_mcmc_obs import load_pmpsi_from_chain
from pmxh_shape_emulator import PMXHShapeEmulator
from xhi_shape_emulator import XHIShapeEmulator


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="check_point")
    p.add_argument("--zeta", type=float, required=True)
    p.add_argument("--tvir", type=float, required=True)

    p.add_argument("--pmpsi-chain-root", default="mcmc/Polychord_result/full_reio_extendprior/Polychord_fullcombined_extendprior")
    p.add_argument("--ignore-rows", type=float, default=0.2)

    p.add_argument("--emu-model", default="pmxh_shape_emulator.joblib")
    p.add_argument("--xhi-model", default="xhi_shape_emulator.joblib")
    p.add_argument("--xhi-z", type=float, default=5.37)
    p.add_argument("--xhi-max", type=float, default=0.05)

    p.add_argument("--k-min", type=float, default=0.06)
    p.add_argument("--k-max", type=float, default=0.2)

    p.add_argument("--base-params", nargs="*", type=float, default=None)
    p.add_argument("--zeta-idx", type=int, default=0)
    p.add_argument("--tvir-idx", type=int, default=1)
    p.add_argument("--tvir-in-log10", action="store_true", default=False)

    p.add_argument("--zeta-min", type=float, default=0.0)
    p.add_argument("--zeta-max", type=float, default=200.0)
    p.add_argument("--tvir-min", type=float, default=1e4)
    p.add_argument("--tvir-max", type=float, default=1e6)

    p.add_argument("--z-select", type=float, default=2.33)
    p.add_argument("--no-emu-std", action="store_true", default=False)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)

    zeta_uv = float(args.zeta)
    tvir = float(args.tvir)

    emu = PMXHShapeEmulator.load(str(args.emu_model))
    xhi_emu = XHIShapeEmulator.load(str(args.xhi_model))

    k_grid_full = np.asarray(emu.k_grid, dtype=float)
    k_min = float(args.k_min)
    k_max = float(args.k_max)
    k_mask = (k_grid_full > k_min) & (k_grid_full < k_max)
    if not np.any(k_mask):
        raise SystemExit(f"No k modes left after applying {k_min}<k<{k_max} cut to emu.k_grid")
    k_grid = k_grid_full[k_mask]

    pmpsi = load_pmpsi_from_chain(
        chain_root=str(args.pmpsi_chain_root),
        k_bin=k_grid.tolist(),
        ignore_rows=float(args.ignore_rows),
    )
    k_obs = np.asarray(pmpsi["k"], dtype=float)
    pmpsi_obs = np.asarray(pmpsi["median"], dtype=float)
    pmpsi_sigma = np.asarray(pmpsi["sigma"], dtype=float)

    ndim = int(emu.train_params.shape[1])
    if args.base_params is None or len(args.base_params) == 0:
        base_params = np.median(np.asarray(emu.train_params, dtype=float), axis=0)
    else:
        base_params = np.asarray(args.base_params, dtype=float)
        if base_params.size != ndim:
            raise SystemExit(f"--base-params must have length {ndim} (emu param dimension)")

    # 1) hard prior
    lp = _log_prior(
        zeta_uv=zeta_uv,
        tvir=tvir,
        zeta_min=float(args.zeta_min),
        zeta_max=float(args.zeta_max),
        tvir_min=float(args.tvir_min),
        tvir_max=float(args.tvir_max),
    )
    if not math.isfinite(lp):
        print("rejected: YES")
        print("reason: out_of_hard_prior_range")
        print("log_prior:", lp)
        return

    # 2) xHI prior
    xhi_mean, xhi_std = xhi_emu.predict(params=[zeta_uv, tvir], z=float(args.xhi_z))
    print("xhi(z):", float(xhi_mean), "at z=", float(args.xhi_z), "std=", float(xhi_std))
    if math.isfinite(xhi_mean) and xhi_mean > float(args.xhi_max):
        print("rejected: YES")
        print("reason: xhi_prior_failed")
        print("threshold:", float(args.xhi_max))
        print("log_prior:", lp)
        return

    # 3) model prediction + likelihood
    full_params = _build_full_params(
        base_params=base_params,
        zeta_uv=zeta_uv,
        tvir=tvir,
        zeta_idx=int(args.zeta_idx),
        tvir_idx=int(args.tvir_idx),
        tvir_in_log10=bool(args.tvir_in_log10),
    )
    pmxh_mean_grid, pmxh_std_grid = emu.predict_grid(params=full_params.tolist())
    pmxh_mean_grid = np.asarray(pmxh_mean_grid, dtype=float)[:, k_mask]
    pmxh_std_grid = np.asarray(pmxh_std_grid, dtype=float)[:, k_mask]

    pmpsi_model, pmpsi_model_std = pmpsi_from_pmxh(
        pmxh_mean_grid=pmxh_mean_grid,
        pmxh_std_grid=pmxh_std_grid,
        z_grid=np.asarray(emu.z_grid, dtype=float),
        k_grid=k_grid,
        k_obs=k_obs,
        z_select=float(args.z_select),
    )

    sigma_tot = pmpsi_sigma.copy()
    if not bool(args.no_emu_std):
        sigma_tot = np.sqrt(sigma_tot * sigma_tot + np.asarray(pmpsi_model_std, dtype=float) ** 2)

    ll = _log_likelihood_gaussian_diag(y_obs=pmpsi_obs, y_model=pmpsi_model, sigma=sigma_tot)
    if not math.isfinite(ll):
        print("rejected: YES")
        print("reason: invalid_likelihood")
        print("log_prior:", lp)
        print("log_like:", ll)
        return

    print("rejected: NO")
    print("reason: passed_all_checks")
    print("log_prior:", float(lp))
    print("log_like:", float(ll))
    print("log_posterior:", float(lp + ll))
    print("chi2_like_term:", float(np.sum(((pmpsi_obs - pmpsi_model) / sigma_tot) ** 2)))
    print("k_obs_n:", int(k_obs.size), "k_min:", float(np.min(k_obs)), "k_max:", float(np.max(k_obs)))


if __name__ == "__main__":
    main()
