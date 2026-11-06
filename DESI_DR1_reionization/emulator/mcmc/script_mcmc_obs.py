from getdist import loadMCSamples
import numpy as np
from pmxh_shape_emulator import PMXHShapeEmulator

"""
    Base script for Yifan to modify later.
    It currently loads the 1 sigma error and data for Pmpsi based on DESI DR1
    Note that it can load the combine all or the Lya auto. 
"""

def load_pmpsi_from_chain(
    *,
    chain_root: str,
    k_bin: list[float],
    ignore_rows: float = 0.2,
) -> dict[str, np.ndarray]:
    s = loadMCSamples(chain_root, settings={"ignore_rows": float(ignore_rows)})
    samples = s.samples
    yukA = samples[:, s.index["yuk_A"]]
    yukalpha = samples[:, s.index["yuk_alpha"]]
    yukbeta = samples[:, s.index["yuk_beta"]]

    kk = np.asarray(k_bin, dtype=float)
    pmpsi_samples = -yukA[:, None] * np.exp(-yukalpha[:, None] * kk[None, :]) * kk[None, :] ** (-yukbeta[:, None])
    factor = kk**3 / (2.0 * np.pi**2)
    pmpsi_samples = pmpsi_samples * factor[None, :]
    median = np.percentile(pmpsi_samples, 50, axis=0)
    low = np.percentile(pmpsi_samples, 16, axis=0)
    high = np.percentile(pmpsi_samples, 84, axis=0)
    sigma = 0.5 * (high - low)

    return {
        "k": kk,
        "median": np.asarray(median, dtype=float),
        "low": np.asarray(low, dtype=float),
        "high": np.asarray(high, dtype=float),
        "sigma": np.asarray(sigma, dtype=float),
    }


def main() -> None:
    dir_combine = "mcmc/Polychord_result/full_reio_extendprior/Polychord_fullcombined_extendprior"
    emu = PMXHShapeEmulator.load("pmxh_shape_emulator.joblib")
    k_grid_full = np.asarray(emu.k_grid, dtype=float)
    k_bin = k_grid_full[(k_grid_full > 0.06) & (k_grid_full < 0.4)].tolist()
    out = load_pmpsi_from_chain(chain_root=dir_combine, k_bin=k_bin, ignore_rows=0.2)
    print(out["low"])
    print(out["median"])
    print(out["high"])

    dir_auto = "mcmc/Polychord_result/auto_reio_extendprior/Polychord_auto_extendprior"
    _ = load_pmpsi_from_chain(chain_root=dir_auto, k_bin=k_bin, ignore_rows=0.2)


if __name__ == "__main__":
    main()
