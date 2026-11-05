import os
import numpy as np
from vega import VegaInterface
from vega.vega_interface import BLIND_FIXED_PARS
import copy


# # full combined reio
# MAIN_INI = "/home/zhuxing/vega/analysis/main_reio.ini"
# POLY_TXT = "/home/zhuxing/vega/Polychord_result/full_reio_extendprior/Polychord_fullcombined_extendprior.txt"
# POLY_PARAMNAMES = "/home/zhuxing/vega/Polychord_result/full_reio_extendprior/Polychord_fullcombined_extendprior.paramnames"

# full combined fid
MAIN_INI = "/home/zhuxing/vega/analysis/main_fid.ini"
POLY_TXT = "/home/zhuxing/vega/Polychord_result/full_fid/Polychord_fullcombine_fid.txt"
POLY_PARAMNAMES = "/home/zhuxing/vega/Polychord_result/full_fid/Polychord_fullcombine_fid.paramnames"

## auto fiducial
# MAIN_INI = "/home/zhuxing/vega/analysis/main_fid.ini"
# POLY_TXT = "/home/zhuxing/vega/Polychord_result/auto_fiducial/lyaxlya_lyaxlyb_s3.txt"
# POLY_PARAMNAMES = "/home/zhuxing/vega/Polychord_result/auto_fiducial/lyaxlya_lyaxlyb_s3.paramnames"

## auto reio
# MAIN_INI = "/home/zhuxing/vega/analysis/main_reio.ini"
# POLY_TXT = "/home/zhuxing/vega/Polychord_result/auto_reio_extendprior/Polychord_auto_extendprior.txt"
# POLY_PARAMNAMES = "/home/zhuxing/vega/Polychord_result/auto_reio_extendprior/Polychord_auto_extendprior.paramnames"

R_BINS = [(10, 40), (40, 70), (70, 100), (100, 180)]


def read_paramnames(path):
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            names.append(line.split()[0])
    return names


def read_polychord_map(txt_path, paramnames):
    arr = np.loadtxt(txt_path)
    npar = len(paramnames)
    expected_cols = 2 + npar
    if arr.shape[1] != expected_cols:
        raise ValueError(f"Unexpected number of columns in {txt_path}: got {arr.shape[1]}, expected {expected_cols}")

    weights = arr[:, 0]
    m2logL = arr[:, 1]
    theta = arr[:, 2:]
    logL = -0.5 * m2logL
    imax = int(np.argmax(logL))

    params = {name: float(theta[imax, i]) for i, name in enumerate(paramnames)}
    return {
        "index": imax,
        "logL": float(logL[imax]),
        "m2logL": float(m2logL[imax]),
        "weight": float(weights[imax]),
        "params": params,
    }


def _init_blinding(vega):
    if vega._blind is not None:
        return
    vega._blind = any(getattr(data_obj, "blind", False) for data_obj in vega.data.values())


def build_local_params(vega, params):
    _init_blinding(vega)
    local_params = copy.deepcopy(vega.params)
    for par, val in params.items():
        local_params[par] = val
    if vega._blind:
        for par in list(local_params.keys()):
            if par in BLIND_FIXED_PARS:
                local_params[par] = 1.0
    return local_params


def compute_data_diff_and_r(vega, sample_params):
    local_params = build_local_params(vega, sample_params)

    if vega._use_global_cov:
        full_model_parts = []
        full_data_parts = []
        full_r_parts = []

        for name in vega.corr_items:
            model_cf = vega.models[name].compute(
                local_params, vega.fiducial["pk_full"], vega.fiducial["pk_smooth"]
            )
            full_model_parts.append(model_cf)
            full_data_parts.append(vega.data[name].masked_data_vec)
            full_r_parts.append(
                vega.data[name].data_coordinates.r_regular_grid[vega.data[name].data_mask]
            )

        full_model = np.concatenate(full_model_parts)
        full_data = np.concatenate(full_data_parts)
        r_masked = np.concatenate(full_r_parts)

        diff = full_data - full_model[vega.full_model_mask]
        y = vega.masked_global_invcov.dot(diff)
        return diff, y, r_masked

    diff_parts = []
    y_parts = []
    r_parts = []
    for name in vega.corr_items:
        model_cf = vega.models[name].compute(
            local_params, vega.fiducial["pk_full"], vega.fiducial["pk_smooth"]
        )
        diff = vega.data[name].masked_data_vec - model_cf[vega.data[name].model_mask]
        y = vega.data[name].inv_masked_cov.dot(diff)
        r = vega.data[name].data_coordinates.r_regular_grid[vega.data[name].data_mask]
        diff_parts.append(diff)
        y_parts.append(y)
        r_parts.append(r)
    return np.concatenate(diff_parts), np.concatenate(y_parts), np.concatenate(r_parts)


def split_chi2_by_r(diff, y, r_masked, r_bins):
    total = float(diff.dot(y))
    results = []
    used = np.zeros(r_masked.shape[0], dtype=bool)

    for rmin, rmax in r_bins:
        sel = (r_masked > rmin) & (r_masked < rmax)
        used |= sel
        chi2_bin = float(diff[sel].dot(y[sel]))
        results.append({"rmin": rmin, "rmax": rmax, "n": int(sel.sum()), "chi2": chi2_bin})

    leftover = ~used
    if int(leftover.sum()) > 0:
        chi2_other = float(diff[leftover].dot(y[leftover]))
        results.append({"rmin": None, "rmax": None, "n": int(leftover.sum()), "chi2": chi2_other})

    return total, results


main_ini = os.path.expandvars(MAIN_INI)
poly_txt = os.path.expandvars(POLY_TXT)
poly_paramnames = os.path.expandvars(POLY_PARAMNAMES)

if not os.path.exists(main_ini):
    raise FileNotFoundError(f"MAIN_INI not found: {main_ini}")
if not os.path.exists(poly_txt):
    raise FileNotFoundError(f"POLY_TXT not found: {poly_txt}")
if not os.path.exists(poly_paramnames):
    raise FileNotFoundError(f"POLY_PARAMNAMES not found: {poly_paramnames}")

paramnames = read_paramnames(poly_paramnames)
map_point = read_polychord_map(poly_txt, paramnames)

vega = VegaInterface(main_ini)
chi2_total = float(vega.chi2(map_point["params"]))

saved_priors = vega.priors
vega.priors = {}
chi2_data_only = float(vega.chi2(map_point["params"]))
vega.priors = saved_priors

diff, y, r_masked = compute_data_diff_and_r(vega, map_point["params"])
chi2_data_only_check, chi2_bins = split_chi2_by_r(diff, y, r_masked, R_BINS)

print(f"MAIN_INI = {main_ini}")
print(f"POLY_TXT = {poly_txt}")
print(f"POLY_PARAMNAMES = {poly_paramnames}")
print(f"MAP_INDEX = {map_point['index']}")
print(f"MAP_LOG_L = {map_point['logL']}")
print(f"MAP_-2LOG_L = {map_point['m2logL']}")
print(f"MAP_WEIGHT = {map_point['weight']}")
print(f"CHI2_TOTAL = {chi2_total}")
print(f"CHI2_DATA_ONLY = {chi2_data_only}")
print(f"CHI2_DATA_ONLY_CHECK = {chi2_data_only_check}")
print(f"CHI2_PRIOR = {chi2_total - chi2_data_only}")

print("CHI2_BY_R (data-only decomposition using global precision):")
for item in chi2_bins:
    if item["rmin"] is None:
        print(f"  other: N = {item['n']}, CHI2 = {item['chi2']}")
    else:
        print(f"  r in ({item['rmin']},{item['rmax']}): N = {item['n']}, CHI2 = {item['chi2']}")
