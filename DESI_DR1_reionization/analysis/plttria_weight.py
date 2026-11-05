from getdist import loadMCSamples
from getdist import plots
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

"""
    Usage: python python plttria.py <points> #<label> 
    # label not needed if everything if the same directory that has the chains
    contains the parameter name. 
    e.g. if points file is lya_auto_fid/lyaxlya_lyaxlyb_s1.txt then
    python plttria.py lya_auto_fid/lyaxlya_lyaxlyb_s1 conservative_fid
    
"""

plt.style.use('emulator/mcmc/perf.mplstyle')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
chain_file = sys.argv[1]
model_str = str(sys.argv[2])

print(chain_file)
#print(label)
s = loadMCSamples(str(chain_file))


samples = []
samples.append(s)
labels = ['Yuk3'] #['Fid.']

# let's include the new parametrization,
# first define the k_star
k_star = 0.06 # [1/Mpc]
k0 = 1 # [1/Mpc]
x_star = k_star / k0

#A_eff = A_yuk * x_star^(-beta_yuk) * exp(-alpha_yuk * x_star)
#alpha_eff = alpha_yuk * x_star
#beta_eff = beta_yuk
if model_str == 'conservative_yuk':
    s.addDerived(s.getParams().yuk_A * x_star**(-1. * s.getParams().yuk_beta) * np.exp(-1. * s.getParams().yuk_alpha * x_star), name='A_reio', label='A_\mathrm{reio}')
    s.addDerived(s.getParams().yuk_alpha * x_star, name='alpha_reio', label=r'\alpha_\mathrm{reio}')
    s.addDerived(s.getParams().yuk_beta, name='beta_reio', label=r'\beta_\mathrm{reio}')
elif model_str == 'optimistic_yuk':
    s.addDerived(s.getParams().yuk_A * x_star**(-1. * s.getParams().yuk_beta) * np.exp(-1. * s.getParams().yuk_alpha * x_star), name='A_reio', label='A_\mathrm{reio}')
    s.addDerived(s.getParams().yuk_alpha * x_star, name='alpha_reio', label=r'\alpha_\mathrm{reio}')
    s.addDerived(s.getParams().yuk_beta, name='beta_reio', label=r'\beta_\mathrm{reio}')




g = plots.get_subplot_plotter(subplot_size=1)


#g.settings.line_styles = ['-C0', '--C1', '-.C3', ':C2', '--C7', ':C5']
g.settings.line_styles = ['--C2', '-C0']
g.settings.linewidth = 4
g.settings.solid_colors = ['C2', 'C0']
g.settings.alpha_filled_add = 0.75
g.settings.title_limit = 1
g.settings.title_limit_fontsize = 14
g.settings.legend_fontsize = 28
g.settings.figure_legend_frame = False
#g.triangle_plot(s, legend_labels=[label], filled=True)
g.triangle_plot(samples, legend_labels=labels, filled=True)
if model_str == 'optimistic_yuk':
    g.export(os.path.join(OUT_DIR, 'corner_eff_optimistic_yuk.pdf'))
elif model_str == 'optimistic_fid':
    g.export(os.path.join(OUT_DIR, 'corner_eff_optimistic_fid.pdf'))
elif model_str == 'conservative_yuk':
    g.export(os.path.join(OUT_DIR, 'corner_eff_conservative_yuk.pdf'))
elif model_str == 'conservative_fid':
    g.export(os.path.join(OUT_DIR, 'corner_eff_conservative_fid.pdf'))
else:
    print('Check model input')
print('Saved corner plot to', os.path.join(OUT_DIR, f'corner_eff_{model_str}.pdf'))
#g.export('corner_eff_yuk_auto_10_r_180.pdf')




# due to weights and smoothing we are doing our own constraints
def weighted_percentile(values, weights, percentiles):
    """
    Compute weighted percentiles.

    Parameters
    ----------
    values : array_like
        Samples.
    weights : array_like
        Corresponding sample weights.
    percentiles : array_like
        Percentiles in the range [0, 100].

    Returns
    -------
    result : ndarray
        Weighted percentiles.
    """

    values = np.asarray(values)
    weights = np.asarray(weights)
    percentiles = np.asarray(percentiles)

    # Sort samples by parameter value
    order = np.argsort(values)

    values_sorted = values[order]
    weights_sorted = weights[order]

    # Normalize weights
    weights_sorted = weights_sorted / np.sum(weights_sorted)

    # Cumulative posterior probability
    cumulative = np.cumsum(weights_sorted)

    # Interpolate requested quantiles
    return np.interp(
        percentiles / 100.0,
        cumulative,
        values_sorted
    )

samples = s.samples
weights = s.weights

# need central and 1sig
percentiles = [16, 50, 84]


print(f'Number of samples = {len(samples)}')
print(f'Sum of weights    = {np.sum(weights):.6e}')

results = []

for i, par in enumerate(s.getParamNames().names):
    if i >= samples.shape[1]:
        continue

    values = samples[:, i]
    if not np.any(np.isfinite(values)):
        continue
    if model_str == 'optimistic_yuk':
        # Special handling for alpha_reio: report 95% upper bound
        if par.name == 'alpha_reio':
            # Weighted 95th percentile (upper credible limit)
            upper_95 = weighted_percentile(values, weights, [95])[0]

            results.append({
                'parameter': par.name,
                'upper_95': upper_95,
                'constraint': f'{par.name} < {upper_95:.6g}'
            })
        else:
            # Original summary for other parameters
            p16, p50, p84 = weighted_percentile(
                values,
                weights,
                [16, 50, 84]
            )

            minus_error = p50 - p16
            plus_error = p84 - p50

            results.append({
                'parameter': par.name,
                'lower_16': p16,
                'median': p50,
                'upper_84': p84,
                'minus_error': minus_error,
                'plus_error': plus_error,
                'constraint': (
                    f'{p50:.6g} '
                    f'+{plus_error:.6g} '
                    f'-{minus_error:.6g}'
                )
            })
    elif model_str == 'optimistic_fid':
        # Original summary for other parameters
        p16, p50, p84 = weighted_percentile(
                values,
                weights,
                [16, 50, 84]
        )

        minus_error = p50 - p16
        plus_error = p84 - p50

        results.append({
                'parameter': par.name,
                'lower_16': p16,
                'median': p50,
                'upper_84': p84,
                'minus_error': minus_error,
                'plus_error': plus_error,
                'constraint': (
                    f'{p50:.6g} '
                    f'+{plus_error:.6g} '
                    f'-{minus_error:.6g}'
            )
        })
    elif model_str == 'conservative_yuk':
         # Special handling for alpha_reio: unconstrained
        if par.name == 'alpha_reio':

            results.append({
                'parameter': par.name,
                'constraint': f'{par.name} unconstrained'
            })
        elif par.name == 'bias_hcd':
            # Weighted 95th percentile (upper credible limit)
            upper_95 = weighted_percentile(values, weights, [95])[0]

            results.append({
                'parameter': par.name,
                'upper_95': upper_95,
                'constraint': f'{par.name} < {upper_95:.6g}'
            })
        elif par.name == 'bias_SiIII(1207)':
            # Weighted 95th percentile (upper credible limit)
            upper_95 = weighted_percentile(values, weights, [95])[0]

            results.append({
                'parameter': par.name,
                'upper_95': upper_95,
                'constraint': f'{par.name} < {upper_95:.6g}'
            })
        else:
            # Original summary for other parameters
            p16, p50, p84 = weighted_percentile(
                values,
                weights,
                [16, 50, 84]
            )

            minus_error = p50 - p16
            plus_error = p84 - p50

            results.append({
                'parameter': par.name,
                'lower_16': p16,
                'median': p50,
                'upper_84': p84,
                'minus_error': minus_error,
                'plus_error': plus_error,
                'constraint': (
                    f'{p50:.6g} '
                    f'+{plus_error:.6g} '
                    f'-{minus_error:.6g}'
                )
            })
    elif model_str == 'conservative_fid':
        if par.name == 'bias_hcd':
            # Weighted 95th percentile (upper credible limit)
            upper_95 = weighted_percentile(values, weights, [95])[0]

            results.append({
                'parameter': par.name,
                'upper_95': upper_95,
                'constraint': f'{par.name} < {upper_95:.6g}'
            })
        elif par.name == 'bias_SiIII(1207)':
            # Weighted 95th percentile (upper credible limit)
            upper_95 = weighted_percentile(values, weights, [95])[0]

            results.append({
                'parameter': par.name,
                'upper_95': upper_95,
                'constraint': f'{par.name} < {upper_95:.6g}'
            })
        else:
            # Original summary for other parameters
            p16, p50, p84 = weighted_percentile(
                    values,
                    weights,
                    [16, 50, 84]
            )

            minus_error = p50 - p16
            plus_error = p84 - p50

            results.append({
                    'parameter': par.name,
                    'lower_16': p16,
                    'median': p50,
                    'upper_84': p84,
                    'minus_error': minus_error,
                    'plus_error': plus_error,
                    'constraint': (
                        f'{p50:.6g} '
                        f'+{plus_error:.6g} '
                        f'-{minus_error:.6g}'
                )
            })



import pandas as pd
results_df = pd.DataFrame(results)

print('\nWeighted posterior constraints:\n')

for _, row in results_df.iterrows():

    print(
        f"{row['parameter']:25s} = "
        f"{row['constraint']}"
    )


# save
if model_str == 'optimistic_yuk':
    output_file = os.path.join(OUT_DIR, 'polychord_weighted_constraints_opt_yuk.csv')
elif model_str == 'optimistic_fid':
    output_file = os.path.join(OUT_DIR, 'polychord_weighted_constraints_opt_fid.csv')
elif model_str == 'conservative_yuk':
    output_file = os.path.join(OUT_DIR, 'polychord_weighted_constraints_conservative_yuk.csv')
elif model_str == 'conservative_fid':
    output_file = os.path.join(OUT_DIR, 'polychord_weighted_constraints_conservative_fid.csv')
else:
    print('Check model input')

results_df.to_csv(
    output_file,
    index=False
)
print('Saved weighted constraints to', output_file)

sys.exit()
print("Number of samples:", len(s.weights))
print("Weight min/max:", np.min(s.weights), np.max(s.weights))
print("Effective sample size:", 1 / np.sum(s.weights**2))




A_reio = s.samples[:, s.index['A_reio']]

print("mean =", np.average(A_reio, weights=s.weights))

# Weighted median and equal-tail 68% interval
print(
    "quantiles =",
    s.confidence(s.index['A_reio'], 0.16),
    s.confidence(s.index['A_reio'], 0.16, upper=True)
)

Aidx = s.index['A_reio']

lo = s.confidence(Aidx, 0.16)
hi = s.confidence(Aidx, 0.16, upper=True)
print('GetDist confidence:')
print(lo, hi)


# Get the actual derived samples stored by GetDist
A_gd = s.samples[:, s.index['A_reio']]

print("GetDist stored A_reio:")
print("mean   =", np.mean(A_gd))
print("std    =", np.std(A_gd))
print("median =", np.median(A_gd))
print("16,84  =", np.percentile(A_gd, [16, 84]))


stats = s.getMargeStats()

p = stats.parWithName('A_reio')

print("GetDist:")
print("mean =", p.mean)
print("limits =", [(x.lower, x.upper) for x in p.limits])

# Direct calculation from the original samples
raw = s.samples

yuk_A = raw[:, s.index['yuk_A']]
yuk_alpha = raw[:, s.index['yuk_alpha']]
yuk_beta = raw[:, s.index['yuk_beta']]

A_reio_direct = (
    yuk_A
    * x_star**(-yuk_beta)
    * np.exp(-yuk_alpha * x_star)
)

print("\nDIRECT NumPy A_reio:")
print("mean   =", np.mean(A_reio_direct))
print("std    =", np.std(A_reio_direct))
print("median =", np.median(A_reio_direct))
print("16,84  =", np.percentile(A_reio_direct, [16, 84]))

# conservative yuk
# python analysis/plttria_weight.py /Users/zhengyifan/vega_result/40_180/auto_reio_40_180_3free/Poly_bbt_40_180_auto conservative_yuk
# test python analysis/plttria_weight.py /Users/zhengyifan/vega_result/40_180/full_reio_40_180_Uyuk/Poly_bbt_40_180_full_Uyuk conservative_yuk

# conservative fid
# python analysis/plttria_weight.py /Users/zhengyifan/vega_result/40_180/auto_fid_40_180/Poly_bbt_40_180_auto_fid conservative_fid
# test python analysis/plttria_weight.py /Users/zhengyifan/vega_result/40_180/full_fid_40_180/Poly_bbt_40_180_full_fid conservative_fid

# optimistic yuk
# python analysis/plttria_weight.py /Users/zhengyifan/vega_result/Polychord_fullcombined_extendprior optimistic_yuk

# optimistic fid
# python analysis/plttria_weight.py /Users/zhengyifan/vega/Polychord_result/full_fid/Polychord_fullcombine_fid optimistic_fid
# test python analysis/plttria_weight.py /Users/zhengyifan/vega/Polychord_result/auto_fiducial/lyaxlya_lyaxlyb_s3 optimistic_fid
