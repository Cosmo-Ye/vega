from getdist import loadMCSamples
from getdist import plots
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pandas as pd


def weighted_percentile(values, weights, percentiles):
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    percentiles = np.asarray(percentiles, dtype=float)

    finite = np.isfinite(values) & np.isfinite(weights)
    if not np.any(finite):
        return np.full(percentiles.shape, np.nan)
    values = values[finite]
    weights = weights[finite]

    if np.sum(weights) <= 0:
        return np.full(percentiles.shape, np.nan)

    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]

    weights_sorted = weights_sorted / np.sum(weights_sorted)
    cumulative = np.concatenate(([0.0], np.cumsum(weights_sorted)))
    values_extended = np.concatenate(([values_sorted[0]], values_sorted))

    return np.interp(percentiles / 100.0, cumulative, values_extended)


def effective_sample_size(weights):
    w = np.asarray(weights, dtype=float)
    wsum = np.sum(w)
    if wsum <= 0:
        return 0.0
    wn = w / wsum
    return 1.0 / np.sum(wn * wn)


def compute_weighted_constraints(samples, tag=''):
    s_samples = samples.samples
    s_weights = samples.weights

    print('')
    header = f'[Weighted posterior constraints]'
    if tag:
        header += f'  {tag}'
    print(header)
    print(f'  N samples        = {len(s_samples)}')
    print(f'  Sum(weights)     = {np.sum(s_weights):.6e}')
    print(f'  ESS (Kish)       = {effective_sample_size(s_weights):.1f}')

    results = []
    for i, par in enumerate(samples.getParamNames().names):
        if i >= s_samples.shape[1]:
            continue
        values = s_samples[:, i]
        if not np.any(np.isfinite(values)):
            continue

        w_mean = np.average(values, weights=s_weights)
        p16, p50, p84 = weighted_percentile(values, s_weights, [16, 50, 84])
        minus_error = p50 - p16
        plus_error = p84 - p50

        results.append({
            'parameter': par.name,
            'weighted_mean': w_mean,
            'lower_16': p16,
            'median': p50,
            'upper_84': p84,
            'minus_error': minus_error,
            'plus_error': plus_error,
            'constraint_median': (
                f'{p50:.6g} +{plus_error:.6g} -{minus_error:.6g}'
            ),
            'constraint_mean_1sigma': (
                f'{w_mean:.6g} +/- {0.5 * (plus_error + minus_error):.6g}'
            ),
        })

    df = pd.DataFrame(results)
    for _, row in df.iterrows():
        print(
            f"  {row['parameter']:28s} = "
            f"{row['constraint_median']}   (mean = {row['weighted_mean']:.6g})"
        )
    return df


"""
    Usage: python plttria.py <points1> <points2> <points3>
    e.g. if points files are lya_auto_fid/chain1.txt, chain2.txt, chain3.txt then
    python plttria.py lya_auto_fid/chain1 lya_auto_fid/chain2 lya_auto_fid/chain3

"""

plt.style.use('emulator/mcmc/perf.mplstyle')
chain_file1 = sys.argv[1]
chain_file2 = sys.argv[2]
chain_file3 = sys.argv[3]

tag1 = os.path.splitext(os.path.basename(os.path.normpath(chain_file1)))[0]
tag2 = os.path.splitext(os.path.basename(os.path.normpath(chain_file2)))[0]
tag3 = os.path.splitext(os.path.basename(os.path.normpath(chain_file3)))[0]

print(chain_file1)
print(chain_file2)
print(chain_file3)
s1 = loadMCSamples(str(chain_file1), settings={'ignore_rows': 0})
s2 = loadMCSamples(str(chain_file2), settings={'ignore_rows': 0})
s3 = loadMCSamples(str(chain_file3), settings={'ignore_rows': 0})

k_star = 0.06
k0 = 1
x_star = k_star / k0

for s in [s1, s2, s3]:
    s.addDerived(s.getParams().yuk_A * x_star**(-1. * s.getParams().yuk_beta) * np.exp(-1. * s.getParams().yuk_alpha * x_star), name='A_reio', label='A_\mathrm{reio}')
    s.addDerived(s.getParams().yuk_alpha * x_star, name='alpha_reio', label=r'\alpha_\mathrm{reio}')
    s.addDerived(s.getParams().yuk_beta, name='beta_reio', label=r'\beta_\mathrm{reio}')

labels = ['10', '20', '40']
chains = [(s1, tag1, labels[0]), (s2, tag2, labels[1]), (s3, tag3, labels[2])]

for s, tag, label in chains:
    df_constraints = compute_weighted_constraints(s, tag=f'{label} ({tag})')
    safe = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in tag)
    out_csv = f'weighted_constraints_{safe}.csv'
    df_constraints.to_csv(out_csv, index=False)
    print(f'Saved weighted constraints to {out_csv}')

samples = [s1, s2, s3]

g = plots.get_subplot_plotter(subplot_size=1)

g.settings.line_styles = ['-C0', '--C1', '-.C2']
g.settings.linewidth = 4
g.settings.solid_colors = ['C0', 'C1', 'C2']
g.settings.alpha_filled_add = 0.75
g.settings.title_limit_fontsize = 14
g.settings.legend_fontsize = 28
g.settings.figure_legend_frame = False
g.triangle_plot(samples, legend_labels=labels, filled=True)
out_pdf = f'corner_3chains_{tag1}__{tag2}__{tag3}.pdf'
g.export(out_pdf)
print(f'Saved corner plot to {out_pdf}')

# python analysis/plttria.py /Users/zhengyifan/vega_result/Polychord_auto_extendprior /Users/zhengyifan/vega_result/Poly_r20_3free_auto_new /Users/zhengyifan/vega_result/40_180/auto_reio_40_180_3free/Poly_bbt_40_180_auto
# python analysis/plttria.py /Users/zhengyifan/vega_result/Polychord_fullcombined_extendprior /Users/zhengyifan/vega_result/Poly_r20_3free_full /Users/zhengyifan/vega_result/40_180/full_reio_40_180_Uyuk/Poly_bbt_40_180_full_Uyuk
