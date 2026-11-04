# Reanalysis of DESI results with our reionization model
This folder contains scripts to visualize and analyze the results of Vega for our runs.

## Introduction of files
**5par.mplstyle** is the style file for figures.

**get_best_maximum_likelihood.py** returns the point with the maximum likelihood from the input Polychord chain. The input file is written in hard coded format within the file.

**param_comp.py** does the comparison between the parameters from different runs, for example, base DESI DR1 results and our reproduced DESI DR1 results, by showing them in the same figure. All the parameter values are written in hard coded format.

**plttria_weight.py** makes the contour corner plot for the posterior distribution of free parameters. There are 4 alternative modes: conservative_fid, conservative_yuk, optimistic_fid and optimistic_yuk. Conservative_* modes correspond to lyalya x lyalya autocorrelation, and optimistic modes correspond to all combined correlations. *_fid modes correspond to fiducial model without reionization effect, while *_yuk modes correspond to our reionization model, which induces 3 more reionization parameters. Call it with the command `python analysis/plttria_weight.py /your/path/to/polychord_chains mode`, for example:
```
python analysis/plttria_weight.py /Users/zhengyifan/vega_result/40_180/auto_reio_40_180_3free/Poly_bbt_40_180_auto conservative_yuk
```

**replot_from_fits.py** makes the standard wedge plot from the output file of Minimizer, while it can show two different fit results for comparison. Call it with the command `python analysis/replot_from_fits.py -wedge --fit1 /path/to/fit1.fits --fit2 /path/to/fit2.fits --config path/to/ini_file.ini --corr corr_type --plot-rmin n`, for example:
```
python analysis/replot_from_fits.py -wedge --fit1 /work/zhuxing/vega_result/0318_fid_auto_20-180.fits --fit2 /work/zhuxing/vega_result/0318_yuk_auto_20-180.fits --config analysis/main_minimizer.ini --corr lyaxlya --plot-rmin 20
```
