# Inference of reionization timeline based on Vega results

This folder contains scripts to run MCMC, build emulators, and do inference.

## Introduction of files

The python scripts in this directory can be devided into 3 parts: train and call emulators, error evaluation and visualization, and run MCMC based on emulators.

### Train and call emulators

**xhi_shape_emulator.py** is the emulator for reionization timeline, i.e. xHI(z). It has 3 different modes. Firstly you need to run `train` mode to train the emulator, with command like this:
```bash
python xhi_shape_emulator.py train \
  --sobol sobol.txt \
  --xhi-dir train_data/xHI \
  --optimize-kernel \
  --out xhi_shape_emulator.joblib
```
Check the meaning of sobol.txt and train_data/xHI below. After generating the **xhi_shape_emulator.joblib**, you can call the xHI(z) emulator for prediction. Mode `predict` is for single point prediction, i.e. for a given set of (zeta_UV, Tvir), output the neutral hydrogen fraction xHI at interested redshift z, for example:
```bash
python xhi_shape_emulator.py predict \
  --model xhi_shape_emulator.joblib \
  --z 12.345 \
  --params 35.51756401546299458 93822.94044829905033
```
While `predict-curve` mode can provide you the whole xHI(z) curve for a given set of (zeta_UV, Tvir), the output is assigned on the same z-grid as the z-file you input, call it with:
```bash
python xhi_shape_emulator.py predict-curve \
  --model xhi_shape_emulator.joblib \
  --params 35.51756401546299458 93822.94044829905033 \
  --z-file test_data/xHI/xH_concat_emu_0 \
  --out shape_pred_emu_0.txt
```


Similarly, **pmxh_shape_emulator.py** is the emulator for matter-neutral hydrogen fraction cross-power spectrum P_m,xHI(k, z). `train` mode is the first thing you need to run, with the command:
```bash
python pmxh_shape_emulator.py train \
  --sobol sobol.txt \
  --cross-dir train_data/cross \
  --optimize-kernel \
  --out pmxh_shape_emulator.joblib
```
After that you can do prediction for P_m,xHI at assigned k and z for a given set of (zeta_UV, Tvir) with `predict` mode:
```bash
python pmxh_shape_emulator.py predict \
  --model pmxh_shape_emulator.joblib \
  --z 7.08 \
  --k 0.07542991 \
  --params 35.51756401546299458 93822.94044829905033
```

In **example_emu**, we show several samples of realization gained by 21cmFAST simulation, which are used as train and test dataset in **emulator** directory. Folder **xHI** contains the example data files of xHI, the first column is the redshift z-grid, and the second column is the neutral hydrogen fraction xHI. The number n in the name **xH_concat_emu_n** represents the serial number. Folder **cross** contains the example data files of P_m,xHI(k, z), the first column is the redshift z-grid, the second column is the wavenumber k-grid, and the third column is the matter-neutral hydrogen fraction cross-power spectrum P_m,xHI, while you don't need the rest columns.

The train and test dataset are generated with 21cmFAST by Sobol sampling in the parameter space (zeta_UV, Tvir). So you also need a sobol.txt file to record the parameter values following the serial number.


### Error evaluation

After training the emulators, you need to test their performance before you further analyze. The most straight way is to show the emulator-predicted results and the true results in the same figure to do the comparison. **compare_xhi_shape_emulator.py** can do the comparison for xHI emulator, by running:
```bash
python compare_xhi_shape_emulator.py \
  --model xhi_shape_emulator.joblib \
  --sobol sobol_inner.txt \
  --test-file test_data/xHI/xH_concat_emu_0 \
  --out-dir compare_plot_shape
```
you can get this: 

And similarly, do the comparison for P_m,xHI emulator by running:
```bash
python compare_pmxh_shape_emulator.py \
  --model pmxh_shape_emulator.joblib \
  --sobol sobol_inner.txt \
  --test-file test_data/cross/cross_power_concat_emu_0 \
  --k 0.1 \
  --out-dir test_data/compare_cross
```
note that P_m,xHI is a 2D emulator, so we do the comparison at a given k.

For quantitatively evaluating the performance of the emulators, you can run **eval_xhi.py** and **eval_pmxh.py** to get RMSE, Cov68, Cov95, NLL and some other indices. The command is simple:
```bash
python eval_xhi.py --model xhi_shape_emulator.joblib --out eval_xhi.tsv
python eval_pmxh.py --model pmxh_shape_emulator.joblib --out eval_pmxh.tsv
```

And **error_xHI.py** and **error_pmxh.py** provide the Truth-Prediction scatter plot and report the R^2, based on the test dataset, by running:
```bash
python error_xHI.py \
  --model xhi_shape_emulator.joblib \
  --test-dir test_data/xHI \
  --test-glob 'xH_concat_emu_*' \
  --test-sobol sobol_inner.txt \
  --out-dir error_plot
```
```
python error_pmxh.py \
  --model pmxh_shape_emulator.joblib \
  --test-dir test_data/cross \
  --test-glob 'cross_power_concat_emu_*' \
  --test-sobol sobol_inner.txt \
  --out-dir error_plot
```


### Run MCMC and infer reionization timeline

After verifying the emulators, you can run the MCMC with them, to get the posterior distribution of (zeta_UV, Tvir). In **run_mcmc.py**, the sampled point (zeta_UV, Tvir) will be transformed into P_m,xHI by emulator, then into P_m,psi^emu, to calculate the likelihood by comparing with the P_m,psi^obs obtained by the yukawa parameters from Polychord runs.

















