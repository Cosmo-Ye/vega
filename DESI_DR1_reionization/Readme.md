In this directory, there are our own scripts to do the DESI DR1 reanalysis work with reionization model.

Notice that, to include the reionization relics effect, we did some modifications in **vega/vega/power_spectrum.py**, adding functions **compute_reionization_yukawa**, **complete_reionization** and **complete_reionization_cross** to inject P_m,psi to Pk.

The directory **analysis** is created to visualize and analyze the results of Vega for our runs, i.e., the reproduced DESI DR1 results and new results obtained by adding reionization effect.

The directory **emulator** is created to do the inference of reionization timeline based on the Vega runs.

Besides, here are several config files and job scripts which are used to run the key steps of workflow on the cluster.
