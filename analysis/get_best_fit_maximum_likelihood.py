import numpy as np

def get_best_fit_row(filename):
    data = np.loadtxt(filename)
    # assuming column 1 is chi2
    # Column 1 = -2 * log(Likelihood)
    best_idx = np.argmin(data[:, 1]) # maximum likelihood point
    # extract only the model parameters
    params = data[best_idx, 2:]
    # saving the ln likelihood
    logL = -0.5 * data[best_idx, 1]
    return params, logL

yukawa_params, yukawa_logL = get_best_fit_row('./lya_auto_yuk3_extendedprior/Polychord_auto_extendprior.txt')
fiducial_params, fiducial_logL = get_best_fit_row('./lya_auto_fid/lyaxlya_lyaxlyb_s3.txt')


print("For the auto Lya: \n")
print("Yukawa maximum likelihood Parameters:")
for i, val in enumerate(yukawa_params):
    print(f"p{i+1}: {val:.10e}")
print(f"Max ln L: {yukawa_logL}")

print("\nFiducial maximum likelihood Parameters:")
for i, val in enumerate(fiducial_params):
    print(f"p{i+1}: {val:.10e}")
print(f"Max ln L: {fiducial_logL}")


# next let's do the full combined

yukawa_params, yukawa_logL = get_best_fit_row('./full_combine_yuk3_extendedprior/Polychord_fullcombined_extendprior.txt')
fiducial_params, fiducial_logL = get_best_fit_row('./full_combine_fid/Polychord_fullcombine_fid.txt')


print("###\n")
print("For the full combined: \n")
print("Yukawa maximum likelihood Parameters:")
for i, val in enumerate(yukawa_params):
    print(f"p{i+1}: {val:.10e}")
print(f"Max ln L: {yukawa_logL}")

print("\nFiducial maximum likelihood Parameters:")
for i, val in enumerate(fiducial_params):
    print(f"p{i+1}: {val:.10e}")
print(f"Max ln L: {fiducial_logL}")

