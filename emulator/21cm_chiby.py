import numpy as np
import os
import re
import matplotlib.pyplot as plt



"""
    Chibi code 
"""


plt.style.use('mcmc/perf.mplstyle')

data_direc = 'origin_data/extend_xHI'

fig = plt.figure(figsize=(4.7, 3))
ax = fig.add_subplot(111)

total = 0
matched = 0
plotted = 0
for name in os.listdir(data_direc):
    total += 1
    if name.startswith("."):
        continue
    if re.match(r"^xH_concat_emu_\d+$", name) is None:
        continue
    matched += 1
    filename = os.path.join(data_direc, name)
    if not os.path.isfile(filename):
        continue
    temp = np.loadtxt(filename)
    if temp.ndim != 2 or temp.shape[1] < 2:
        continue
    z = temp[:, 0]
    xhi = temp[:, 1]
    m = np.isfinite(z) & np.isfinite(xhi)
    if not np.any(m):
        continue
    z = z[m]
    xhi = xhi[m]
    target = 5.37
    idx = np.where(np.isclose(z, target, rtol=0.0, atol=5e-4))[0]
    if idx.size == 0:
        continue
    if float(xhi[int(idx[0])]) < 0.05:
        ax.plot(temp[:, 0], temp[:, 1], "-", color="lightgray", lw=0.5)
        plotted += 1
        
ax.set_xlim(5,18)
plt.tight_layout()
print(f"total_entries={total} matched_files={matched} plotted={plotted}")
plt.savefig('xHIsim_chibi.pdf')
