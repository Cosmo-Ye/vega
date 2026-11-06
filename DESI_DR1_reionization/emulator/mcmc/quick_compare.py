import numpy as np
import matplotlib.pyplot as plt

#比较两个xHI(z)的不同

def load_band(path):
    data = np.loadtxt(path, dtype=float)
    z = data[:, 0]
    c1, c2, c3 = data[:, 1], data[:, 2], data[:, 3]

    candidates = [
        (c1, c3, c2),
        (c1, c2, c3),
        (c2, c3, c1),
        (c2, c1, c3),
        (c3, c2, c1),
        (c3, c1, c2),
    ]
    best = None
    best_score = -1
    for q16, q50, q84 in candidates:
        m = np.isfinite(q16) & np.isfinite(q50) & np.isfinite(q84)
        if not np.any(m):
            score = -1
        else:
            score = np.sum((q16[m] <= q50[m]) & (q50[m] <= q84[m]))
        if score > best_score:
            best_score = score
            best = (q16, q50, q84)

    if best is None or best_score <= 0:
        raise ValueError(f"Cannot infer q16/q50/q84 order in {path}")
    q16, q50, q84 = best
    return z, q16, q50, q84

z1, q16_1, q50_1, q84_1 = load_band("mcmc/xhi_extwhole.txt")
zw, q16_w, q50_w, q84_w = load_band("mcmc/xhi_extwhole_desicos.txt")

fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=160)

ax.fill_between(zw, q16_w, q84_w, alpha=0.20, color="C0", linewidth=0.0, label="21cmFAST 16–84%")
ax.plot(zw, q50_w, color="C0", lw=2.0, label="21cmFAST median")

ax.fill_between(z1, q16_1, q84_1, alpha=0.20, color="C3", linewidth=0.0, label="DESI 16–84%")
ax.plot(z1, q50_1, color="C3", lw=2.0, label="DESI median")

ax.set_xlabel("z")
ax.set_ylabel("xHI")
ax.set_ylim(-0.02, 1.02)
ax.set_xlim(5, 12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
fig.tight_layout()

plt.show()
fig.savefig("mcmc/xhi_differentcos.png")