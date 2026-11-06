from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import numpy as np


def _apply_mplstyle() -> None:
    import matplotlib.pyplot as plt

    style_path = os.path.join(os.path.dirname(__file__), "..", "5par.mplstyle")
    if os.path.exists(style_path):
        try:
            plt.style.use(style_path)
        except Exception as e:
            print(f"Warning: Failed to apply style file {style_path}: {e}")
    else:
        print(f"Warning: Style file {style_path} not found. Using default style.")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def _lag1_rho(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 3:
        return float("nan")
    xc = x - float(np.mean(x))
    a = xc[:-1]
    b = xc[1:]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ess_from_rho1(n: int, rho1: float) -> float:
    if not np.isfinite(rho1) or abs(1.0 + rho1) < 1e-12:
        return float("nan")
    return float(n * (1.0 - rho1) / (1.0 + rho1))


def _autocorr_rho_fft(x: np.ndarray, *, max_lag: int | None) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(x.size)
    if n < 3 or not np.all(np.isfinite(x)):
        return np.asarray([np.nan], dtype=float)
    xc = x - float(np.mean(x))
    v = float(np.dot(xc, xc) / float(n))
    if not (v > 0):
        return np.asarray([np.nan], dtype=float)

    if max_lag is None:
        m = n - 1
    else:
        m = int(max(0, min(int(max_lag), n - 1)))

    nfft = 1 << int((2 * n - 1).bit_length())
    f = np.fft.rfft(xc, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[: m + 1].real
    denom = np.arange(n, n - (m + 1), -1, dtype=float)
    acov = acf / denom
    rho = acov / float(acov[0])
    return np.asarray(rho, dtype=float)


def _iat_geyer_from_rho(rho: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=float).reshape(-1)
    if rho.size < 3 or not np.isfinite(rho[0]):
        return float("nan")
    if rho[0] <= 0:
        return float("nan")

    max_pairs = (rho.size - 1) // 2
    if max_pairs <= 0:
        return 1.0

    gammas: list[float] = []
    prev = float("inf")
    for k in range(max_pairs):
        i1 = 2 * k + 1
        i2 = 2 * k + 2
        g = float(rho[i1] + rho[i2])
        if not np.isfinite(g) or g <= 0:
            break
        if g > prev:
            g = prev
        gammas.append(g)
        prev = g

    tau = 1.0 + 2.0 * float(np.sum(gammas)) if len(gammas) else 1.0
    return float(max(tau, 1.0))


def _ess_from_tau(n: int, tau: float) -> float:
    if not np.isfinite(tau) or tau <= 0:
        return float("nan")
    return float(n / float(tau))


def _chunk_means(x: np.ndarray, chunks: int) -> list[float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if chunks <= 0:
        return []
    m = n // chunks
    if m <= 0:
        return []
    out: list[float] = []
    for i in range(chunks):
        sl = slice(i * m, (i + 1) * m)
        if (i + 1) * m <= n:
            out.append(float(np.mean(x[sl])))
    return out


def _contour_levels_from_hist(H: np.ndarray, probs: Sequence[float]) -> list[float]:
    hf = np.asarray(H, dtype=float).ravel()
    m = np.isfinite(hf) & (hf > 0)
    if not np.any(m):
        return [float("nan") for _ in probs]
    hf = hf[m]
    order = np.argsort(hf)[::-1]
    s = hf[order]
    cs = np.cumsum(s)
    if cs.size == 0 or cs[-1] <= 0:
        return [float("nan") for _ in probs]
    cs = cs / cs[-1]
    levels: list[float] = []
    for p in probs:
        idx = int(np.searchsorted(cs, float(p)))
        idx = min(max(idx, 0), s.size - 1)
        levels.append(float(s[idx]))
    return levels


def _load_train_points(*, sobol_path: Path, kept_indices_path: Path, ntrain: int | None) -> np.ndarray:
    sobol = np.loadtxt(str(sobol_path), dtype=float)
    if sobol.ndim != 2 or sobol.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {sobol_path}, got shape {sobol.shape}")

    kept = np.loadtxt(str(kept_indices_path), dtype=int).reshape(-1)
    if ntrain is not None:
        kept = kept[: int(ntrain)]
    kept = np.asarray(kept, dtype=int)
    valid = (kept >= 0) & (kept < sobol.shape[0])
    kept = np.unique(kept[valid])
    if kept.size == 0:
        return np.empty((0, 2), dtype=float)

    pts = np.asarray(sobol[kept, :2], dtype=float)
    out = np.empty((pts.shape[0], 2), dtype=float)
    out[:, 0] = pts[:, 0]
    out[:, 1] = np.log10(pts[:, 1])
    return out


_PRIOR_BOUNDARY_SEGMENTS = (
    (10.0, 15.455066, 36.072697),
    (15.455066, 4.498625, 36.072697, 5.035638),
    (36.072697, 5.035638, 50.0, 5.25529),
    (50.0, 5.25529, 100.0, 5.59915),
    (100.0, 5.59915, 150.0, 5.76),
    (150.0, 5.76, 250.0, 5.925),
)


def _prior_boundary_logt_at_zeta(zeta: float | np.ndarray) -> float | np.ndarray:
    z = np.asarray(zeta, dtype=float)
    out = np.empty_like(z, dtype=float)
    segs = [
        (10.0, 15.455066, 4.498625, 36.072697, 5.035638),
        (15.455066, 15.455066, 4.498625, 36.072697, 5.035638),
        (36.072697, 36.072697, 5.035638, 50.0, 5.25529),
        (50.0, 50.0, 5.25529, 100.0, 5.59915),
        (100.0, 100.0, 5.59915, 150.0, 5.76),
        (150.0, 150.0, 5.76, 250.0, 5.925),
    ]
    for i, zi in enumerate(z.ravel()):
        y = float("nan")
        for xmin_s, x0, y0, x1, y1 in segs:
            if zi < xmin_s:
                continue
            if zi <= x1:
                if x1 - x0 < 1e-15:
                    y = float(y0)
                else:
                    t = (zi - x0) / (x1 - x0)
                    y = float(y0 + t * (y1 - y0))
                break
        out.ravel()[i] = y
    if np.ndim(zeta) == 0:
        return float(out.ravel()[0])
    return out


def _build_prior_boundary_curve(*, xlo: float, xhi: float) -> tuple[np.ndarray, np.ndarray]:
    zs: list[float] = []
    ls: list[float] = []
    for seg in _PRIOR_BOUNDARY_SEGMENTS:
        if len(seg) == 3:
            x_start, xa, xb = seg
            y0 = 4.498625
            y1 = 5.035638
            slope = (y1 - y0) / (xb - xa)
            x_end = xa
            if x_start > x_end:
                continue
            zs.append(float(x_start))
            ls.append(float(y0 + slope * (x_start - xa)))
            zs.append(float(x_end))
            ls.append(float(y0 + slope * (x_end - xa)))
        else:
            x0, y0, x1, y1 = seg
            zs.append(float(x0))
            ls.append(float(y0))
            zs.append(float(x1))
            ls.append(float(y1))
    z_arr = np.asarray(zs, dtype=float)
    l_arr = np.asarray(ls, dtype=float)
    order = np.argsort(z_arr)
    z_arr = z_arr[order]
    l_arr = l_arr[order]
    m = (z_arr >= float(xlo)) & (z_arr <= float(xhi))
    z_arr = z_arr[m]
    l_arr = l_arr[m]
    if z_arr.size < 2:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return z_arr, l_arr


def _plot_contour(
    *,
    x: np.ndarray,
    y: np.ndarray,
    out: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    bins: int,
    best_point: tuple[float, float] | None = None,
    train_points: np.ndarray | None = None,
    show_prior_boundary: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.path import Path

    _apply_mplstyle()

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    xlo, xhi = 10.0, 250.0
    ylo, yhi = 4.0, 6.0
    m = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi) & np.isfinite(x) & np.isfinite(y)
    xf = x[m]
    yf = y[m]
    prior_logt = _prior_boundary_logt_at_zeta(xf)
    inside = (yf <= np.asarray(prior_logt, dtype=float))
    if np.any(inside):
        xf = xf[inside]
        yf = yf[inside]

    H, xe, ye = np.histogram2d(xf, yf, bins=int(bins), density=True, range=[[xlo, xhi], [ylo, yhi]])
    lvl68, lvl95 = _contour_levels_from_hist(H, probs=[0.68, 0.95])
    maxH = float(np.nanmax(H)) if np.size(H) else float("nan")
    if np.isfinite(maxH) and maxH > 0:
        cap = maxH * (1.0 - 1e-12)
        if np.isfinite(lvl68):
            lvl68 = float(min(lvl68, cap))
        if np.isfinite(lvl95):
            lvl95 = float(min(lvl95, cap))
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=170)

    n_clip = 1000
    z_clip = np.linspace(xlo, xhi, n_clip)
    logt_clip = np.asarray(_prior_boundary_logt_at_zeta(z_clip), dtype=float)
    logt_clip = np.where(np.isfinite(logt_clip), logt_clip, yhi)
    logt_clip = np.minimum(logt_clip, yhi)
    y_bin_size = (yhi - ylo) / max(1, bins)
    if bool(show_prior_boundary):
        clip_eps_above = 0.0
    else:
        clip_eps_above = 2.2 * y_bin_size
    logt_clip_top = np.minimum(yhi, logt_clip + clip_eps_above)
    poly_verts = []
    poly_verts.append((xlo, ylo))
    for zi, li in zip(z_clip, logt_clip_top):
        poly_verts.append((float(zi), float(li)))
    poly_verts.append((xhi, ylo))
    poly_verts.append((xlo, ylo))
    poly_arr = np.asarray(poly_verts, dtype=float)
    clip_polygon = Polygon(poly_arr, closed=True, transform=ax.transData, visible=False)
    ax.add_patch(clip_polygon)

    if np.isfinite(lvl95) and np.isfinite(maxH) and lvl95 < maxH:
        cf95 = ax.contourf(X, Y, H, levels=[lvl95, maxH], alpha=0.20, extend="neither")
        for coll in cf95.collections:
            coll.set_clip_path(clip_polygon)
    if np.isfinite(lvl68) and np.isfinite(maxH) and lvl68 < maxH:
        cf68 = ax.contourf(X, Y, H, levels=[lvl68, maxH], alpha=0.35, extend="neither")
        for coll in cf68.collections:
            coll.set_clip_path(clip_polygon)
    if np.isfinite(lvl95) and np.isfinite(lvl68):
        lvls = sorted({float(lvl95), float(lvl68)})
        if len(lvls) >= 2 and lvls[0] < lvls[1]:
            cc = ax.contour(X, Y, H, levels=lvls, linewidths=1.4)
            for coll in cc.collections:
                coll.set_clip_path(clip_polygon)

    if bool(show_prior_boundary):
        pz, pl = _build_prior_boundary_curve(xlo=xlo, xhi=xhi)
        if pz.size >= 2:
            ax.plot(pz, pl, color="black", lw=1.8, ls="-", label="Prior boundary", zorder=4)

    if train_points is not None:
        tp = np.asarray(train_points, dtype=float)
        if tp.ndim == 2 and tp.shape[1] >= 2 and tp.shape[0] > 0:
            m_tp = (
                np.isfinite(tp[:, 0])
                & np.isfinite(tp[:, 1])
                & (tp[:, 0] >= xlo)
                & (tp[:, 0] <= xhi)
                & (tp[:, 1] >= ylo)
                & (tp[:, 1] <= yhi)
            )
            tp = tp[m_tp]
            if tp.shape[0] > 0:
                logt_tp = np.asarray(_prior_boundary_logt_at_zeta(tp[:, 0]), dtype=float)
                m_tp2 = np.isfinite(logt_tp) & (tp[:, 1] <= logt_tp)
                tp = tp[m_tp2]
                if tp.shape[0] > 0:
                    ax.scatter(
                        tp[:, 0],
                        tp[:, 1],
                        s=14,
                        c="red",
                        alpha=0.85,
                        linewidths=0.0,
                        label="Training points",
                        zorder=5,
                    )

    if best_point is not None:
        bx, by = float(best_point[0]), float(best_point[1])
        if (
            np.isfinite(bx)
            and np.isfinite(by)
            and bx >= xlo
            and bx <= xhi
            and by >= ylo
            and by <= yhi
        ):
            logt_at_bx = float(_prior_boundary_logt_at_zeta(bx))
            if np.isfinite(logt_at_bx) and by <= logt_at_bx:
                ax.scatter(
                    bx,
                    by,
                    marker="*",
                    s=35,
                    c="blue",
                    linewidths=1.8,
                    label="Max-likelihood",
                    zorder=6,
                )
                dy_gap = logt_at_bx - by
                zetas_nbh = np.array([bx - 1e-4, bx + 1e-4], dtype=float)
                logt_nbh = np.asarray(_prior_boundary_logt_at_zeta(zetas_nbh), dtype=float)
                slope = (logt_nbh[1] - logt_nbh[0]) / (zetas_nbh[1] - zetas_nbh[0])
                NEAR_GAP_LOGT = 0.005
                if 0.0 <= dy_gap < NEAR_GAP_LOGT and xf.size > 0:
                    dx_half = max(0.07, min(0.30, 1.6 * max(abs(dy_gap / max(slope, 1e-12)), 0.005)))
                    dy_below = max(0.0014, 3.0 * max(dy_gap, 1e-4))
                    dy_above = max(0.0005, 1.2 * max(dy_gap, 1e-4))
                    zoom_x0 = max(xlo, float(bx - dx_half))
                    zoom_x1 = min(xhi, float(bx + dx_half))
                    b_left = float(_prior_boundary_logt_at_zeta(zoom_x0))
                    b_right = float(_prior_boundary_logt_at_zeta(zoom_x1))
                    y_mid_line = 0.5 * (b_left + b_right)
                    y_span_line = abs(b_right - b_left)
                    pad_y = max(0.0008, 0.07 * y_span_line, 2.4 * max(dy_gap, 1e-4))
                    zoom_y0 = max(ylo, float(min(by - dy_below, y_mid_line - pad_y)))
                    zoom_y1 = min(yhi, float(max(b_right + dy_above, y_mid_line + pad_y)))
                    if zoom_x1 - zoom_x0 > 0.005 and zoom_y1 - zoom_y0 > 0.001:
                        zoom_ax = ax.inset_axes([0.047, 0.690, 0.280, 0.260])
                        pad_x_nbh = 1.10 * (zoom_x1 - zoom_x0)
                        pad_y_nbh = 1.10 * (zoom_y1 - zoom_y0)
                        mz = (
                            (xf >= zoom_x0 - pad_x_nbh)
                            & (xf <= zoom_x1 + pad_x_nbh)
                            & (yf >= zoom_y0 - pad_y_nbh)
                            & (yf <= zoom_y1 + pad_y_nbh)
                        )
                        xz = np.asarray(xf[mz], dtype=float)
                        yz = np.asarray(yf[mz], dtype=float)
                        zz_clip: np.ndarray | None = None
                        zlogt_clip: np.ndarray | None = None
                        if xz.size >= 10:
                            from scipy.stats import gaussian_kde

                            zbins_x = max(40, min(110, int(2.6 * bins * (zoom_x1 - zoom_x0) / (xhi - xlo))))
                            zbins_y = max(40, min(110, int(2.6 * bins * (zoom_y1 - zoom_y0) / (yhi - ylo))))
                            Hz, xze, yze = np.histogram2d(
                                xz,
                                yz,
                                bins=[zbins_x, zbins_y],
                                density=True,
                                range=[[zoom_x0, zoom_x1], [zoom_y0, zoom_y1]],
                            )
                            zlvl68, zlvl95 = _contour_levels_from_hist(Hz, probs=[0.68, 0.95])
                            Hz_max = float(np.nanmax(Hz)) if np.size(Hz) else float("nan")
                            if np.isfinite(Hz_max) and Hz_max > 0:
                                Hcap = Hz_max * (1.0 - 1e-12)
                                if np.isfinite(zlvl68):
                                    zlvl68 = float(min(zlvl68, Hcap))
                                if np.isfinite(zlvl95):
                                    zlvl95 = float(min(zlvl95, Hcap))
                            vals = np.vstack([xz, yz])
                            kde_auto = gaussian_kde(vals, bw_method="scott")
                            scott_factor = float(getattr(kde_auto, "factor", 1.0) or 1.0)
                            bw = 0.85 * scott_factor
                            kde = gaussian_kde(vals, bw_method=bw)
                            nkx = 200
                            nky = 200
                            xgk = np.linspace(zoom_x0, zoom_x1, nkx)
                            ygk = np.linspace(zoom_y0, zoom_y1, nky)
                            Xgk, Ygk = np.meshgrid(xgk, ygk, indexing="ij")
                            pts_kde = np.vstack([Xgk.ravel(), Ygk.ravel()])
                            Zk = kde(pts_kde).reshape(Xgk.shape)
                            m_valid = np.isfinite(Zk) & (Zk > 0)
                            nz_clip = 700
                            zz_clip = np.linspace(zoom_x0, zoom_x1, nz_clip)
                            zlogt_clip = np.asarray(_prior_boundary_logt_at_zeta(zz_clip), dtype=float)
                            zlogt_clip = np.where(np.isfinite(zlogt_clip), zlogt_clip, zoom_y1)
                            zlogt_clip = np.minimum(zlogt_clip, zoom_y1)
                            z_y_bin_size = (zoom_y1 - zoom_y0) / max(1, zbins_y)
                            if bool(show_prior_boundary):
                                z_clip_eps_above = 0.0
                            else:
                                z_clip_eps_above = 2.5 * z_y_bin_size
                            zlogt_clip_top = np.minimum(zoom_y1, zlogt_clip + z_clip_eps_above)
                            zpoly_verts: list[tuple[float, float]] = []
                            zpoly_verts.append((zoom_x0, zoom_y0))
                            for zi_, li_ in zip(zz_clip, zlogt_clip_top):
                                zpoly_verts.append((float(zi_), float(li_)))
                            zpoly_verts.append((zoom_x1, zoom_y0))
                            zpoly_verts.append((zoom_x0, zoom_y0))
                            zpoly_arr = np.asarray(zpoly_verts, dtype=float)
                            zclip_poly = Polygon(zpoly_arr, closed=True, transform=zoom_ax.transData, visible=False)
                            zoom_ax.add_patch(zclip_poly)
                            zlvl68_z = float("nan")
                            zlvl95_z = float("nan")
                            zmax_draw = float("nan")
                            if np.any(m_valid) and np.isfinite(zlvl68) and np.isfinite(zlvl95):
                                Zk_valid = Zk[m_valid]
                                zmaxK = float(np.max(Zk_valid)) if Zk_valid.size else float("nan")
                                if np.isfinite(zmaxK) and zmaxK > 0 and np.isfinite(Hz_max) and Hz_max > 0:
                                    Zk = Zk * (Hz_max / zmaxK)
                                zlvl68_z = float(zlvl68)
                                zlvl95_z = float(zlvl95)
                                zmax_draw = float(Hz_max) if (np.isfinite(Hz_max) and Hz_max > 0) else (float(np.nanmax(Zk)) if np.size(Zk) else float("nan"))
                                if np.isfinite(zlvl95_z) and np.isfinite(zmax_draw) and zlvl95_z < zmax_draw:
                                    zcf95 = zoom_ax.contourf(Xgk, Ygk, Zk, levels=[zlvl95_z, zmax_draw], alpha=0.22, extend="neither")
                                    for coll in zcf95.collections:
                                        coll.set_clip_path(zclip_poly)
                                if np.isfinite(zlvl68_z) and np.isfinite(zmax_draw) and zlvl68_z < zmax_draw:
                                    zcf68 = zoom_ax.contourf(Xgk, Ygk, Zk, levels=[zlvl68_z, zmax_draw], alpha=0.40, extend="neither")
                                    for coll in zcf68.collections:
                                        coll.set_clip_path(zclip_poly)
                                if np.isfinite(zlvl95_z) and np.isfinite(zlvl68_z) and zlvl95_z < zlvl68_z:
                                    zlvls = sorted([float(zlvl95_z), float(zlvl68_z)])
                                    zcc = zoom_ax.contour(Xgk, Ygk, Zk, levels=zlvls, linewidths=1.1)
                                    for coll in zcc.collections:
                                        coll.set_clip_path(zclip_poly)
                        zoom_ax.set_xlim(zoom_x0, zoom_x1)
                        zoom_ax.set_ylim(zoom_y0, zoom_y1)
                        from matplotlib.ticker import MaxNLocator

                        zoom_ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2, steps=[1, 2, 2.5, 5, 10]))
                        zoom_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2, steps=[1, 2, 2.5, 5, 10]))
                        zoom_ax.tick_params(axis="both", which="both", labelsize=6, pad=1.0, length=2.0, width=0.7)
                        for lbl in zoom_ax.get_xticklabels():
                            lbl.set_fontsize(6)
                        for lbl in zoom_ax.get_yticklabels():
                            lbl.set_fontsize(6)
                        for spine in zoom_ax.spines.values():
                            spine.set_linewidth(0.9)
                        zoom_ax.set_facecolor("white")
                        if zz_clip is not None and zlogt_clip is not None:
                            zoom_ax.plot(
                                zz_clip,
                                zlogt_clip,
                                color="black",
                                lw=2.4,
                                ls="-",
                                zorder=9,
                                alpha=1.0,
                                solid_capstyle="round",
                                solid_joinstyle="round",
                                antialiased=True,
                                clip_on=False,
                            )
                        if train_points is not None:
                            tpz = np.asarray(train_points, dtype=float)
                            if tpz.ndim == 2 and tpz.shape[1] >= 2 and tpz.shape[0] > 0:
                                m_tpz = (
                                    np.isfinite(tpz[:, 0])
                                    & np.isfinite(tpz[:, 1])
                                    & (tpz[:, 0] >= zoom_x0)
                                    & (tpz[:, 0] <= zoom_x1)
                                    & (tpz[:, 1] >= zoom_y0)
                                    & (tpz[:, 1] <= zoom_y1)
                                )
                                tpz = tpz[m_tpz]
                                if tpz.shape[0] > 0:
                                    logt_tpz = np.asarray(_prior_boundary_logt_at_zeta(tpz[:, 0]), dtype=float)
                                    m_tpz2 = np.isfinite(logt_tpz) & (tpz[:, 1] <= logt_tpz)
                                    tpz = tpz[m_tpz2]
                                    if tpz.shape[0] > 0:
                                        zoom_ax.scatter(
                                            tpz[:, 0],
                                            tpz[:, 1],
                                            s=20,
                                            c="red",
                                            alpha=0.95,
                                            linewidths=0.0,
                                            zorder=10,
                                        )
                        zoom_ax.scatter(
                            bx,
                            by,
                            marker="*",
                            s=35,
                            c="blue",
                            linewidths=2.3,
                            zorder=11,
                            alpha=1.0,
                        )
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        leg = ax.legend(
            loc="lower left",
            fontsize=12,
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            facecolor="white",
        )
        leg.set_zorder(15)
    if xlabel is not None and len(str(xlabel)) > 0:
        ax.set_xlabel(
            r"$\zeta_{\rm UV}$"
            if "zeta" in str(xlabel).lower() or "ζ" in str(xlabel)
            else xlabel,
            fontsize=14,
        )
    if ylabel is not None and len(str(ylabel)) > 0:
        ylab_lc = str(ylabel).lower()
        if "tvir" in ylab_lc or "t_vir" in ylab_lc or "log10t" in ylab_lc or "log_10" in ylab_lc or "tvir" in ylab_lc or "vir" in ylab_lc or "logt" in ylab_lc:
            lbl = r"$\log_{10}\,T_{\rm Vir}^{\rm min}$"
        else:
            lbl = ylabel
        ax.set_ylabel(lbl, fontsize=14)
    if title is not None and len(str(title).strip()) > 0 and False:
        ax.set_title(title, fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12, pad=5.0, length=6.0, width=1.2)
    ax.tick_params(axis="both", which="minor", labelsize=10, pad=3.0, length=3.5, width=0.9)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))


def _plot_trace(*, x: np.ndarray, y: np.ndarray, out: Path, xlabel: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_mplstyle()

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.8), dpi=150, sharex=True)
    axes[0].plot(np.asarray(x, dtype=float), lw=0.8)
    xlab_lc = str(xlabel).lower()
    if "zeta" in xlab_lc or "ζ" in str(xlabel):
        xl = r"$\zeta_{\rm UV}$"
    else:
        xl = xlabel
    axes[0].set_ylabel(xl, fontsize=13)
    axes[0].tick_params(axis="both", which="major", labelsize=11, pad=4.0, length=5.0, width=1.0)
    axes[0].grid(alpha=0.25)
    axes[1].plot(np.asarray(y, dtype=float), lw=0.8)
    ylab_lc = str(ylabel).lower()
    if "tvir" in ylab_lc or "t_vir" in ylab_lc or "log10t" in ylab_lc or "log_10" in ylab_lc or "logt" in ylab_lc or "vir" in ylab_lc:
        yl = r"$\log_{10}\,T_{\rm Vir}^{\rm min}$"
    else:
        yl = ylabel
    axes[1].set_ylabel(yl, fontsize=13)
    axes[1].set_xlabel("sample index", fontsize=13)
    axes[1].tick_params(axis="both", which="major", labelsize=11, pad=4.0, length=5.0, width=1.0)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="check_chain")
    p.add_argument("--in", dest="in_paths", nargs="+", default=["mcmc/mh_zeta_tvir_chain.npz"])
    p.add_argument("--out-dir", default="mcmc")
    p.add_argument("--prefix", default=None)
    p.add_argument("--bins", type=int, default=70)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--labels", nargs=2, default=["zeta_UV", "log10Tvir"])
    p.add_argument("--no-iat", action="store_true", default=False)
    p.add_argument("--iat-max-lag", type=int, default=200000)
    p.add_argument("--no-trace", action="store_true", default=False)
    p.add_argument("--no-contour", action="store_true", default=False)
    p.add_argument("--show-train-points", action="store_true", default=False)
    p.add_argument("--show-prior-boundary", action="store_true", default=False)
    p.add_argument("--sobol", default="sobol.txt")
    p.add_argument("--kept-indices", default="train_data/kept_indices.txt")
    p.add_argument("--ntrain", type=int, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    in_paths = [Path(str(p)) for p in (args.in_paths or [])]
    if len(in_paths) == 0:
        raise SystemExit("--in requires at least 1 path")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    logps: list[np.ndarray] = []
    ns: list[int] = []
    accs: list[float] = []

    print("files:")
    for p in in_paths:
        if not p.exists():
            raise SystemExit(f"Not found: {p}")
        payload = _load_npz(p)
        if "chain" not in payload:
            raise SystemExit(f"NPZ missing 'chain': {p}")
        chain = np.asarray(payload["chain"], dtype=float)
        if chain.ndim != 2 or chain.shape[1] < 2:
            raise SystemExit(f"Bad chain shape in {p}: {chain.shape}")
        n = int(chain.shape[0])
        acc_rate = float(np.asarray(payload["acc_rate"])) if "acc_rate" in payload else float("nan")
        logp = np.asarray(payload["logp"], dtype=float) if "logp" in payload else None
        if logp is not None and logp.ndim != 1:
            logp = logp.reshape(-1)
        print(f"- {p}  N_saved={n}  acc_rate={acc_rate}")
        ns.append(n)
        accs.append(acc_rate)
        if n > 0:
            xs.append(chain[:, 0])
            ys.append(chain[:, 1])
            if logp is not None and logp.size == n:
                logps.append(logp)
        else:
            if "last_x" in payload:
                last_x = np.asarray(payload["last_x"], dtype=float).reshape(-1)
                if last_x.size >= 2:
                    print(f"  last_x: {last_x[:2]}")
            if "last_logp" in payload:
                print(f"  last_logp: {float(np.asarray(payload['last_logp']).reshape(()))}")

    if len(xs) == 0:
        raise SystemExit("All chains are empty.")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    n = int(x.size)
    best_point: tuple[float, float] | None = None
    if len(logps) > 0:
        logp_all = np.concatenate(logps, axis=0)
        if logp_all.size == n:
            m_finite = np.isfinite(logp_all)
            if np.any(m_finite):
                xl = x[m_finite]
                yl = y[m_finite]
                ll = logp_all[m_finite]
                idx = int(np.argmax(ll))
                best_point = (float(xl[idx]), float(yl[idx]))
                print(
                    f"max-logp: {float(ll[idx]):.6f}  "
                    f"@ ({args.labels[0]}={best_point[0]:.6f}, "
                    f"{args.labels[1]}={best_point[1]:.6f})"
                )
    rho1_x = _lag1_rho(x)
    rho1_y = _lag1_rho(y)
    ess_x = _ess_from_rho1(n, rho1_x)
    ess_y = _ess_from_rho1(n, rho1_y)
    tau_x = float("nan")
    tau_y = float("nan")
    ess_tau_x = float("nan")
    ess_tau_y = float("nan")
    if not bool(args.no_iat):
        rho_x = _autocorr_rho_fft(x, max_lag=int(args.iat_max_lag))
        rho_y = _autocorr_rho_fft(y, max_lag=int(args.iat_max_lag))
        tau_x = _iat_geyer_from_rho(rho_x)
        tau_y = _iat_geyer_from_rho(rho_y)
        ess_tau_x = _ess_from_tau(n, tau_x)
        ess_tau_y = _ess_from_tau(n, tau_y)

    qx = np.quantile(x, [0.16, 0.5, 0.84])
    qy = np.quantile(y, [0.16, 0.5, 0.84])
    mx = _chunk_means(x, int(args.chunks))
    my = _chunk_means(y, int(args.chunks))

    print("N_saved_total:", n)
    print("q16/50/84", args.labels[0] + ":", qx)
    print("q16/50/84", args.labels[1] + ":", qy)
    print("chunk means", args.labels[0] + ":", mx)
    print("chunk means", args.labels[1] + ":", my)
    print("lag1 rho", args.labels[0] + ":", rho1_x)
    print("lag1 rho", args.labels[1] + ":", rho1_y)
    print("approx ESS", args.labels[0] + ":", ess_x)
    print("approx ESS", args.labels[1] + ":", ess_y)
    if not bool(args.no_iat):
        print("tau_int", args.labels[0] + ":", tau_x)
        print("tau_int", args.labels[1] + ":", tau_y)
        print("ESS(tau)", args.labels[0] + ":", ess_tau_x)
        print("ESS(tau)", args.labels[1] + ":", ess_tau_y)

    out_dir = Path(str(args.out_dir))
    if args.prefix:
        prefix = str(args.prefix)
    else:
        prefix = "combined" if len(in_paths) > 1 else in_paths[0].stem
    if not bool(args.no_trace):
        out_trace = out_dir / f"{prefix}_trace.png"
        _plot_trace(x=x, y=y, out=out_trace, xlabel=str(args.labels[0]), ylabel=str(args.labels[1]))
        print("saved:", out_trace)
    if not bool(args.no_contour):
        out_contour = out_dir / f"{prefix}_contour.png"
        train_points = None
        if bool(args.show_train_points):
            train_points = _load_train_points(
                sobol_path=Path(str(args.sobol)),
                kept_indices_path=Path(str(args.kept_indices)),
                ntrain=(None if args.ntrain is None else int(args.ntrain)),
            )
        _plot_contour(
            x=x,
            y=y,
            out=out_contour,
            xlabel=str(args.labels[0]),
            ylabel=str(args.labels[1]),
            title="Posterior Contours (68% / 95%)",
            bins=int(args.bins),
            best_point=best_point,
            train_points=train_points,
            show_prior_boundary=bool(args.show_prior_boundary),
        )
        print("saved:", out_contour)


if __name__ == "__main__":
    main()
