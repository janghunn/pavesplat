#!/usr/bin/env python
"""Step-paired statistical analysis of two training runs (PaveSplat vs. SurfSplat, supplement section 6 / Figure 1).

Goes beyond tools/compare_runs.py (single same-step snapshot) by aggregating ALL
step-paired eval points and computing n / mean / std / SE / 95% CI / sign-test
p-value / paired effect size for every test/ and val/ metric, plus trajectory
regression, residual-mechanism correlations, and diagnostic-stability quantification.

Pairing: eval steps are identical across the two runs (same val_check_interval,
same config), so deltas are matched at EXACT common global_step values. The
faster run's extra (unpaired) evals are ignored.

Usage:
    python tools/paired_stats.py <method.wandb> <baseline.wandb> [--outdir results/paired_stats]

Outputs: markdown tables on stdout, paired_stats.csv + PNG figures in --outdir.
Reads the .wandb inputs without modifying them.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_runs import parse_series  # noqa: E402

# direction: +1 = higher is better, -1 = lower is better
METRIC_DIRECTION = {
    "psnr": +1, "ssim": +1, "lpips": -1, "drop": -1, "rise": -1,
}


def direction_of(key):
    name = key.split("/")[-1]
    if "drop" in name or "rise" in name:
        return -1
    for tok in ("lpips", "psnr", "ssim"):
        if tok in name:
            return METRIC_DIRECTION[tok]
    return +1


def paired(method, base, key):
    """Return (steps, method_vals, base_vals) at exact common steps."""
    c = dict(method.get(key, []))
    b = dict(base.get(key, []))
    steps = sorted(set(c) & set(b))
    return (np.array(steps),
            np.array([c[s] for s in steps], dtype=float),
            np.array([b[s] for s in steps], dtype=float))


def paired_stats(steps, cv, bv, key):
    """Full paired-delta statistics for one metric."""
    d = cv - bv
    n = len(d)
    if n < 2:
        return None
    mean = d.mean()
    std = d.std(ddof=1)
    se = std / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    sgn = direction_of(key)
    better = int(np.sum(sgn * d > 0))
    ties = int(np.sum(d == 0))
    ntrials = n - ties
    p_sign = stats.binomtest(better, ntrials, 0.5).pvalue if ntrials > 0 else 1.0
    # paired t-test as a secondary check
    t_stat, p_t = stats.ttest_1samp(d, 0.0)
    dz = mean / std if std > 0 else float("inf")
    return dict(metric=key, n=n, mean=mean, std=std, se=se,
                ci_lo=ci[0], ci_hi=ci[1], better=better, ties=ties,
                p_sign=p_sign, p_t=p_t, dz=dz, direction=sgn,
                reliable=(ci[0] > 0) or (ci[1] < 0))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def fmt_p(p):
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def lin_reg(x, y):
    """Slope, slope-SE, two-sided p for slope=0, intercept."""
    res = stats.linregress(x, y)
    return res.slope, res.stderr, res.pvalue, res.intercept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method")
    ap.add_argument("baseline")
    ap.add_argument("--outdir", default="results/paired_stats")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    method = parse_series(a.method)
    base = parse_series(a.baseline)
    c_latest = method["train/psnr"][-1][0]
    b_latest = base["train/psnr"][-1][0]
    print(f"# Paired analysis — method latest step {c_latest}, "
          f"base latest step {b_latest}\n")

    # ---------- A. step-paired aggregate over all eval metrics ----------
    test_keys = ["test/psnr", "test/ssim", "test/lpips"]
    val_keys = [k for k in sorted(method) if k.startswith("val/")]
    rows = []
    for key in test_keys + val_keys:
        steps, cv, bv = paired(method, base, key)
        st = paired_stats(steps, cv, bv, key)
        if st:
            st["steps"] = steps
            st["cv"], st["bv"] = cv, bv
            rows.append(st)

    # Holm correction within the val family (sign-test p)
    val_rows = [r for r in rows if r["metric"].startswith("val/")]
    if val_rows:
        adj = holm(np.array([r["p_sign"] for r in val_rows]))
        for r, pa in zip(val_rows, adj):
            r["p_sign_holm"] = pa
    for r in rows:
        r.setdefault("p_sign_holm", r["p_sign"])

    hdr = ("| metric | n | meanΔ | std | SE | 95% CI | method-better | "
           "sign-p | sign-p(Holm) | t-p | d_z | verdict |")
    sep = "|" + "---|" * 12
    print("## A. Step-paired deltas (Δ = method − base, exact same global_step)\n")
    print(hdr)
    print(sep)
    for r in rows:
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        verdict = "RELIABLE" if r["reliable"] else "noise(CI∋0)"
        arrow = "↑good" if r["direction"] > 0 else "↓good"
        print(f"| {r['metric']} ({arrow}) | {r['n']} | {r['mean']:+.4f} | "
              f"{r['std']:.4f} | {r['se']:.4f} | {ci} | "
              f"{r['better']}/{r['n'] - r['ties']} | {fmt_p(r['p_sign'])} | "
              f"{fmt_p(r['p_sign_holm'])} | {fmt_p(r['p_t'])} | "
              f"{r['dz']:+.2f} | {verdict} |")

    # late-half stability check for the headline metric
    print("\n### A2. test/* late-half (step ≥ 33023) stability check\n")
    print(hdr)
    print(sep)
    for key in test_keys:
        steps, cv, bv = paired(method, base, key)
        m = steps >= 33023
        st = paired_stats(steps[m], cv[m], bv[m], key)
        if st:
            ci = f"[{st['ci_lo']:+.4f}, {st['ci_hi']:+.4f}]"
            verdict = "RELIABLE" if st["reliable"] else "noise(CI∋0)"
            print(f"| {st['metric']} (late) | {st['n']} | {st['mean']:+.4f} | "
                  f"{st['std']:.4f} | {st['se']:.4f} | {ci} | "
                  f"{st['better']}/{st['n'] - st['ties']} | {fmt_p(st['p_sign'])} | "
                  f"— | {fmt_p(st['p_t'])} | {st['dz']:+.2f} | {verdict} |")

    # CSV digest
    csv_path = os.path.join(a.outdir, "paired_stats.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "mean_delta", "std", "se", "ci_lo", "ci_hi",
                    "method_better", "ties", "p_sign", "p_sign_holm", "p_t", "dz",
                    "direction", "reliable"])
        for r in rows:
            w.writerow([r["metric"], r["n"], r["mean"], r["std"], r["se"],
                        r["ci_lo"], r["ci_hi"], r["better"], r["ties"],
                        r["p_sign"], r["p_sign_holm"], r["p_t"], r["dz"],
                        r["direction"], r["reliable"]])
    print(f"\nCSV digest -> {csv_path}")

    # ---------- B. trajectory / trend ----------
    print("\n## B. Trajectory (test/psnr)\n")
    steps, cv, bv = paired(method, base, "test/psnr")
    d = cv - bv
    sl, sl_se, sl_p, ic = lin_reg(steps, d)
    print(f"- Δ(test/psnr) vs step regression: slope = {sl * 10000:+.4f} dB / 10k steps "
          f"(SE {sl_se * 10000:.4f}, p = {fmt_p(sl_p)}) — "
          f"{'gap CHANGING' if sl_p < 0.05 else 'no evidence the gap grows or shrinks'}")
    half = len(steps) // 2
    print(f"- first-half meanΔ = {d[:half].mean():+.4f}, "
          f"second-half meanΔ = {d[half:].mean():+.4f}")
    # absolute-level trend on the late half (log-step fit, descriptive only)
    m = steps >= steps[len(steps) // 2]
    for label, vals in (("method", cv), ("base", bv)):
        s2, _, _, i2 = lin_reg(np.log(steps[m]), vals[m])
        pred150k = s2 * np.log(150000) + i2
        print(f"- {label}: latest test/psnr = {vals[-1]:.3f} @ step {steps[-1]}; "
              f"log-step fit on late half extrapolates ~{pred150k:.2f} dB at 150k "
              f"(descriptive only)")

    # ---------- C. residual mechanism ----------
    print("\n## C. Residual mechanism (method run, train-time diag)\n")
    dq = dict(method["diag/delta_q_angle"])
    dq_steps = np.array(sorted(dq))
    dq_vals = np.array([dq[s] for s in dq_steps])
    final_mean = dq_vals[-50:].mean()
    for cut in (1400, 5000, 10000):
        idx = dq_steps <= cut
        if idx.any():
            frac = dq_vals[idx][-1] / final_mean
            print(f"- delta_q_angle at step {cut}: {dq_vals[idx][-1]:.4f} "
                  f"({frac * 100:.0f}% of late-run mean {final_mean:.4f}) ")
    # plateau check: slope over post-10k window
    m = dq_steps >= 10000
    sl, sl_se, sl_p, _ = lin_reg(dq_steps[m], dq_vals[m])
    print(f"- post-10k slope: {sl * 10000:+.5f} rad / 10k steps (p = {fmt_p(sl_p)})")
    for other in ("diag/scale_aniso_ratio", "diag/isotropic_frac"):
        ov = dict(method[other])
        common = sorted(set(dq) & set(ov))
        x = np.array([dq[s] for s in common])
        y = np.array([ov[s] for s in common])
        pr, pp = stats.pearsonr(x, y)
        srho, sp = stats.spearmanr(x, y)
        cm = np.array(common) >= 10000
        pr_l, pp_l = stats.pearsonr(x[cm], y[cm])
        print(f"- corr(delta_q_angle, {other.split('/')[-1]}): "
              f"pearson {pr:+.3f} (p={fmt_p(pp)}), spearman {srho:+.3f}; "
              f"post-10k pearson {pr_l:+.3f} (p={fmt_p(pp_l)})")
    nz = dict(method["diag/delta_q_nonzero_frac"])
    nz_vals = np.array([nz[s] for s in sorted(nz)])
    print(f"- delta_q_nonzero_frac: min {nz_vals.min():.4f}, mean {nz_vals.mean():.4f}")
    p99 = dict(method["diag/delta_q_angle_p99"])
    p99_vals = np.array([p99[s] for s in sorted(p99)])
    print(f"- delta_q_angle_p99: late-run mean {p99_vals[-50:].mean():.3f} rad")

    # ---------- D. diagnostic stability ----------
    print("\n## D. Diagnostic stability (paired at common diag steps)\n")
    for key in ("diag/frame_orth_err", "diag/frame_det_min", "diag/opacity_mean",
                "diag/scales_mul_min_frac", "diag/scale_aniso_ratio"):
        steps_d, cvd, bvd = paired(method, base, key)
        if len(steps_d) == 0:
            continue
        late = steps_d >= steps_d[-1] - 10000
        line = (f"- {key}: method late-mean {cvd[late].mean():.4g}, "
                f"base late-mean {bvd[late].mean():.4g}")
        if key == "diag/frame_orth_err":
            line += f"  (base/method ratio ~{bvd[late].mean() / cvd[late].mean():.0f}x)"
        if key == "diag/opacity_mean":
            line += (f"; run-min method {cvd.min():.4f} / base {bvd.min():.4f} "
                     f"(collapse floor was ~0.008)")
        if key == "diag/scales_mul_min_frac":
            gap = cvd - bvd
            sl, sl_se, sl_p, _ = lin_reg(steps_d[late], gap[late])
            line += (f"; gapΔ late {gap[late].mean():+.3f}, late slope "
                     f"{sl * 10000:+.4f}/10k (p={fmt_p(sl_p)})")
        print(line)
    # encoder runtime overhead
    steps_r, cr, br = paired(method, base, "test/runtime_avg_encoder")
    if len(steps_r):
        print(f"- test/runtime_avg_encoder: method/base = "
              f"{cr.mean() / br.mean():.3f}x (method overhead "
              f"{(cr.mean() / br.mean() - 1) * 100:+.1f}%)")

    # ---------- figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig 1: test psnr absolute + delta
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    steps, cv, bv = paired(method, base, "test/psnr")
    axes[0].plot(steps, cv, "o-", label="method (PaveSplat)")
    axes[0].plot(steps, bv, "s-", label="baseline (SurfSplat)")
    axes[0].axhline(28.5, ls=":", c="gray", label="saturation ref ~28.5")
    axes[0].set_ylabel("test/psnr (6474-scene fixed split)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    d = cv - bv
    axes[1].axhline(0, c="k", lw=0.8)
    axes[1].plot(steps, d, "o-", c="tab:green")
    mu, sd = d.mean(), d.std(ddof=1)
    axes[1].axhline(mu, ls="--", c="tab:green",
                    label=f"meanΔ {mu:+.3f} ± {1.96 * sd / np.sqrt(len(d)):.3f} (95% CI)")
    axes[1].fill_between(steps, mu - 1.96 * sd / np.sqrt(len(d)),
                         mu + 1.96 * sd / np.sqrt(len(d)), alpha=0.15, color="tab:green")
    axes[1].set_ylabel("Δ psnr (method−base)")
    axes[1].set_xlabel("global step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "fig_test_psnr.png"), dpi=120)
    plt.close(fig)

    # fig 2: forest plot of standardized paired deltas
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.5))
    labels, lo, hi, mid, colors = [], [], [], [], []
    for r in rows:
        scale = abs(r["mean"]) if abs(r["mean"]) > 0 else 1.0
        # plot in d_z units (CI on dz via CI/std)
        labels.append(f"{r['metric']} (n={r['n']})")
        mid.append(r["mean"] / r["std"])
        lo.append(r["ci_lo"] / r["std"])
        hi.append(r["ci_hi"] / r["std"])
        good = r["reliable"] and (r["direction"] * r["mean"] > 0)
        bad = r["reliable"] and (r["direction"] * r["mean"] < 0)
        colors.append("tab:green" if good else ("tab:red" if bad else "tab:gray"))
    y = np.arange(len(rows))[::-1]
    for yi, l, h, m_, c in zip(y, lo, hi, mid, colors):
        ax.plot([l, h], [yi, yi], c=c, lw=2)
        ax.plot(m_, yi, "o", c=c)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("paired effect size d_z = meanΔ/stdΔ (95% CI); green = reliable & method-favoring")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "fig_forest.png"), dpi=120)
    plt.close(fig)

    # fig 3: residual mechanism
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(dq_steps, dq_vals, lw=0.8, label="delta_q_angle (mean)")
    p99s = np.array(sorted(p99))
    axes[0].plot(p99s, p99_vals, lw=0.8, alpha=0.7, label="p99")
    axes[0].set_xlabel("global step")
    axes[0].set_ylabel("rad")
    axes[0].set_title("rotation-residual activation (method)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    ar = dict(method["diag/scale_aniso_ratio"])
    common = sorted(set(dq) & set(ar))
    x = np.array([dq[s] for s in common])
    yv = np.array([ar[s] for s in common])
    cmask = np.array(common) >= 10000
    axes[1].scatter(x[~cmask], yv[~cmask], s=6, alpha=0.4, label="step < 10k")
    axes[1].scatter(x[cmask], yv[cmask], s=6, alpha=0.4, label="step ≥ 10k")
    axes[1].set_xlabel("delta_q_angle")
    axes[1].set_ylabel("scale_aniso_ratio")
    axes[1].set_title("Δq vs anisotropy (temporal scatter)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "fig_residual_mechanism.png"), dpi=120)
    plt.close(fig)

    # fig 4: diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axi, key, logy in ((0, "diag/frame_orth_err", True),
                           (1, "diag/opacity_mean", False),
                           (2, "diag/scales_mul_min_frac", False)):
        s_, c_, b_ = paired(method, base, key)
        axes[axi].plot(s_, c_, lw=0.8, label="method")
        axes[axi].plot(s_, b_, lw=0.8, label="base")
        if logy:
            axes[axi].set_yscale("log")
        axes[axi].set_title(key)
        axes[axi].set_xlabel("global step")
        axes[axi].legend()
        axes[axi].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "fig_diagnostics.png"), dpi=120)
    plt.close(fig)

    print(f"\nFigures -> {a.outdir}/fig_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
