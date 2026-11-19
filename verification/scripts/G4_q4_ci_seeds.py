"""How much the percentile CI for the hidden net change moves with the bootstrap seed (numpy RNG, linear-interpolated quantiles)."""
import numpy as np
for label, fixed, broke in (("full hidden (6 fixed, 4 broken)", 6, 4), ("4-bit hidden (3 fixed, 5 broken)", 3, 5)):
    d = np.array([1.0] * fixed + [-1.0] * broke + [0.0] * (480 - fixed - broke))
    out = set()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        m = d[rng.integers(0, 480, size=(10000, 480))].mean(axis=1)
        lo, hi = np.quantile(m, [0.025, 0.975]); out.add((round(100 * lo, 2), round(100 * hi, 2)))
        frac_le0 = (m <= 0).mean()
    print(label, "CIs over 20 seeds:", sorted(out), "| share of resamples <= 0 (last seed):", round(float(frac_le0), 3))
