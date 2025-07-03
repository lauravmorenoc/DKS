import re
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────── read file ─────────────────────────────
df = pd.read_csv("correlation_peaks.csv", header=[0, 1])   # two-row header

# ─────────────────── split columns by Baseline / NewScher ────────────
baseline_cols = [c for c in df.columns if c[1].lower().startswith("baseline")]
new_cols      = [c for c in df.columns if c[1].lower().startswith("newsche")]

# helper → extract the integer after "Pos "
pos_id = lambda col: int(re.search(r"Pos (\d+)", col[0]).group(1))

# sort columns so Pos 1, Pos 2, … appear in order
baseline_cols.sort(key=pos_id)
new_cols.sort(key=pos_id)

# ─────────────────── compute column means in position order ──────────
baseline_means = [df[c].mean() for c in baseline_cols]
new_means      = [df[c].mean() for c in new_cols]

max_mean=max(new_means)
baseline_means=baseline_means/max_mean
new_means=new_means/max_mean

positions = list(range(1, len(baseline_means) + 1))   # x-axis

# ─────────────────────────────── plot ────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(positions, baseline_means, 'o-', label="Baseline")
plt.plot(positions, new_means,      's-', label="New scheme")

plt.xlabel("Position")
plt.ylabel("Mean correlation peak")
plt.title("Baseline vs. New-scheme correlation (per position)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
