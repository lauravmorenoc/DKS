import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
csv_path      = "correlation_peaks.csv"
pos           = 2        # 1 … 6  ← which position to show
include_noise = True     # ← set to False if you DON’T want the noise trace
noise_pos     = 7        # column that holds the single "Noise" series
# ────────────────────────────────────────────────────────────────

# 1) load CSV written with a 2-row header (first = Pos X, second = Baseline/Optimized/Noise)
df = pd.read_csv(csv_path, header=[0, 1]).dropna(how="all")   # drop padding rows

# 2) build column keys
col_baseline  = (f"Pos {pos}",   "Baseline")
col_optimized = (f"Pos {pos}",   "Optimized")
col_noise     = (f"Pos {noise_pos}", "Noise")                 # only one sub-header

# 3) plot
plt.figure(figsize=(8, 5))

if col_baseline in df.columns:
    plt.plot(df.index, df[col_baseline],   marker='o', label="Baseline")
if col_optimized in df.columns:
    plt.plot(df.index, df[col_optimized],  marker='o', label="Optimized")

if include_noise and col_noise in df.columns:
    plt.plot(df.index, df[col_noise],
             linestyle='--', marker='x', label="Noise (Pos 7)")

plt.title(f"Correlation Peaks – Position {pos}")
plt.xlabel("Measurement Index")
plt.ylabel("Correlation Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
