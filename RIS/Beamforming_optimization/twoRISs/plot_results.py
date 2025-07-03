import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────────
csv_path      = "correlation_peaks.csv"
pos           = 3       # 1 … 6  ← which position to show
include_noise = True     # ← set to False if you DON’T want the noise trace
noise_pos     = 10        # column that holds the single "Noise" series
# ────────────────────────────────────────────────────────────────

# 1) load CSV written with a 2-row header (first = Pos X, second = Baseline/Optimized/Noise)
df = pd.read_csv(csv_path, header=[0, 1]).dropna(how="all")   # drop padding rows

# 2) build column keys
col_baseline  = (f"Pos {pos}",   "Baseline")
col_optimized = (f"Pos {pos}",   "NewScheme")
col_noise     = (f"Pos {noise_pos}", "Noise")                 # only one sub-header

# 3) plot
plt.figure(figsize=(8, 5))

if col_baseline in df.columns:
    plt.plot(df.index, df[col_baseline],   marker='o', label="Baseline")
if col_optimized in df.columns:
    plt.plot(df.index, df[col_optimized],  marker='o', label="NewScheme")

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

ratio = 10*np.log10(df[col_optimized]/df[col_baseline])

plt.figure(figsize=(8, 5))
plt.plot(df.index, ratio, 'o-')
plt.plot(df.index, [np.mean(ratio)]*len(ratio), 'x-')

plt.xlabel("Measurement index")
plt.ylabel("Correlation Amplitude Ratio (dB)")
plt.title("Baseline vs. New scheme correlation peaks")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




