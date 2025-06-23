import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
csv_path = "correlation_peaks.csv"
pos = 4                       # 1 … 6   ← change me
# ------------------------------------------------------------

# ❶  Read the CSV that has a 2-row header (Pos / Mode)
df = pd.read_csv(csv_path, header=[0, 1])

# ❷  In case some rows are still empty NaNs (file was padded earlier)
df = df.dropna(how="all")

# ❸  Select the two columns that correspond to this position
#     - MultiIndex key is ("Pos {n}",  "Baseline"/"Optimized")
col_baseline  = (f"Pos {pos}", "Baseline")
col_optimized = (f"Pos {pos}", "Optimized")

# ❹  Plot
plt.figure(figsize=(8, 5))

if col_baseline in df.columns:
    plt.plot(df.index, df[col_baseline], marker='o', label="Baseline")
if col_optimized in df.columns:
    plt.plot(df.index, df[col_optimized], marker='o', label="Optimized")

plt.title(f"Correlation Peaks – Position {pos}")
plt.xlabel("Measurement Index")
plt.ylabel("Correlation Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
