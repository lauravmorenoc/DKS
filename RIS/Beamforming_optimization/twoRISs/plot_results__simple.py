import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- file names -------------------------------------------------
file_baseline = "correlation_baseline.csv"
file_new      = "correlation_newscheme.csv"

# ---- load the *first* column from each file --------------------
y_baseline = pd.read_csv(file_baseline).iloc[:, 0].values
y_new      = pd.read_csv(file_new).iloc[:, 0].values

ratio = 10*np.log10(y_new/y_baseline)

# ---- x-axis (sample index) -------------------------------------
x_baseline = range(len(y_baseline))
x_new      = range(len(y_new))

# ---- plot ------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(x_baseline, y_baseline, 'o-', label="Baseline")
plt.plot(x_new,       y_new,      's-', label="New scheme")

plt.xlabel("Measurement index")
plt.ylabel("Correlation peak")
plt.title("Baseline vs. New scheme correlation peaks")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# ============================================

plt.figure(figsize=(8, 5))
plt.plot(x_baseline, ratio, 'o-')
plt.plot(x_baseline, [np.mean(ratio)]*len(ratio), 'x-')

plt.xlabel("Measurement index")
plt.ylabel("Correlation Amplitude Ratio (dB)")
plt.title("Baseline vs. New scheme correlation peaks")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

