import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV File ===
csv_path = "correlation_peaks.csv"  # Update path if needed
df = pd.read_csv(csv_path)

# === Split Columns into Two Groups ===
group_A = df.iloc[:, 0:6]   # First 6 columns
group_B = df.iloc[:, 6:12]  # Next 6 columns

# === Compute Mean of Each Column ===
means_A = group_A.mean()
means_B = group_B.mean()

# === Create X-axis values ===
x = range(1, 7)

# === Plot Both Groups ===
plt.figure(figsize=(8, 5))
plt.plot(x, means_A.values, marker='o', linestyle='-', color='blue', label="Baseline")
plt.plot(x, means_B.values, marker='s', linestyle='--', color='orange', label="Optimized")

# === Formatting ===
plt.title("Average Correlation Peak per Run Group")
plt.xlabel("Pos Index")
plt.ylabel("Mean Correlation Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
