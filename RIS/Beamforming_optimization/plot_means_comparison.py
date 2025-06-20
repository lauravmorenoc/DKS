import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV File ===
csv_path = "correlation_peaks.csv"  # Update path if needed
df = pd.read_csv(csv_path)

# === Split Columns into Two Groups ===
#group_A = df.iloc[:, 0:6]   # First 6 columns
#group_B = df.iloc[:, 6:12]  # Next 6 columns
group_A_indices = [2, 4, 6, 8, 10, 12, 14]  # columns 3, 5, ..., 15
group_B_indices = [3, 5, 7, 9, 11, 13, 15]  # columns 4, 6, ..., 16

# Select the columns
group_A = df.iloc[:, group_A_indices]
group_B = df.iloc[:, group_B_indices]

# === Compute Mean of Each Column ===
means_A = group_A.mean()
means_B = group_B.mean()

# === Create X-axis values ===
x = range(1, 8)

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
