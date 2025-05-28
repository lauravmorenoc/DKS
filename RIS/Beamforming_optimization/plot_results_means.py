import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV File ===
csv_path = "correlation_peaks.csv"  # Update path if needed
df = pd.read_csv(csv_path)

# === Compute Means for Each Run ===
column_means = df.mean()

# === Plot Mean Values ===
plt.figure(figsize=(8, 5))
plt.plot(column_means.index, column_means.values, marker='o', linestyle='-', color='blue')

plt.title("Average Correlation Peak per Run")
plt.xlabel("Run")
plt.ylabel("Mean Correlation Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
