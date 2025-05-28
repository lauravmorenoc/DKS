import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV File ===
csv_path = "correlation_peaks.csv"  # Change to your file path if needed
df = pd.read_csv(csv_path)

# === Plot All Runs ===
plt.figure(figsize=(10, 6))

for column in df.columns:
    plt.plot(df.index, df[column], marker='o', label=column)

plt.title("Correlation Peaks Over Time")
plt.xlabel("Measurement Index")
plt.ylabel("Correlation Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
