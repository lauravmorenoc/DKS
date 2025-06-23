import adi
import numpy as np
from scipy.signal import correlate
import os
import pandas as pd

def find_sdr_uris():
    try:
        import subprocess
        import re

        result = subprocess.run(["iio_info", "-s"], capture_output=True, text=True)
        output = result.stdout

        uri_tx = None
        uri_rx = None

        # Match lines like:
        # serial=104473... [usb:1.10.5]
        pattern = re.compile(r'serial=(\w+).*?\[(usb:[\d\.]+)\]')

        for match in pattern.finditer(output):
            serial = match.group(1)
            uri = match.group(2)

            print(f"Detected device → Serial: {serial}, URI: {uri}")

            if serial.endswith("51"):
                uri_tx = uri
                print("Assigned as Tx")
            elif serial.endswith("74"):
                uri_rx = uri
                print("Assigned as Rx")

        if not uri_tx or not uri_rx:
            raise RuntimeError(f"Could not find both required SDRs. Found TX={uri_tx}, RX={uri_rx}")

        return uri_tx, uri_rx

    except Exception as e:
        print(f"Error during SDR URI detection: {e}")
        raise

def save_corr_peaks(corr_peaks, path, pos, mode,
                    max_pos=6, window_size=20):
    """
    (pos = 1‒6 , mode = 'Baseline' | 'Optimized')

    Writes/overwrites the column ('Pos{pos}', mode) and keeps the file in the
    2-row-header format shown in the screenshot.
    """
    assert 1 <= pos <= max_pos, "pos must be 1..6"
    assert mode in ("Baseline", "Optimized")

    # --- Build the full column structure we always want --------------------
    first_level  = [f"Pos {p}" for p in range(1, max_pos + 1)]
    second_level = ["Baseline", "Optimized"]
    full_cols = pd.MultiIndex.from_product([first_level, second_level])

    # --- Load or create DataFrame -----------------------------------------
    if os.path.exists(path):
        df = pd.read_csv(path, header=[0, 1])
    else:
        df = pd.DataFrame(columns=full_cols)

    # Make sure *all* Pos/Modes are present (fill with NaNs if missing)
    for col in full_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Make sure we have enough rows
    if len(df) < window_size:
        df = df.reindex(range(window_size))

    # --- Insert / replace the data ----------------------------------------
    df[(f"Pos {pos}", mode)] = np.asarray(corr_peaks)[:window_size]

    # Keep the columns in canonical order
    df = df[full_cols]

    # --- Atomic write ------------------------------------------------------
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f"Saved → Pos {pos} / {mode}  in {path}")


# === Parameters ===
samp_rate = 5.3e5
tx_gain = 0
NumSamples = 300000
rx_lo = 5.3e9
rx_mode = "manual"
rx_gain = 0
fc0 = int(200e3)
downsample_factor = 30
averaging_factor = 5
window_size = 20
threshold_factor = 3

pos  = 4              # 1 … 6
mode = "Baseline"     # "Baseline"  or  "Optimized"


# === m-sequence ===
mseq = np.array([0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0])
amp = 1
mseq = np.where(mseq == 0, amp, -amp)
sps = 18750 // downsample_factor
mseq_upsampled = np.repeat(mseq, sps)
M_up = len(mseq_upsampled)

# === Initialize SDR ===
uri_tx, uri_rx = find_sdr_uris()
sdr_tx = adi.ad9361(uri=uri_tx)
sdr_rx = adi.ad9361(uri=uri_rx)
sdr_tx.sample_rate = sdr_rx.sample_rate = int(samp_rate)
sdr_tx.rx_rf_bandwidth = sdr_rx.rx_rf_bandwidth = int(3 * 0)
sdr_tx.rx_lo = sdr_rx.rx_lo = int(rx_lo)
sdr_tx.gain_control_mode = sdr_rx.gain_control_mode = "manual"
sdr_tx.rx_hardwaregain_chan0 = sdr_rx.rx_hardwaregain_chan0 = rx_gain
sdr_tx.rx_buffer_size = sdr_rx.rx_buffer_size = NumSamples
sdr_tx.tx_rf_bandwidth = int(3 * 0)
sdr_tx.tx_lo = int(rx_lo)
sdr_tx.tx_cyclic_buffer = True
sdr_tx.tx_hardwaregain_chan0 = tx_gain
sdr_tx.tx_buffer_size = int(2**18)

# Transmit constant BPSK pattern (all 0s)
num_symbols = 10
sps = 16
x_symbols = np.ones(num_symbols)
samples = np.repeat(x_symbols, sps)
samples_tx = samples * (2**14)
sdr_tx.tx([samples_tx, samples_tx])

# === Initial reads to let AGC stabilize (if using "slow_attack") ===
for _ in range(5):
    _ = sdr_rx.rx()

# === Take correlation measurements ===
corr_peaks = []
corr_array = []

for i in range(window_size):
    data = sdr_rx.rx()
    Rx = data[0]
    Rx = Rx[::downsample_factor]
    envelope = np.abs(Rx) / 2**12
    envelope -= np.mean(envelope)
    envelope /= np.max(envelope)
    corr = np.max(np.abs(correlate(mseq_upsampled, envelope, mode='full')) / M_up)

    corr_array.append(corr)
    if len(corr_array) > averaging_factor:
        corr_array = corr_array[-averaging_factor:]

    avg_corr = np.mean(corr_array)
    corr_peaks.append(avg_corr)

# === Save results to file ===
csv_path = "correlation_peaks.csv"
save_corr_peaks(corr_peaks, csv_path, pos, mode)
print(f"Saved to: {csv_path}")
