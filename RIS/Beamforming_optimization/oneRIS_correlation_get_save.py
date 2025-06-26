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

def save_corr_peaks(corr_peaks, path, pos, mode, window_size,
                    noise_pos=7, max_pos=7):
    """
    pos ∈ 1..6  with mode 'Baseline' | 'Optimized'
    pos = noise_pos (=7) with mode 'Noise'
    """
    # ----------- guard rails -------------------------------------------------
    if pos == noise_pos:
        assert mode == "Noise",       "Position 7 only accepts mode='Noise'"
    else:
        assert 1 <= pos <  noise_pos, "pos must be 1-6 unless it's the noise pos"
        assert mode in ("Baseline", "Optimized"), "pos 1-6 need Baseline/Optimized"

    # ----------- build *exactly* the columns we want -------------------------
    cols = []
    for p in range(1, max_pos + 1):
        modes = ["Noise"] if p == noise_pos else ["Baseline", "Optimized"]
        cols.extend([(f"Pos {p}", m) for m in modes])
    full_cols = pd.MultiIndex.from_tuples(cols)

    # ----------- load or create the DataFrame --------------------------------
    if os.path.exists(path):
        df = pd.read_csv(path, header=[0, 1])
        # keep only the columns we want, drop any old extras
        df = df.reindex(columns=full_cols)
    else:
        df = pd.DataFrame(columns=full_cols)

    # ----------- resize to exactly window_size rows --------------------------
    df = df.reindex(range(window_size))

    # ----------- insert the new data ----------------------------------------
    df[(f"Pos {pos}", mode)] = np.asarray(corr_peaks)[:window_size]

    # ----------- save atomically --------------------------------------------
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f"Saved  →  Pos {pos} / {mode}  ({len(corr_peaks)} samples)")


# === Parameters ===
samp_rate = 5.3e5
tx_gain = 0
NumSamples = 300000
rx_lo = 5.3e9
rx_mode = "manual"
rx_gain = 30
fc0 = int(200e3)
downsample_factor = 30
averaging_factor = 5
window_size = 50
threshold_factor = 3

pos  = 2              # 1 … 6 (7 for noise)
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
save_corr_peaks(corr_peaks, csv_path, pos, mode, window_size)
print(f"Saved to: {csv_path}")
