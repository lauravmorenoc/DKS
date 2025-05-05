import threading
import time

# ---- Lade deine beiden Funktionen oder importiere sie aus anderen Dateien ----
from RISpython import ris_init, ris_sequence
from sdr_visualizer import run_sdr_process  # Du müsstest deinen SDR-Code in eine Funktion packen

def run_ris_process():
    all_off = '!0x0000000000000000000000000000000000000000000000000000000000000000'
    all_on  = '!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'
    high = all_off
    low = all_on
    period = 2.5e-6
    duration = 50
    sleep_time = 0
    mseq = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    ris = ris_init('COM5', 115200)
    ris_sequence(ris, high, low, mseq, period, duration, sleep_time)

# ---- Hauptprogramm ----
if __name__ == "__main__":
    # RIS-Sequenz in separatem Thread starten
    ris_thread = threading.Thread(target=run_ris_process)
    ris_thread.start()

    # SDR-Plot in Hauptthread laufen lassen
    run_sdr_process()

    # Optional: Auf beide Threads warten
    ris_thread.join()
