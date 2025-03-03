import numpy as np
import matplotlib.pyplot as plt

def generate_msequence(m, poly, initial_state):
    """
    Generates an m-sequence (maximum-length sequence) using a Linear Feedback Shift Register (LFSR).

    Parameters:
        m (int): Order of the LFSR (register length).
        poly (list of int): The primitive polynomial for feedback taps.
        initial_state (list of int): Initial state of the LFSR (must be of length m).

    Returns:
        mseq (numpy array): Generated m-sequence of length 2^m - 1.
    """
    N = 2**m - 1  # Length of the m-sequence
    lfsr = np.array(initial_state, dtype=int)  # Initialize LFSR with the initial state
    mseq = np.zeros(N, dtype=int)  # Output m-sequence
    
    for i in range(N):
        mseq[i] = lfsr[-1]  # Output bit (last bit in LFSR)

        # Compute feedback using XOR of selected taps
        feedback = np.mod(np.sum(lfsr[np.where(np.array(poly[1:]) == 1)]), 2)
        
        # Shift LFSR and insert feedback bit
        lfsr = np.roll(lfsr, 1)
        lfsr[0] = feedback

    # Repeat the first bit at the end to make it 64 bits
    mseq = np.append(mseq, mseq[0])

    return mseq

def autocorrelation(sequence):
    """
    Computes the autocorrelation of a given binary sequence.
    
    Parameters:
        sequence (numpy array): The sequence to compute the autocorrelation.

    Returns:
        autocorr (numpy array): The normalized autocorrelation function.
    """
    seq_bipolar = 2 * sequence - 1  # Convert 0 -> -1 and 1 -> +1
    autocorr = np.correlate(seq_bipolar, seq_bipolar, mode='full')
    autocorr = autocorr / np.max(autocorr)  # Normalize
    return autocorr

# Parámetros
m = 6  # Orden del LFSR
poly = [1, 0, 0, 0, 0, 1, 1]  # Polinomio primitivo para m=6
init_state = [1, 0, 0, 0, 0, 1]  # Estado inicial para la secuencia m

# Generar la secuencia m de 64 bits
mseq = generate_msequence(m, poly, init_state)
print(mseq)

# Calcular autocorrelación
autocorr = autocorrelation(mseq)