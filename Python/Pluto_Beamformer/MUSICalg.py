import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# Simulation parameters
M = 8                      # Number of antennas
K = 3                       # Number of sources (APs)
d = 0.0107 / 2              # Distance between antennas (half-wavelength)
wavelength = 0.0107         # Wavelength at 28 GHz (in meters)
SNR = 20                    # Signal-to-noise ratio (dB)

# True DoA angles
true_DoAs = [-30, 0, 40]    # True DoA angles (in degrees)

# Check if M >= K
if M < K:
    print('Error: Number of antennas must be greater than or equal to number of sources')


### Functions ###

# Function to generate synthetic data
def generate_synthetic_data(M, K, d, wavelength, DoAs, SNR):
    N = 1000 # Number of samples
    A = np.exp(-1j * 2 * np.pi * d * np.arange(0, M)[:, None] * np.sin(np.radians(DoAs)) / wavelength) # Steering matrix
    S = np.sqrt(0.5) * (np.random.randn(K, N) + 1j * np.random.randn(K, N)) # Complex Gaussian signals
    noise_variance = 10**(-SNR / 10) * np.trace(A @ (S @ S.conj().T) @ A.conj().T) / (M * N)
    n = np.sqrt(noise_variance / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N)) # Complex Gaussian noise
    x = x = A @ S + n # Received signal matrix
    return x, A

'''
# Algorithm function
def algo(x, M, K, d, wavelength):
    # Compute the MUSIC spectrum
    theta = np.arange(-90, 90.1, 0.1)
    Vn, Pmusic = compute_music_spectrum(x, M, K, d, wavelength, theta)

    # Find the initial estimates of DoAs
    doa_indices = np.argsort(Pmusic)[::-1]

    # If more than K peaks are found, select the K largest ones
    if len(doa_indices) > K:
        doa_indices = doa_indices[:K]

    initial_DoAs = theta[doa_indices]

    # Redefine DoAs
    doa_estimates = np.zeros(K, dtype=float)
    for i in range(len(initial_DoAs)):
        # Define error function
        error_function = lambda x: abs(
            1 / abs((a(x, d, M, wavelength).conj().T @ Vn) @ (Vn.conj().T @ a(x, d, M, wavelength)))
            - Pmusic[doa_indices[i]]
        )
        # Optimize using scipy.optimize.minimize with Nelder-Mead method (equivalent to fminsearch)
        result = minimize(error_function, initial_DoAs[i], method="Nelder-Mead")
        doa_estimates[i] = result.x[0]  # Obtain minimal value

    # If less than K estimates were found, fill the remaining ones with NaN
    if len(doa_estimates) < K:
        doa_estimates = np.concatenate([doa_estimates, np.full(K - len(doa_estimates), np.nan)])
    return doa_estimates
    


# Function to compute steering vector
def a(angle, d, M, wavelength):
    a_vec = np.exp(-1j * 2 * np.pi * d * np.arange(0, M)[:, None] * np.sin(np.radians(angle)) / wavelength)
    return a_vec 


# Function to compute the MUSIC spectrum
def compute_music_spectrum(x, M, K, d, wavelength, theta):
    Rxx = (x @ x.conj().T) / x.shape[1]
    D, V = np.linalg.eig(Rxx)
    idx = np.argsort(np.diag(D))[::-1]
    V = V[:, idx]
    Vn = V[:, K:]

    # Calculate MUSIC spectrum
    Pmusic = np.array([
        1 / abs((a(angle, d, M, wavelength).conj().T @ Vn) @ (Vn.conj().T @ a(angle, d, M, wavelength)))
        for angle in theta
    ])
    return Vn, Pmusic


# Function to plot the results
def plot_results(x, _, doa_estimates, M, K, d, wavelength, true_DoAs):
    # Compute the MUSIC spectrum
    theta = np.arange(-90, 90.1, 0.1)  # Angle range in degrees
    _, Pmusic = compute_music_spectrum(x, M, K, d, wavelength, theta)

    # Plot the MUSIC spectrum
    plt.figure()
    plt.plot(
        theta,
        10 * np.log10(np.abs(Pmusic) / np.max(np.abs(Pmusic))),  # Normalize and convert to dB
        linewidth=2
    )
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Normalized Power (dB)')
    plt.title('MUSIC Spectrum')
    plt.grid(True)

    # Plot the true and estimated DoAs
    plt.figure()
    plt.plot(
        np.arange(1, K + 1),
        np.sort(true_DoAs),  # Sort true DoAs for visualization
        'ro',
        linewidth=2,
        markersize=10,
        label='True DoAs'
    )
    plt.plot(
        np.arange(1, K + 1),
        np.sort(doa_estimates),  # Sort estimated DoAs for comparison
        'bx',
        linewidth=2,
        markersize=10,
        label='Estimated DoAs'
    )
    plt.xlabel('Source index')
    plt.ylabel('Angle (degrees)')
    plt.title('True and Estimated DoAs')
    plt.legend()  # Add a legend
    plt.grid(True)
    plt.show()


### End Functions ###
'''

# Generate synthetic data
x, A_true = generate_synthetic_data(M, K, d, wavelength, true_DoAs, SNR)

print(A_true)
'''
# Apply the algorithm
doa_estimates = algo(x, M, K, d, wavelength)

# Plot the results
plot_results(x, A_true, doa_estimates, M, K, d, wavelength, true_DoAs)

'''