import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
F = 2e6  # 2 MHz
Fs = 5e6  # 5 MHz sampling rate

# Time array for sampled signal
N = 128  # Number of samples
t = np.arange(N) / Fs

# Generate the sampled signal
x = np.cos(2 * np.pi * F * t)

# High-definition signal for plotting
t_hd = np.linspace(0, N/Fs, 1024*N)  # 1024 points per sample
x_hd = np.cos(2 * np.pi * F * t_hd)

# Blackman window
window = np.blackman(N)

# Apply window
x_windowed = x * window

# High-definition windowed signal for plotting
window_hd = np.interp(t_hd, t, window)
x_windowed_hd = x_hd * window_hd

# Perform DFT for both original and windowed signals
X = np.fft.fft(x)
X_windowed = np.fft.fft(x_windowed)
freq = np.fft.fftfreq(N, 1/Fs)

# Consider only positive frequencies (64 points)
positive_freq_mask = freq >= 0
X_positive = X[positive_freq_mask]
X_windowed_positive = X_windowed[positive_freq_mask]
freq_positive = freq[positive_freq_mask]

# Normalize frequency and magnitude
freq_normalized = freq_positive / Fs
X_magnitude = 2 * np.abs(X_positive) / N
X_windowed_magnitude = 2 * np.abs(X_windowed_positive) / N

# Plot time domain signals
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(t_hd * 1e6, x_hd, 'b-', label='Continuous (High-def)')
plt.plot(t * 1e6, x, 'ro', markersize=4, label='Samples')
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(212)
plt.plot(t_hd * 1e6, x_windowed_hd, 'g-', label='Continuous (High-def)')
plt.plot(t * 1e6, x_windowed, 'mo', markersize=4, label='Samples')
plt.title('Blackman Windowed Signal (Time Domain)')
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# Plot frequency domain
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.stem(freq_normalized, X_magnitude, 'r', markerfmt='ro', linefmt='r-', basefmt=" ")
plt.title('Normalized Frequency Domain (Original Signal)')
plt.xlabel('Normalized Frequency (f/Fs)')
plt.ylabel('Normalized Magnitude')
plt.xlim(0, 0.5)
plt.grid(True)

plt.subplot(212)
plt.stem(freq_normalized, X_windowed_magnitude, 'b', markerfmt='bo', linefmt='b-', basefmt=" ")
plt.title('Normalized Frequency Domain (Blackman Windowed Signal)')
plt.xlabel('Normalized Frequency (f/Fs)')
plt.ylabel('Normalized Magnitude')
plt.xlim(0, 0.5)
plt.grid(True)

plt.tight_layout()
plt.show()