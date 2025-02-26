import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
F1 = 200e6  # 200 MHz
F2 = 400e6  # 400 MHz
Fs = 500e6    # 1 GHz sampling rate

# Time array for sampled signal
N = 128  # Number of samples
t = np.arange(N) / Fs

# Generate the sampled signal
y = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)

# High-definition signal for plotting
t_hd = np.linspace(0, N/Fs, 1024*N)  # 1024 points per sample
y_hd = np.cos(2 * np.pi * F1 * t_hd) + np.cos(2 * np.pi * F2 * t_hd)

# Blackman window
window = np.blackman(N)

# Apply window
y_windowed = y * window

# High-definition windowed signal for plotting
window_hd = np.interp(t_hd, t, window)
y_windowed_hd = y_hd * window_hd

# Perform DFT for both original and windowed signals
Y = np.fft.fft(y)
Y_windowed = np.fft.fft(y_windowed)
freq = np.fft.fftfreq(N, 1/Fs)

# Consider only positive frequencies (64 points)
positive_freq_mask = freq >= 0
Y_positive = Y[positive_freq_mask]
Y_windowed_positive = Y_windowed[positive_freq_mask]
freq_positive = freq[positive_freq_mask]

# Normalize frequency and magnitude
freq_normalized = freq_positive / Fs
Y_magnitude = 2 * np.abs(Y_positive) / N
Y_windowed_magnitude = 2 * np.abs(Y_windowed_positive) / N

# Plot time domain signals
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(t_hd * 1e9, y_hd, 'b-', label='Continuous (High-def)')
plt.plot(t * 1e9, y, 'ro', markersize=4, label='Samples')
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(212)
plt.plot(t_hd * 1e9, y_windowed_hd, 'g-', label='Continuous (High-def)')
plt.plot(t * 1e9, y_windowed, 'mo', markersize=4, label='Samples')
plt.title('Blackman Windowed Signal (Time Domain)')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# Plot frequency domain
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.stem(freq_normalized, Y_magnitude, 'r', markerfmt='ro', linefmt='r-', basefmt=" ")
plt.title('Normalized Frequency Domain (Original Signal)')
plt.xlabel('Normalized Frequency (f/Fs)')
plt.ylabel('Normalized Magnitude')
plt.xlim(0, 0.5)
plt.grid(True)

plt.subplot(212)
plt.stem(freq_normalized, Y_windowed_magnitude, 'b', markerfmt='bo', linefmt='b-', basefmt=" ")
plt.title('Normalized Frequency Domain (Blackman Windowed Signal)')
plt.xlabel('Normalized Frequency (f/Fs)')
plt.ylabel('Normalized Magnitude')
plt.xlim(0, 0.5)
plt.grid(True)

plt.tight_layout()
plt.show()