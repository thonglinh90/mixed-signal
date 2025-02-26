import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
f_s = 100.0  # Hz sampling frequency
time = np.arange(0.0, 10.0, 1/f_s)
x = 5 * np.sin(13.2 * 2 * np.pi * time) + 3 * np.sin(43.9 * 2 * np.pi * time)
x = x + np.random.randn(len(time)) * 0.5  # Reduced noise amplitude for better visibility

# Apply Hann window and take the FFT
win = np.hanning(len(x))
FFT = np.fft.fft(win * x)
n = len(FFT)
freq = np.fft.fftfreq(n, 1/f_s)

# Calculate the magnitude spectrum
magnitude_spectrum = np.abs(FFT) / n  # Normalize by dividing by n
magnitude_spectrum[1:-1] *= 2  # Multiply by 2 (except DC and Nyquist)

# Get positive frequencies
positive_freq_mask = freq >= 0
freq_positive = freq[positive_freq_mask]
magnitude_spectrum_positive = magnitude_spectrum[positive_freq_mask]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(freq_positive, magnitude_spectrum_positive)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Magnitude Spectrum of Windowed Signal")
plt.grid(True)
plt.xlim(0, 50)  # Limit x-axis to show relevant frequencies
plt.show()

# Print peak amplitudes
peak_indices = np.argsort(magnitude_spectrum_positive)[-2:]
for idx in peak_indices:
    print(f"Peak at {freq_positive[idx]:.2f} Hz with amplitude {magnitude_spectrum_positive[idx]:.2f}")
