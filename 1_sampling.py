import numpy as np
import matplotlib.pyplot as plt

# Define parameters
f_signal = 300e6  # Signal frequency (800 MHz)
f_sampling = 500e6  # Sampling frequency (500 MHz)
t_end = 1e-6  # Duration of the signal (1 microsecond)
num_samples = int(t_end * f_sampling)  # Number of samples

# Time vector for sampling
t = np.linspace(0, t_end, num_samples, endpoint=False)

# Continuous-time signal
signal = np.cos(2 * np.pi * f_signal * t)

# Perform FFT on the sampled signal
fft_signal = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(len(signal), d=1/f_sampling)

# Shift FFT for plotting and normalize
fft_signal_shifted = np.fft.fftshift(fft_signal)
fft_freqs_shifted = np.fft.fftshift(fft_freqs)
magnitude_spectrum = np.abs(fft_signal_shifted) / len(signal)

# Plot in frequency domain
plt.figure(figsize=(12, 6))

# Highlight triangles at integer multiples of sampling frequency in gray
for n in range(-4, 5):  # Show replicas from -4 to 4 times f_sampling
    center_freq = n
    triangle_x = [center_freq - f_signal / f_sampling,
                  center_freq,
                  center_freq + f_signal / f_sampling]
    triangle_y = [0, max(magnitude_spectrum), 0]
    plt.fill(triangle_x, triangle_y, color='gray', alpha=0.3)  # Set triangles to gray

# Add labels and adjust plot
plt.title("Frequency Domain Representation of Sampled Signal")
plt.xlabel("Normalized Frequency (f/fs)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.xlim(-4, 4)  # Set x-axis limits to show -2GHz to 2GHz (normalized)
plt.ylim(0, max(magnitude_spectrum) * 1.1)  # Adjust y-axis for better visibility
plt.tight_layout()

# Display the plot
plt.show()
