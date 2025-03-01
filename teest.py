import numpy as np
import matplotlib.pyplot as plt

# Parameters
signal_length = 1024  # Total number of samples
pulse_width = 256     # Width of the pulse

# Create a rectangular pulse
rect_pulse = np.zeros(signal_length)
rect_pulse[signal_length//2 - pulse_width//2 : signal_length//2 + pulse_width//2] = 1

# Create a non-perfect rectangular pulse (diagonal increase)
non_perfect_pulse = np.zeros(signal_length)
for i in range(signal_length//2 - pulse_width//2, signal_length//2 + pulse_width//2):
    non_perfect_pulse[i] = (i - (signal_length//2 - pulse_width//2)) / pulse_width

# Compute FFTs
fft_rect = np.fft.fft(rect_pulse)
fft_non_perfect = np.fft.fft(non_perfect_pulse)

# Compute magnitudes and normalize by signal length
fft_rect_magnitude_normalized = np.abs(fft_rect) / signal_length
fft_non_perfect_magnitude_normalized = np.abs(fft_non_perfect) / signal_length

# Plot the pulses and their FFTs
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(rect_pulse, label="Rectangular Pulse")
plt.plot(non_perfect_pulse, label="Non-Perfect Rectangular Pulse", linestyle='--')
plt.title("Pulses")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(fft_rect_magnitude_normalized[:signal_length//2], label="Rectangular Pulse")
plt.plot(fft_non_perfect_magnitude_normalized[:signal_length//2], label="Non-Perfect Rectangular Pulse", linestyle='--')
plt.title("FFT of Pulses (Normalized)")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.legend()

plt.tight_layout()
plt.show()