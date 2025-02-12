import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define the filter coefficients
b = [1, 1, 1,1,1]  # Numerator coefficients
a = [1]        # Denominator coefficient

# Compute the frequency response
# w = 2*pi*f
w, h = signal.freqz(b, a)

# Convert frequency to Hz (assuming a sample rate of 1 Hz)
freq = w / (2 * np.pi)

# Plot the magnitude response
plt.figure(figsize=(10, 6))
plt.plot(freq, np.abs(h))
plt.title('Magnitude Response of H = 1 + z^-1 + z^-1')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()