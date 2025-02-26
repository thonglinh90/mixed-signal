import numpy as np
import matplotlib.pyplot as plt

# Parameters
signal_frequency = 2e6  # 2 MHz
sampling_frequency = 5e6  # 5 MHz
amplitude = 1
SNR_dB = 50  # Given SNR in dB
duration = 1000 * 1e-6  # Duration of the signal in seconds (100 µs)

# Calculate noise variance based on SNR
SNR_linear = 10 ** (SNR_dB / 10)
signal_power = amplitude ** 2 / 2  # RMS power of a sinusoid
noise_variance = signal_power / SNR_linear

# Time vector
t = np.arange(0, duration, 1 / sampling_frequency)

# Generate the sinusoidal signal
signal = amplitude * np.sin(2 * np.pi * signal_frequency * t)

# Add zero-mean white noise with calculated variance
noise = np.random.normal(0, np.sqrt(noise_variance), len(t))
sampled_signal = signal + noise

# Apply Blackman window
window = np.hanning(len(sampled_signal))
windowed_signal = sampled_signal * window

# Calculate Equivalent Noise Bandwidth (ENBW) of Blackman window
ACF = len(sampled_signal) / np.sum(window)
ECF = 1 / np.sqrt(np.mean(window**2))
print(ACF)
print(ECF)

# Calculate the DFT using numpy.fft
N = len(windowed_signal)
DFT = np.fft.fft(windowed_signal)
DFT_magnitude = np.abs(DFT) * ECF
frequencies = np.fft.fftfreq(N, d=1 / sampling_frequency)

# Calculate the Power Spectral Density (PSD)
PSD = DFT_magnitude ** 2 / N**2
PSD[1:N // 2 - 1] *= 2  # Double the power for positive frequencies, except DC and Nyquist

# Normalize frequencies for plotting
normalized_frequencies = frequencies / sampling_frequency

# Plot the Blackman window
plt.figure(figsize=(12, 4))
plt.plot(window)
plt.title('Blackman Window')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

# Plot the DFT magnitude and PSD
plt.figure(figsize=(12, 6))

# Plot DFT magnitude
plt.subplot(2, 1, 1)
plt.semilogy(normalized_frequencies[1:N // 2 - 1], np.abs(DFT[1:N // 2 - 1]) / N * 2)
plt.title('DFT Magnitude with Normalized Frequency (Blackman Window)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude (log scale)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlim(0, 0.5)

# Plot PSD
plt.subplot(2, 1, 2)
plt.semilogy(normalized_frequencies[1:N // 2 - 1], PSD[1:N // 2 - 1])
plt.title('Power Spectral Density (PSD) with Normalized Frequency (Blackman Window)')
plt.xlabel('Normalized Frequency')
plt.ylabel('Power Spectral Density (log scale)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlim(0, 0.5)

plt.tight_layout()
plt.show()

# Recalculate SNR from DFT plot
signal_idx = np.argmax(PSD[:N // 2])                 # Locate the frequency with peak power
signal_power_dft = PSD[signal_idx]        # Signal power at peak frequency
noise_power_dft = np.sum(PSD[:N // 2]) - signal_power_dft   # Total noise power in positive frequencies
SNR_dft = signal_power_dft / noise_power_dft         # Compute SNR from PSD
SNR_dB_dft = 10 * np.log10(SNR_dft)                  # Convert to dB

print(f"Signal Power (DFT): {signal_power_dft}")
print(f"Noise Power (DFT): {noise_power_dft}")
print(f"Calculated Noise Variance: {noise_variance:.4e}")
print(f"Recalculated SNR from DFT: {SNR_dB_dft:.2f} dB")
