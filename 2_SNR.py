import numpy as np
import matplotlib.pyplot as plt

# Parameters
signal_frequency = 2e6  # 2 MHz
sampling_frequency = 5e6  # 5 MHz
amplitude = 1
SNR_dB = 50  # Given SNR in dB
duration = 100*1e-6  # Duration of the signal in seconds (1 ms)

# Calculate noise variance based on SNR
SNR_linear = 10 ** (SNR_dB / 10)
signal_power = amplitude ** 2 / 2  # RMS power of a sinusoid
noise_variance = signal_power / SNR_linear

# Time vector
t = np.arange(0, duration, 1/sampling_frequency)

# Generate the sinusoidal signal
signal = amplitude * np.sin(2 * np.pi * signal_frequency * t)

# Add zero-mean white noise with calculated variance
noise = np.random.normal(0, np.sqrt(noise_variance), len(t))
sampled_signal = signal + noise

# Calculate the DFT using numpy.fft
N = len(sampled_signal)
DFT = np.fft.fft(sampled_signal) # a half of N
frequencies = np.fft.fftfreq(N, d=1/sampling_frequency) #sampling happend here
# print(frequencies)

# Calculate the Power Spectral Density (PSD) manually
PSD = np.abs(DFT) ** 2 / N**2
PSD[1:N//2 - 1] *= 2 # half of positive frequency, except DC and Nyquist

# Normalize frequencies for plotting
normalized_frequencies = frequencies / sampling_frequency

# Plot the DFT magnitude
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.semilogy(normalized_frequencies[1:N//2 - 1], np.abs(DFT[1:N//2 - 1])/ N *2)
plt.title('DFT Magnitude with Normalized Frequency')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude (log scale)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlim(0, 0.5)

# Plot the PSD
plt.subplot(2, 1, 2)
plt.semilogy(normalized_frequencies[1:N//2 - 1], PSD[1:N//2 - 1])
plt.title('Power Spectral Density (PSD) with Normalized Frequency')
plt.xlabel('Normalized Frequency')
plt.ylabel('Power Spectral Density (log scale)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlim(0, 0.5)

plt.tight_layout()
plt.show()

# Recalculate SNR from DFT plot
signal_idx = np.argmax(PSD[:N//2])  # Locate the frequency with the peak power
signal_power_dft = PSD[signal_idx] # Divide by N for correct normalization
noise_power_dft = np.sum(PSD[:N//2]) - signal_power_dft  # Total noise power
SNR_dft = signal_power_dft / noise_power_dft
SNR_dB_dft = 10 * np.log10(SNR_dft)

print(f"signal_power_dft: {signal_power_dft}")
print(f"noise_power_dft: {noise_power_dft}")
print(f"Calculated noise variance: {noise_variance:.2e}")
print(f"Recalculated SNR from DFT: {SNR_dB_dft:.2f} dB")
