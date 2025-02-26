import numpy as np
import matplotlib.pyplot as plt

# Parameters
signal_frequency = 2e6  # 2 MHz
sampling_frequency = 5e6  # 5 MHz
amplitude = 1
SNR_dB = 50
duration = 100e-6  # 100 μs

# Derived parameters
N = int(sampling_frequency * duration)
t = np.arange(N) / sampling_frequency
signal = amplitude * np.sin(2*np.pi*signal_frequency*t)

# Noise calculation
SNR_linear = 10**(SNR_dB/10)
signal_power = amplitude**2/2
noise_variance = signal_power / SNR_linear
noise = np.random.normal(0, np.sqrt(noise_variance), N)
sampled_signal = signal + noise

# Windowing and correction factors
window = np.hanning(N)
windowed_signal = sampled_signal * window
window_scale = np.sum(window ** 2)
print(window_scale)
ENBW = 1.5  # Hanning window property

# Critical correction factors
ACF = N / np.sum(window)  # Amplitude correction (≈2.0)
ECF = np.sqrt(N / np.sum(window**2))  # Energy correction (≈1.633)

# Spectral processing
DFT = np.fft.fft(windowed_signal) * ACF
frequencies = np.fft.fftfreq(N, 1/sampling_frequency)


# Proper PSD calculation
# PSD = (np.abs(DFT)**2) / (N * np.sum(window**2))
PSD = (np.abs(DFT)**2) / (sampling_frequency*window_scale)
PSD = PSD[:N//2]
PSD[1: N // 2 - 1] *= 2  # Correct single-sided scaling

# Normalized frequency axis
normalized_freq = frequencies[:N//2] / sampling_frequency

# Plotting configuration
plt.figure(figsize=(12, 8))

# DFT Magnitude plot
plt.subplot(2, 1, 1)
plt.semilogy(normalized_freq[1:], np.abs(DFT[:N//2])[1:]/N, 'b')
plt.title('Corrected DFT Magnitude Spectrum')
plt.xlabel('Normalized Frequency (f/fs)')
plt.ylabel('Magnitude (V)')
plt.grid(True)
plt.xlim(0, 0.5)

# PSD plot
plt.subplot(2, 1, 2)
plt.semilogy(normalized_freq[1:], PSD[1:], 'r')
plt.title('Corrected Power Spectral Density')
plt.xlabel('Normalized Frequency (f/fs)')
plt.ylabel('Power/Frequency (V²/Hz)')
plt.grid(True)
plt.xlim(0, 0.5)

plt.tight_layout()
plt.show()

# SNR calculation with main lobe integration
signal_bins = 3  # Hanning window main lobe width
peak_idx = np.argmax(PSD)
start_idx = max(0, peak_idx - signal_bins//2)
end_idx = min(len(PSD), peak_idx + signal_bins//2 + 1)

signal_power = np.sum(PSD[start_idx:end_idx])
noise_power = (np.sum(PSD) - signal_power) / ENBW
SNR = 10 * np.log10(signal_power / noise_power)

print(f"Theoretical SNR: {SNR_dB} dB")
print(f"Corrected SNR: {SNR:.2f} dB")
print(f"Signal Power: {signal_power:.2e} V²")
print(f"Noise Power: {noise_power:.2e} V²")
print(f"Noise Variance: {noise_variance:.2e} V²")
