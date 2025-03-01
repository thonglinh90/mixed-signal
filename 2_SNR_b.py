import numpy as np
import matplotlib.pyplot as plt

# Parameters
signal_frequency = 2e6  # 2 MHz
sampling_frequency = 5e6  # 5 MHz
amplitude = 1  # Signal amplitude (1 V)
SNR_dB = 50  # Target SNR in dB
duration = 100e-6  # Signal duration (100 μs)

# Derived parameters
N = int(sampling_frequency * duration)  # Number of samples
t = np.arange(N) / sampling_frequency   # Time vector
signal = amplitude * np.sin(2 * np.pi * signal_frequency * t)  # Sinewave

# Noise calculation (Gaussian noise)
SNR_linear = 10**(SNR_dB / 10)  # Convert SNR from dB to linear scale
signal_power = amplitude**2 / 2  # Power of sinewave (A^2/2 for sinusoid)
noise_variance = signal_power / SNR_linear  # Noise variance
noise = np.random.normal(0, np.sqrt(noise_variance), N)  # Gaussian noise
sampled_signal = signal + noise  # Noisy sinewave

# Windowing and correction factors
window = np.blackman(N)  # Hanning window
windowed_signal = sampled_signal * window  # Apply window to signal

# Correction factors for windowing
ACF = N / np.sum(window)  # Amplitude Correction Factor (~2.0 for Hanning)
ECF = np.sqrt(N / np.sum(window**2))  # Energy Correction Factor (~1.633)
ENBW = N*np.sum(window**2)/(np.sum(window)**2)         # Equivalent Noise Bandwidth (~1.5 for Hanning)

# Spectral processing
DFT = np.fft.fft(windowed_signal) * ACF / N  # Corrected DFT with amplitude scaling
frequencies = np.fft.fftfreq(N, d=1/sampling_frequency)[:N//2]  # Frequency axis

# Proper PSD calculation with ENBW correction
PSD = (np.abs(DFT[:N//2])**2)  # PSD in V^2

# Plotting configuration
plt.figure(figsize=(12, 8))

# DFT Magnitude plot
plt.subplot(2, 1, 1)
plt.plot(frequencies / sampling_frequency, np.abs(DFT[:N//2]), 'b')
plt.title('DFT Magnitude Spectrum')
plt.xlabel('Normalized Frequency (f/fs)')
plt.ylabel('Magnitude (V)')
plt.grid(True)
plt.xlim(0, signal_frequency / sampling_frequency + 0.05)

# PSD plot
plt.subplot(2, 1, 2)
plt.semilogy(frequencies, PSD, 'r')
plt.title('Power Spectral Density (PSD)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V²)')
plt.grid(True)
plt.xlim(0, sampling_frequency / 2)

plt.tight_layout()
plt.show()

# SNR calculation from PSD
signal_bin_idx = np.argmax(PSD)
main_lobe_bins = 6  # Fixed value for Hanning window (4 bins total: ±2 bins around peak)

# Integrate power over ±2 bins around the peak
signal_power_estimated = np.sum(PSD[signal_bin_idx - main_lobe_bins : signal_bin_idx + main_lobe_bins])
total_power_estimated = np.sum(PSD)
noise_power_estimated = total_power_estimated - signal_power_estimated

# Calculate SNR
SNR_calculated_dB = 10 * np.log10(signal_power_estimated / noise_power_estimated)

# Print results
print(f"Theoretical SNR: {SNR_dB} dB")
print(f"Calculated SNR: {SNR_calculated_dB:.2f} dB")
print(f"Main Lobe Bins: {main_lobe_bins} bins")
print(f"Signal Power (from PSD): {signal_power_estimated:.4e} V²")
print(f"Noise Power (from PSD): {noise_power_estimated:.4e} V²")
print(f"Noise Variance: {noise_variance:.4e} V²")