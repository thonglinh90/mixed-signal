import numpy as np
import matplotlib.pyplot as plt

# Parameters
bits = 12
amplitude = 2*bits
sampling_rate = 800e6  # Oversampling rate
frequency = 3e6  # Frequency of the sine wave (Hz)
num_cycles = 100  # Number of cycles to sample

# Calculate duration for 30 cycles
duration = num_cycles / frequency

# Generate time vector for the specified duration
time = np.arange(0, duration, 1/sampling_rate)

# Generate sine wave
sine_wave = amplitude * np.cos(2 * np.pi * frequency * time)

# Quantize the sine wave
quantization_levels = 2 ** bits
quantized_wave = np.round(sine_wave * (quantization_levels / (2 * amplitude))) / (quantization_levels / (2 * amplitude))

# Perform FFT and DSP
fft_result = np.fft.fft(quantized_wave)
frequencies = np.fft.fftfreq(len(fft_result), d=1/sampling_rate)
fft_magnitude = np.abs(fft_result) / len(fft_result)

# Calculate DSP (Power Spectral Density)
dsp = fft_magnitude ** 2

# Calculate theoretical quantization noise power and RMS
quantization_step = (2 * amplitude) / quantization_levels
theoretical_noise_power = (quantization_step ** 2) / 12
theoretical_noise_rms = np.sqrt(theoretical_noise_power)

# Calculate noise power from DSP (excluding signal power at fundamental frequency)
signal_index = np.argmax(dsp[:len(frequencies)//2])  # Find the index of the fundamental frequency
noise_power_from_dsp = (
    2 * (np.sum(dsp[:len(frequencies)//2]) - dsp[signal_index]) - dsp[0] - dsp[len(frequencies)//2 + 1]
)

# Calculate SNR in dB
signal_power_from_dsp = 2 * dsp[signal_index]  # Signal power at the fundamental frequency
snr_linear = signal_power_from_dsp / noise_power_from_dsp
snr_db = 10 * np.log10(snr_linear)

# Display theoretical and calculated noise power, RMS, and SNR
print(f"Theoretical Quantization Noise Power: {theoretical_noise_power}")
print(f"Theoretical Quantization Noise RMS: {theoretical_noise_rms}")
print(f"Noise Power from DSP: {noise_power_from_dsp}")
print(f"Signal Power from DSP: {dsp[signal_index]}")
print(f"SNR: {snr_db:.2f} dB")

# Generate higher-resolution sine wave for plotting (1024 points per cycle)
high_res_sampling_rate = frequency * 1024  # 1024 points per cycle
high_res_time = np.arange(0, duration, 1/high_res_sampling_rate)
high_res_sine_wave = amplitude * np.cos(2 * np.pi * frequency * high_res_time)

# Plotting
plt.figure(figsize=(12, 8))

# Plot sampled data with high-resolution sine wave for comparison
plt.subplot(3, 1, 1)
plt.plot(high_res_time, high_res_sine_wave, label="High-Resolution Sine Wave", linestyle='--', alpha=0.7)
plt.stem(time, quantized_wave, label="Quantized Wave", basefmt=" ")
plt.title("Sampled Data")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot FFT
plt.subplot(3, 1, 2)
plt.stem(frequencies, fft_magnitude, basefmt=" ")
plt.title("FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid()

# Plot DSP
plt.subplot(3, 1, 3)
plt.stem(frequencies, dsp, basefmt=" ")
plt.title("DSP")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.grid()

plt.tight_layout()
plt.show()