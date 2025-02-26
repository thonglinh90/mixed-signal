import numpy as np
import matplotlib.pyplot as plt

# Define parameters
frequency = 200e6  # 200 MHz
sampling_rate = 400e6  # 400 MHz
cycles = 100  # Number of cycles
points_per_cycle = 1024  # High resolution for original signal
phase = 0

# Calculate the time period of one cycle
period = 1 / frequency

# Generate time values for high-resolution original signal (1024 points per cycle)
time_high_res = np.linspace(0, cycles * period, cycles * points_per_cycle, endpoint=False)
original_signal = 32 * np.cos(2 * np.pi * frequency * time_high_res + phase)

# Generate time values and sampled signal for sampling rate of 400 MHz
time_sampled = np.arange(0, cycles * period, 1 / sampling_rate)
sampled_signal = np.round(32 * np.cos(2 * np.pi * frequency * time_sampled + phase))

# Perform FFT on the sampled signal
fft_result = np.fft.fft(sampled_signal)
fft_magnitude = np.abs(fft_result)  # Normalize magnitude
fft_frequency = np.fft.fftfreq(len(sampled_signal), d=1/sampling_rate)  # Frequency axis normalized to sampling frequency

# Only take the positive half of the spectrum
positive_freqs = fft_frequency[:len(fft_frequency)//2 + 1]
positive_magnitude = fft_magnitude[:len(fft_magnitude)//2 + 1]
positive_freqs[-1] = np.abs(positive_freqs[-1]) #bring f_s/2 to positive frequency

#psd

psd = 2 * (positive_magnitude ** 2) / (len(sampled_signal)*sampling_rate) 

# Plot the signals
plt.figure(figsize=(12, 8))

# Plot the original high-resolution signal
plt.subplot(2, 1, 1)
plt.plot(time_high_res, original_signal, label="Original Signal (1024 points/cycle)", alpha=0.7)
plt.stem(time_sampled, sampled_signal, linefmt='r-', markerfmt='ro', basefmt=' ', label="Sampled Signal (400 MHz)")
plt.title("200MHz Cosine Signal: Original vs Sampled")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot the FFT magnitude
plt.subplot(2, 1, 2)
plt.stem(positive_freqs / sampling_rate, psd, linefmt='b-', markerfmt='bo', basefmt=' ', label="FFT Magnitude")
plt.title("DSP of Sampled Signal")
plt.xlabel("Normalized Frequency (f / f_s)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

peak_idx = np.argmax(psd)
i = 0
noise_power = 0
while i < len(psd):
    if i != peak_idx:
        noise_power += psd[i]
        # print(noise_power)
    i += 1

noise_power_db = 10*np.log10(noise_power)
signal_power_db = 10*np.log10(np.max(psd))

print('SNR is ',signal_power_db - noise_power_db,' dB')
