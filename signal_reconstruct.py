import numpy as np
import matplotlib.pyplot as plt

# Define parameters
f_signal = 300e6  # Signal frequency: 300MHz
f_sampling = 600e6  # Sampling frequency: 800MHz
cycles = 10  # Number of cycles
extra_cycles = 2  # Extra cycles for interpolation
points_per_cycle = 1024  # Points per cycle

# Add sampling delay
T = 1 / f_sampling  # Sampling period
#sampling_delay = 0  # Delay of T/2
sampling_delay = T / 2  # Delay of T/2

# Calculate time array (including extra cycles)
t = np.linspace(0, (cycles + extra_cycles) / f_signal, (cycles + extra_cycles) * points_per_cycle)

# Generate extended signal
continuous_signal = np.sin(2 * np.pi * f_signal * t)

# Generate sampled data (including extra cycles and delay)
t_sampled = np.arange(sampling_delay, (cycles + extra_cycles) / f_signal, 1 / f_sampling)
discrete_signal = np.sin(2 * np.pi * f_signal * t_sampled)

# Create an array of zeros for the interpolated discrete signal
interpolated_discrete = np.zeros_like(t)

# Create an array to store the positions of sample points
sample_positions = []

# Set the values at the sampled points and record their positions
for sampled_t, sampled_val in zip(t_sampled, discrete_signal):
    idx = np.abs(t - sampled_t).argmin()
    interpolated_discrete[idx] = sampled_val
    sample_positions.append(idx)

# Convert sample_positions to a numpy array
sample_positions = np.array(sample_positions)

# Combine continuous and interpolated discrete signals into one array
combined_signal = np.column_stack((continuous_signal, interpolated_discrete))

def reconstructionFormula(time, sample_signal, sample_positions, sampling_freq, delay):
    T = 1/sampling_freq
    recovered_signal = np.zeros_like(time)
    
    for n, pos in enumerate(sample_positions):
        sample_mag = sample_signal[pos]
        t_diff = (time - (n * T + delay)) / T
        recovered_signal += sample_mag * np.sinc(t_diff)
    
    return recovered_signal

recovered_signal = reconstructionFormula(t, combined_signal[:, 1], sample_positions, f_sampling, sampling_delay)

# Remove extra cycles after recovery
t_final = t[:(cycles * points_per_cycle)]
continuous_signal_final = continuous_signal[:(cycles * points_per_cycle)]
discrete_signal_final = interpolated_discrete[:(cycles * points_per_cycle)]
recovered_signal_final = recovered_signal[:(cycles * points_per_cycle)]

print(f"Mean Square Error: {np.mean((recovered_signal_final - continuous_signal_final) ** 2):.6e}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(t_final * 1e9, continuous_signal_final, label='Continuous Signal', alpha = 0.25)
plt.plot(t_final * 1e9, discrete_signal_final, label='Sampled Signal')
plt.plot(t_final * 1e9, recovered_signal_final, label='Recovered Signal')
# plt.plot(t_sampled[t_sampled <= cycles / f_signal] * 1e9, 
#          discrete_signal[t_sampled <= cycles / f_signal], 'r.', 
#          markersize=4, label='Discrete Signal (Sampled Points)')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.title(f'300MHz Signal Sampled at {f_sampling*1e-6:.2f}MHz with {sampling_delay*1e9:.2f}ns Delay')
plt.legend()
plt.grid(True)
plt.show()