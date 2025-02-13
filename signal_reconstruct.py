import numpy as np
import matplotlib.pyplot as plt

# Define parameters
f_signal = 300e6  # Signal frequency: 300MHz
f_sampling = 800e6  # Sampling frequency: 800MHz
cycles = 10  # Number of cycles
extra_cycles = 2  # Extra cycles for interpolation
points_per_cycle = 1024  # Points per cycle

# Calculate time array (including extra cycles)
t_extended = np.linspace(0, (cycles + extra_cycles) / f_signal, (cycles + extra_cycles) * points_per_cycle)

# Generate extended signal
continuous_signal_extended = np.sin(2 * np.pi * f_signal * t_extended)

# Generate sampled data (including extra cycles)
t_sampled_extended = np.arange(0, (cycles + extra_cycles) / f_signal, 1 / f_sampling)
discrete_signal_extended = np.sin(2 * np.pi * f_signal * t_sampled_extended)

# Remove extra cycles
t = t_extended[:cycles * points_per_cycle]
continuous_signal = continuous_signal_extended[:cycles * points_per_cycle]

# Calculate the actual discrete signal points for the original 10 cycles
t_sampled = t_sampled_extended[t_sampled_extended <= cycles / f_signal]
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

# Plot
plt.figure(figsize=(12, 6))
plt.plot(t * 1e9, combined_signal[:, 0], label='Continuous Signal')
plt.plot(t * 1e9, combined_signal[:, 1], label='Discrete Signal')
plt.plot(t_sampled * 1e9, discrete_signal, 'r.', markersize=4, label='Discrete Signal (Sampled Points)')
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.title('300MHz Signal Sampled at 800MHz (Combined Array)')
plt.legend()
plt.grid(True)
plt.show()

# Print array sizes
print(f"Combined signal array size: {combined_signal.shape}")
print(f"Continuous signal size: {len(combined_signal[:, 0])}")
print(f"Discrete signal size: {len(combined_signal[:, 1])}")
print(f"Actual discrete signal size: {len(discrete_signal)}")
print(f"Sample positions array size: {len(sample_positions)}")
print(f"First few sample positions: {sample_positions[:5]}")
