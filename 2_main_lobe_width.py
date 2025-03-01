import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

def find_main_lobe_bins(window_type='hann', N=1024, pad_factor=16):
    """
    Calculate null-to-null main lobe width for a window function.
    Returns theoretical value for known windows, calculates for others.
    """
    # Theoretical values for common windows (null-to-null width)
    theoretical_widths = {
        'boxcar': 2,  # Rectangular
        'hann': 4,
        'hamming': 4,
        'blackman': 6,
        'blackmanharris': 8
    }
    
    if window_type in theoretical_widths:
        return theoretical_widths[window_type]
    
    # For unknown windows, calculate via FFT zero-crossing detection
    window = get_window(window_type, N)
    padded_window = np.pad(window, (0, N*(pad_factor-1)), mode='constant')
    
    # Compute FFT and find first zeros
    fft_result = np.fft.fftshift(np.fft.fft(padded_window))
    magnitude = np.abs(fft_result)
    
    peak_idx = np.argmax(magnitude)
    
    # Find first zero crossing left of peak
    left_zero = peak_idx
    while left_zero > 0 and magnitude[left_zero] > 1e-6:
        left_zero -= 1
        
    # Find first zero crossing right of peak
    right_zero = peak_idx
    while right_zero < len(magnitude)-1 and magnitude[right_zero] > 1e-6:
        right_zero += 1
        
    # Convert to original bins
    bin_width = (right_zero - left_zero) / pad_factor
    
    return bin_width

# Example usage for Hanning window
main_lobe_bins = find_main_lobe_bins('blackman', 2667, 16)

# Plot verification
N = 512
window = get_window('blackman', N)
fft_result = np.fft.fftshift(np.fft.fft(window, 16*N))
freqs = np.fft.fftshift(np.fft.fftfreq(16*N, 1/N))


plt.figure(figsize=(12, 6))

# Calculate magnitude in dB
magnitude_db = 20 * np.log10(np.abs(fft_result) / np.max(np.abs(fft_result)))

# Use scatter plot for small points
plt.scatter(freqs, magnitude_db, s=1, c='b', marker='.')

plt.title('Hanning Window Frequency Response')
plt.xlabel('Bins (Original Scale)')
plt.ylabel('Magnitude (dB)')
plt.xlim(-10, 10)
plt.ylim(-60, 5)  # Adjust y-axis limits for better visibility

# Add vertical lines for null points
plt.axvline(-2, color='r', linestyle='--', label='Null Points (±2 bins)')
plt.axvline(2, color='r', linestyle='--')

plt.legend()
plt.grid(True)
plt.show()

print(f"Theoretical null-to-null main lobe width: {main_lobe_bins} bins")
