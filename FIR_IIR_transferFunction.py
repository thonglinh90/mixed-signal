import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
np.seterr(divide='ignore', invalid='ignore')

# Define the filter coefficients
fir_num = [1, 1, 1, 1, 1]  # Numerator coefficients
fir_den = [1]        # Denominator coefficient
iir_num = [1, 1]  # Numerator coefficients
iir_den = [1, -1]        # Denominator coefficient

# Compute the frequency response
# w = 2*pi*f
fir_w, fir_h = signal.freqz(fir_num, fir_den)
iir_w, iir_h = signal.freqz(iir_num, iir_den)

# Convert frequency to Hz (assuming a sample rate of 1 Hz)
fir_f = fir_w / (2 * np.pi)
iir_f = iir_w / (2 * np.pi)

#extract zeros and poles
zeros, poles, gain = signal.tf2zpk(iir_num, iir_den)
print(zeros,poles, gain)

# Plot the magnitude response
# plt.figure()
# plt.plot(freq, np.abs(h))
# plt.title('Magnitude Response of H = 1 + z^-1 + z^-2 + z^-3')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.grid(True)
# plt.show()

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('test')
ax1.plot(fir_f, np.abs(fir_h))
ax1.set(xlabel="Normalized Frequency", ylabel="Magnitude")
ax2.plot(iir_f, np.abs(iir_h))
ax2.set(xlabel="Normalized Frequency", ylabel="Magnitude")

# ax3.scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none', edgecolors='r', label='Zeros')
# ax3.scatter(np.real(poles), np.imag(poles), marker='x', color='b', label='Poles')
# unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
# plt.gca().add_artist(unit_circle)

# ax3.axhline(0, color='black', linewidth=0.5)
# ax3.axvline(0, color='black', linewidth=0.5)
# ax3.set(xlim = [-1.5, 1.5], ylim = [-1.5, 1.5], xlabel = 'Real Part', ylabel = 'Img part', title = 'Pole-Zero Plot of H(z) = 1 + z^-1 + z^-2 + z^-3')
# ax3.grid()
# ax3.legend()

print('Zeroes are',zeros)
print('Poles are',poles)
print('Angles of zeores are',np.angle(zeros)/(2*np.pi))
print('Angles of poles are',np.angle(poles)/(2*np.pi))
plt.show()