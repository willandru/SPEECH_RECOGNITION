import audiosegment
import numpy as np
import matplotlib.pyplot as plt
#...
seg = audiosegment.from_file("P9.mp4")
freqs, times, amplitudes = seg.spectrogram(window_length_s=0.03, overlap=0.05)
amplitudes = 10* np.log10(amplitudes + 1e-9)
# Plot
plt.pcolormesh(times, freqs, amplitudes)
plt.xlabel("Time in Seconds")
plt.ylabel("Frequency in Hz")
plt.show()
