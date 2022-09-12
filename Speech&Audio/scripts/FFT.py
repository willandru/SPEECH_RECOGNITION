import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

def FFT(file):
    signal, sr = librosa.load(file, sr=22050) #sr*T -> 22050*30
    fft= np.fft.fft(signal)

    magnitud= np.abs(fft)
    frequency= np.linspace(0,sr,len(magnitud))
    plt.plot(frequency, magnitud)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitud")
    plt.show()
    