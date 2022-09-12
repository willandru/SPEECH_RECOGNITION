import librosa, librosa.display
import matplotlib.pyplot as plt

file="A1.m4a"

signal, sr = librosa.load(file, sr=22050) #sr*T -> 22050*30
librosa.display.waveshow(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitud")
plt.show()