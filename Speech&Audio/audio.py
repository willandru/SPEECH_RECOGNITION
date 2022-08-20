import sounddevice as sd
from time import time
import pandas as pd
import numpy as np
import scipy.io.wavfile as waves


def grabar():
	grabacion = sd.rec(int(300*fs), samplerate=fs, channels=2)
	return grabacion

def reproducir(senal):
	sd.play(senal, fs)

def guardar(nombre, senal):
	titulo=nombre+'.wav'
	waves.write(titulo, fs, senal)
def cargar(nombre):
	titulo=nombre+'.wav'
	muestreo, sonido = waves.read(titulo)
	return sonido


