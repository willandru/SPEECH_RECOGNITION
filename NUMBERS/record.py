import pyaudio
import wave

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS=1
RATE =16000

p= pyaudio.PyAudio()

stream= p.open( format=FORMAT, channels= CHANNELS, rate= RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)

print("start recording")