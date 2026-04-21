import numpy as np
import librosa
import matplotlib.pyplot as plt

y, fs = librosa.load('audio_con_ruido.wav')

y_fft = np.fft.fft(y)
n = len(y)
T = 1.0 / fs

frecuencias = np.fft.fftfreq(n, T)

plt.figure()
plt.plot(frecuencias[:n//2], np.abs(y_fft)[:n//2])
plt.title("Espectro de Frecuencia")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

indice_max = np.argmax(np.abs(y_fft[:n//2]))
frecuencia_dominante = frecuencias[indice_max]

print("Frecuencia dominante:", frecuencia_dominante, "Hz")
