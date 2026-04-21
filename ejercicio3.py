import numpy as np
import librosa
import matplotlib.pyplot as plt

y, fs = librosa.load('audio_con_ruido.wav')

Y = np.fft.fft(y)
n = len(y)
T = 1/fs
frecuencias = np.fft.fftfreq(n, T)

fc = 1000 
Y_filtrado = Y.copy()

Y_filtrado[np.abs(frecuencias) > fc] = 0

y_filtrado = np.fft.ifft(Y_filtrado)
y_filtrado = np.real(y_filtrado)

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(frecuencias[:n//2], np.abs(Y)[:n//2])
plt.title("Espectro original")

plt.subplot(2,1,2)
plt.plot(frecuencias[:n//2], np.abs(Y_filtrado)[:n//2])
plt.title("Espectro filtrado (pasa-bajo)")

plt.tight_layout()
plt.show()