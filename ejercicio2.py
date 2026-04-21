import numpy as np
import librosa
import matplotlib.pyplot as plt

# 1. Cargar audio
y, fs = librosa.load('audio_con_ruido.wav')

# 2. Generar ruido (ruido blanco)
ruido = np.random.normal(0, 0.1, len(y))  # puedes cambiar 0.1 para más/menos ruido

# 3. Señal con ruido
y_ruido = y + ruido

# 4. FFT señal original
Y = np.fft.fft(y)
# 5. FFT señal con ruido
Y_ruido = np.fft.fft(y_ruido)

n = len(y)
T = 1/fs
frecuencias = np.fft.fftfreq(n, T)

# 6. Graficar
plt.figure(figsize=(10,6))

# Señal original
plt.subplot(2,1,1)
plt.plot(frecuencias[:n//2], np.abs(Y)[:n//2])
plt.title("Espectro original")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")

# Señal con ruido
plt.subplot(2,1,2)
plt.plot(frecuencias[:n//2], np.abs(Y_ruido)[:n//2])
plt.title("Espectro con ruido")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()