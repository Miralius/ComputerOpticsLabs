import numpy as np
import matplotlib.pyplot as plt
from scipy.special import *
import time


# Входная функция
def inputFunc(r, n, p):
    res = 0
    for k in range(0, int((n - p) / 2) + 1):
        numerator = (-1 ** k) * np.math.factorial(n - k) * (r ** (n - 2 * k))
        denominator = np.math.factorial(k) * np.math.factorial((n + p) / 2 - k) * np.math.factorial((n - p) / 2 - k)
        res += numerator / denominator
    return res


# Построение двумерной функции
def funcTo2D(func):
    n = len(func)
    arr_2d = np.zeros((2 * n, 2 * n), dtype=np.complex)
    j, k = np.meshgrid(np.arange(0, 2 * n), np.arange(0, 2 * n))
    j = j - n
    k = k - n
    dist = np.round(np.sqrt(j ** 2 + k ** 2)).astype(np.int)
    mask = dist < n
    arr_2d[mask] = func[dist[mask]]
    f = np.arctan2(k, j)
    return arr_2d * np.exp(complex(0, 1) * m * f)


# Преобразование Ханкеля
def transformOfHankel(x, y, m):
    new_x = x
    Y = np.zeros(N, dtype=np.complex128)
    for i, j in zip(new_x, range(len(x))):
        Y[j] = np.sum(y * jv(m, 2 * np.pi * x * i) * x * (R / N))
    return Y * (2 * np.pi / (complex(0, 1) ** m))


# БПФ
def finiteFFT(y, b, a, M, N):
    h_x = (b - a) / (N - 1)

    # Добавление нулей:
    zeros = np.zeros(int((M - N) / 2))
    y = np.concatenate((zeros, y, zeros), axis=None)

    # Свап частей вектора:
    middle = int(len(y) / 2)
    y = np.concatenate((y[middle:], y[:middle]))

    # БПФ:
    Y = np.fft.fft(y, axis=-1) * h_x

    # Свап частей вектора:
    middle = int(len(Y) / 2)
    Y = np.concatenate((Y[middle:], Y[:middle]))

    # Выделение центральных N отсчетов:
    Y = Y[int((M - N) / 2): int((M - N) / 2 + N)]

    # Пересчет области задания функции:
    new_board = abs(N ** 2 / (4 * b * M))
    return Y, new_board


# Двумерное преобразование Фурье через БПФ
def finiteFFT2d(Z, a, b, N, M):
    for i in range(N):
        Z[:, i], new_bound = finiteFFT(Z[:, i], b, a, M, N)
    for i in range(N):
        Z[i, :], new_bound = finiteFFT(Z[i, :], b, a, M, N)
    return Z, new_bound


R = 5
n = 4
p = 2
m = 2
N = 256
M = 2048

x = np.linspace(0, R, N)
x2 = np.linspace(0, R, N)

y = inputFunc(x, n, p)
_, arr = plt.subplots(1, 2, figsize=(15, 5))
arr[0].plot(x, np.absolute(y), color='b')
arr[0].grid()
arr[0].set_title('Амплитуда')
arr[1].plot(x, np.angle(y), color='b')
arr[1].grid()
arr[1].set_title('Фаза')
plt.show()

y2d = funcTo2D(y)
fig, arr = plt.subplots(1, 2, figsize=(15, 5))
amp = arr[0].imshow(np.absolute(y2d), cmap='hot', interpolation='nearest')
arr[0].set_title('Амплитуда восстановленного изображения')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(y2d), cmap='hot', interpolation='nearest')
arr[1].set_title('Фаза восстановленного изображения')
fig.colorbar(phase, ax=arr[1])
plt.show()

start = time.time()
y_hankel = transformOfHankel(x, y, m)
y_hankel_2D = funcTo2D(y_hankel)
end = time.time()
print("Преобразование Ханкеля: %sсек" % (end - start))

_, arr = plt.subplots(1, 2, figsize=(15, 5))
arr[0].plot(x, np.absolute(y_hankel), color='b')
arr[0].grid()
arr[0].set_title('Амплитуда после преобразования Ханкеля')
arr[1].plot(x, np.angle(y_hankel), color='b')
arr[1].grid()
arr[1].set_title('Фаза после преобразования Ханкеля')
plt.show()

fig, arr = plt.subplots(1, 2, figsize=(15, 5))
amp = arr[0].imshow(np.absolute(y_hankel_2D), cmap='hot', interpolation='nearest')
arr[0].set_title('Ханкель: Амплитуда восстановленного изображения')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(y_hankel_2D), cmap='hot', interpolation='nearest')
arr[1].set_title('Ханкель: Фаза восстановленного изображения')
fig.colorbar(phase, ax=arr[1])

N = 2 * N
start = time.time()
y_fourier, _ = finiteFFT2d(y2d, 0, R, N, M)
end = time.time()
print("БПФ: %sсек" % (end - start))

fig, arr = plt.subplots(1, 2, figsize=(15, 5))
amp = arr[0].imshow(np.absolute(y_fourier), cmap='hot', interpolation='nearest')
arr[0].set_title('БПФ: Амплитуда восстановленного изображения')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(y_fourier), cmap='hot', interpolation='nearest')
arr[1].set_title('БПФ: Фаза восстановленного изображения')
fig.colorbar(phase, ax=arr[1])
plt.show()
