import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import time

N = 1000
M = 32768
m = 2
p = -3
n = 5
R = 5
hx = R/N

def z(n, p, r):
    p = abs(p)
    summa = 0
    for k in range((n - p) // 2):
        summa += (-1) ** k * sp.factorial(n - k) * r ** (n - 2 * k) / (sp.factorial(k) * sp.factorial((n + p) / 2 - k) * sp.factorial((n - p) / 2 - k))
    return abs(summa)

def Four(f):
    F = np.fft.fftshift(np.fft.fft(prepare(f))) * hx
    F = F[len(F) // 2 - N // 2: len(F) // 2 + N // 2]
    return F

def Four2d(f):
    F = np.array([Four(x) for x in f])
    F = np.array([Four(x) for x in np.transpose(F)])
    return np.transpose(F)


def funcTo2D(func):
    number = len(func)
    arr_2d = np.zeros((2 * number, 2 * number), dtype=np.complex)
    j, k = np.meshgrid(np.arange(0, 2 * number), np.arange(0, 2 * number))
    j = j - number
    k = k - number
    dist = np.round(np.sqrt(j ** 2 + k ** 2)).astype(np.int)
    mask = dist < number
    arr_2d[mask] = func[dist[mask]]
    f = np.arctan2(k, j)
    return arr_2d * np.exp(complex(0, 1) * m * f)


def hankel(x, y, m):
    new_x = x
    Y = np.zeros(N, dtype=np.complex128)
    for i, j in zip(new_x, range(len(x))):
        Y[j] = np.sum(y * sp.jv(m, 2 * np.pi * x * i) * x * (R / N))
    return Y * (2 * np.pi / (complex(0, 1) ** m))


def showPlots(x, y, nameX, nameY):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, abs(y))
    plt.legend()
    plt.grid(which='major', linewidth=2)
    plt.grid(which='minor')
    plt.minorticks_on()
    plt.xlabel(nameX)
    plt.ylabel(nameY)
    plt.subplot(1, 2, 2)
    plt.plot(x, np.angle(y))
    plt.grid(which='major', linewidth=2)
    plt.grid(which='minor')
    plt.minorticks_on()
    plt.legend()
    plt.xlabel(nameX)
    plt.ylabel(nameY)
    plt.show()

def show2Plots(x):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(abs(x))
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(x), vmin=-np.pi, vmax=np.pi)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()

def prepare(f):
    k = (M-N)//2
    f = np.insert(f, 0, np.zeros(k))
    f = np.append(f, np.zeros(k))
    return np.fft.fftshift(f)


r = np.linspace(0, R, N, endpoint=False)

f1 = z(n, p, r)
showPlots(r, f1, "r", "f(r)")
f2 = funcTo2D(f1)
show2Plots(f2)
start_time = time.time()
F1 = hankel(r, f1, m)
F2 = funcTo2D(F1)
end_time = time.time()
showPlots(r, F1, "r", "F(r)")
show2Plots(F2)
print("Время преобразования Ханкеля, с = " + str(end_time - start_time))
start_time = time.time()
fft = Four2d(f2)
end_time = time.time()
show2Plots(fft)
print("Время БПФ, с = " + str(end_time - start_time))