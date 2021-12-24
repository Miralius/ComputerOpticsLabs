import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


# Модуль полинома Цернике
def z(n, p, r):
    return (1/6) * np.exp(- np.pi ** 2 * r ** 2) * np.pi ** 9 * r ** 8


# Получение разбиения интервала [a, b] и дискретизированной функции
def create_discretized_function(count, n, p, r, func):
    h = r / (count - 1)
    return h, np.array([i * h for i in range(count)]), np.array([func(n, p, i * h) for i in range(count)])


# Получение двумерного поля
def create_discretized_function_2d(m, func):
    n = len(func)
    f = np.zeros((2 * n, 2 * n), dtype=complex)
    for j in range(2 * n):
        for k in range(2 * n):
            r = round(np.sqrt((j - n) ** 2 + (k - n) ** 2))
            phi = np.arctan2(k - n, j - n)
            f[j][k] = func[r] * np.exp(1j * m * phi) if r < n else 0
    return f


# Преобразование Ханкеля
def hankel(func, m, h, rho, r):
    result = 0
    for i in range(len(r)):
        result += func[i] * sp.jv(m, 2 * np.pi * r[i] * rho) * r[i]
    return 2 * np. pi / 1j ** m * result * h


# Одномерное быстрое преобразование Фурье
def fft(n, m, h, function):
    # дополняем нулями, разбиваем на две части и меняем местами
    f_augmented = np.append(np.append(function[n // 2:], np.zeros(m - n)), function[:n // 2])
    # БПФ
    fft_result = h * np.fft.fft(f_augmented, m)
    # разбиваем fft на две половины, меняем местами и вырезаем центральную часть
    return np.append(fft_result[m - n // 2:], fft_result[:n // 2])


# Двумерное быстрое преобразование Фурье
def fft_2d(n, m, h, function):
    fft_rows = np.zeros((n, n), dtype=complex)
    fft_result = np.zeros((n, n), dtype=complex)
    # проходимся по строкам
    for i in range(n):
        fft_rows[i] = fft(n, m, h, function[i])
    fft_rows = fft_rows.T
    # проходимся по столбцам
    for i in range(n):
        fft_result[i] = fft(n, m, h, fft_rows[i])
    return fft_result.T


# Отрисовка одномерных графиков
def plot(x, first_function, first_label, second_function=None, second_label=None):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, abs(first_function), label=first_label)
    if second_function is not None:
        plt.plot(x, abs(second_function), label=second_label)
    plt.title("Амплитуда")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x, np.angle(first_function), label=first_label)
    if second_function is not None:
        plt.plot(x, np.angle(second_function), label=second_label)
    plt.title("Фаза")
    plt.grid()
    plt.legend()
    plt.show()


# Отрисовка двумерных графиков
def plot_2d(field, label):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(abs(field))
    plt.title("Амплитуда " + label, fontsize=10)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(field), vmin=-np.pi, vmax=np.pi)
    plt.title("Фаза " + label, fontsize=10)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()
