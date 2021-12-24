import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import time


# Радиальный полином Цернике
def radial_polynomial_zernike(r, p, n):
    result = 0
    for k in range((n - p) // 2):
        result += (-1) ** k * np.math.factorial(n - k) * r ** (n - 2 * k) / \
                  (np.math.factorial(k) * np.math.factorial((n + p) / 2 - k) * np.math.factorial((n - p) / 2 - k))
    return result


# Входное поле f(r, φ) — полином Цернике
def f(r, p, n):
    return radial_polynomial_zernike(r, abs(p), n)


# Дискретизация отрезка координат и входной функции
# noinspection SpellCheckingInspection
def discretize(r, m, n, number):
    h = r / (number - 1)
    return h, np.array([i * h for i in range(number)]), np.array([f(i * h, m, n) for i in range(number)])


# Дискретизация двумерной функции
# noinspection SpellCheckingInspection
def radial_vortex_function_2d(func, m):
    n = len(func)
    f_2d = np.zeros((2 * n, 2 * n), dtype=complex)
    for j in range(2 * n):
        for k in range(2 * n):
            r = round(np.sqrt((j - n) ** 2 + (k - n) ** 2))
            phi = np.math.atan2(-(j - n), k - n) if j <= n else np.math.atan2(-(j - n), k - n) + 2 * np.math.pi
            f_2d[j][k] = func[r] * np.exp(1j * m * phi) if r < n else 0
    return f_2d


# Преобразование Ханкеля
def hankel_transform(rho, r, func, m, h):
    result = 0
    for i in range(len(r)):
        result += func[i] * scipy.special.jv(m, 2 * np.pi * r[i] * rho) * r[i]
    return 2 * np. pi / 1j ** m * result * h


# Одномерное быстрое преобразование Фурье
def fft_1d(function, h, n, m):
    # дополнение нулями, разбиение на две части и их обмен
    f_zeros = np.append(function[n // 2:], np.zeros(m - n))
    f_zeros = np.append(f_zeros, function[:n // 2])
    # БПФ
    fft = np.fft.fft(f_zeros, m)
    # разбиение fft на две половины, обмен местами и вырезание центральной части из получившегося вектора
    return h * np.append(fft[m - n // 2:], fft[:n // 2])


# Двумерное быстрое преобразование Фурье
def fft_2d(function, h, n, m):
    fft = np.zeros((n, n), dtype=complex)
    temp = np.zeros((n, n), dtype=complex)
    # проход по строкам
    for i in range(n):
        temp[i] = fft_1d(function[i], h, n, m)
    temp = temp.T
    # проход по столбцам
    for i in range(n):
        fft[i] = fft_1d(temp[i], h, n, m)
    return fft.T


# Отрисовка одномерных графиков
def plot(x, function, label, second_function=None, second_label=None):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, abs(function), label=label)
    if second_function is not None:
        plt.plot(x, abs(second_function), label=second_label)
    plt.title("Амплитуда")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(x, np.angle(function), label=label)
    if second_function is not None:
        plt.plot(x, np.angle(second_function), label=second_label)
    plt.title("Фаза")
    plt.grid()
    plt.legend()
    plt.show()


# Отрисовка двумерных графиков
def plot_2d(field, label):
    my_map = plt.get_cmap("plasma")
    plt.set_cmap(my_map)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(abs(field))
    plt.title("Амплитуда " + label, fontsize=9)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(field))
    plt.title("Фаза " + label, fontsize=9)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()
