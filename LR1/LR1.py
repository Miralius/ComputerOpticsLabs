import numpy as np
import cmath
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D


def print_plot(title, label, x, y):
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.plot(x, y, label=label, linewidth=1.5, color="black")
    plt.legend()
    plt.show()


def print_3d_plot(title, func, x, y, z):
    fig = pylab.figure()
    axes = Axes3D(fig)
    axes.plot_surface(x, y, func(z))
    plt.title(title)
    pylab.show()


def make_data(a, b, p, q):
    h__x = (b - a) / n
    h__y = (q - p) / m
    x = np.arange(a, b, h__x)
    y = np.arange(p, q, h__y)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = kernel(y_grid, x_grid)
    return x_grid, y_grid, z_grid


def kernel(ksi, x, alpha=1):
    return np.exp(-alpha * abs(x + ksi * 1j))


def input_signal(x):
    return np.exp((x / 10) * 1j)


def kernel_matrix(A, m, n, q, p, a, b):
    h_y = (q - p) / m
    h_x = (b - a) / n
    for i in range(m):
        y_i = p + i * h_y
        for j in range(n):
            x_j = a + j * h_x
            A[i][j] = kernel(y_i, x_j)
    return A


def input_signal_vector(f, b, a, n):
    h_x = (b - a) / n
    for i in range(len(f)):
        x_i = a + i * h_x
        f[i] = input_signal(x_i)
    return f


def amplitude(z: np.complex128):
    result = np.sqrt(z.real ** 2 + z.imag ** 2)
    return float(result)


def amplitude_for_3d(z: np.complex128):
    result = np.sqrt(z.real ** 2 + z.imag ** 2)
    return result


def phase(z: np.complex128):
    result = cmath.phase(z)
    return result


m, n = 1000, 1000
a, b = -5, 5
p, q = -5, 5
A = np.zeros((m, n), np.complex128)
f = np.zeros((m, 1), np.complex128)

A = kernel_matrix(A, m, n, q, p, a, b)
f = input_signal_vector(f, b, a, n)

h_x = (b - a) / n
x = np.arange(a, b, h_x)
F = A.dot(f) * h_x

amplitude_first, phase_first = [], []
for i in f:
    amplitude_first.append(amplitude(i))
    phase_first.append(phase(i))


print_plot("График амплитуды входного сигнала:",
          "амплитуда", x, amplitude_first)

print_plot("График фазы входного сигнала",
          "фаза", x, phase_first)

amplitude_res, phase_res = [], []
for i in F:
    amplitude_res.append(amplitude(i))
    phase_res.append(phase(i))

print_plot("График амплитуды выходного сигнала:",
          "амплитуда", x, amplitude_res)

print_plot("График фазы выходного сигнала:",
          "фаза", x, phase_res)

x, y, z = make_data(a, b, p, q)


print_3d_plot("Амплитуда ядра", amplitude_for_3d, x, y, z)
print_3d_plot("Фаза ядра", np.angle, x, y, z)
