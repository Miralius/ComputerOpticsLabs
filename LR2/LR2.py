import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def gauss(x):
    return np.exp(-x ** 2)


def gauss2D(x, y):
    return np.exp(-x ** 2 - y ** 2)


def fastFiniteFourierTransform(y, b, a, M, N):
    h = (b - a) / (N - 1)
    # Добавление нулей
    zeros = np.zeros(int((M - N) / 2))
    y = np.concatenate((zeros, y, zeros), axis=None)
    # Смена частей вектора
    middle = int(len(y) / 2)
    y = np.concatenate((y[middle:], y[:middle]))
    # БПФ
    Y = np.fft.fft(y, axis=-1) * h
    # Смена частей вектора
    middle = int(len(Y) / 2)
    Y = np.concatenate((Y[middle:], Y[:middle]))
    # Выделение центральных N элементов
    Y = Y[int((M - N) / 2): int((M - N) / 2 + N)]
    # Пересчет области задания функции
    interval = abs(N ** 2 / (4 * a * M))
    return Y, interval


def fastFiniteFourierTransform2D(Z, a, b, N, M):
    for i in range(N):
        Z[:, i], interval = fastFiniteFourierTransform(Z[:, i], b, a, M, N)
    for i in range(N):
        Z[i, :], interval = fastFiniteFourierTransform(Z[i, :], b, a, M, N)
    return Z, interval


def fastFiniteFourierTransformIntegrate(a, b, M, N):
    # Определение границ
    interval = abs(N ** 2 / (4 * a * M))
    # Шаг дисретизации
    step = 2 * interval / (N - 1)
    Y = np.zeros(N, dtype=np.complex128)
    for i in range(len(Y)):
        u = -interval + i * step
        Y[i] = integrate.quad(lambda x: np.exp(-(x ** 2) - 2 * np.pi * u * x * 1j), a, b)[0]
    return Y, interval


def inputFunc(x):
    return (np.exp(2 * np.pi * complex(0, 1) * x) + np.exp(-5 * np.pi * complex(0, 1) * x))


def inputFunc2D(x, y):
    return (np.exp(2 * np.pi * complex(0, 1) * (x + y)) + np.exp(-5 * np.pi * complex(0, 1) * (x + y)))


def analytic(x):
    return -(2 * np.sin(10 * np.pi * x)) / (np.pi * (2 * x + 5)) + (np.sin(10 * np.pi * x)) / (
            np.pi * (x - 1))


def analytic2D(x, y):
    return ((2 * np.sin(10 * np.pi * y)) / (np.pi * (2 * y + 5)) * (
            (2 * np.sin(10 * np.pi * x)) / (np.pi * (2 * x + 5))) + (
                    (2 * np.sin(10 * np.pi * y)) / (np.pi * (2 * y - 3))) * (
                    (2 * np.sin(10 * np.pi * x)) / (np.pi * (2 * x - 3))))


def renderPlot(x, y, firstTitle, secondTitle, firstLabel, secondLabel):
    _, arr = plt.subplots(1, 2, figsize=(10, 5))
    arr[0].plot(x, np.absolute(y), color='b', label=firstLabel)
    arr[0].legend()
    arr[0].grid()
    arr[0].set_title(firstTitle)
    arr[1].plot(x, np.angle(y), color='b', label=secondLabel)
    arr[1].legend()
    arr[1].grid()
    arr[1].set_title(secondTitle)
    plt.show()

def renderPlots(x, y1, y2, firstTitle, secondTitle, firstLabel, secondLabel):
    _, arr = plt.subplots(1, 2, figsize=(10, 5))
    arr[0].plot(x, np.absolute(y1), color='b', label=firstLabel)
    arr[0].plot(x, np.absolute(y2), color='r', label=secondLabel)
    arr[0].legend()
    arr[0].grid()
    arr[0].set_title(firstTitle)
    arr[1].plot(x, np.angle(y1), color='b', label=firstLabel)
    arr[1].plot(x, np.angle(y2), color='r', label=secondLabel)
    arr[1].legend()
    arr[1].grid()
    arr[1].set_title(secondTitle)
    plt.show()


M = 4096
N = 512
a = 5

# Графики амплитуды и фазы Гауссова пучка
x = np.linspace(-a, a, N)
y = gauss(x)
renderPlot(x, y, 'f(x) = exp(-x^2)', 'f(x) = exp(-x^2)', 'Амплитуда', 'Фаза')

# Графики амплитуды и фазы БПФ
x = np.linspace(-a, a, N, endpoint=False)
y = gauss(x)
Y_fin_fft, interval = fastFiniteFourierTransform(y, a, -a, M, N)
x = np.linspace(-interval, interval, N, endpoint=False)
renderPlot(x, Y_fin_fft, 'f(x) = exp(-x^2) БПФ', 'f(x) = exp(-x^2) БПФ', 'Амплитуда', 'Фаза')

# Графики амплитуды и фазы численного преобразования Фурье
x = np.linspace(-a, a, N, endpoint=False)
Y_fin_fft_integrate, interval = fastFiniteFourierTransformIntegrate(-a, a, M, N)
x = np.linspace(-interval, interval, N, endpoint=False)
renderPlot(x, Y_fin_fft_integrate, 'F(x) = exp(-x^2) ЧМ', 'F(x) = exp(-x^2) ЧМ', 'Амплитуда', 'Фаза')

# Графики амплитуды и фазы численного преобразования Фурье и БПФ
renderPlots(x, Y_fin_fft, Y_fin_fft_integrate, 'Амплитуда exp(-x^2)', 'Фаза exp(-x^2)', 'БПФ', 'ЧМ')

# Графики амплитуды и фазы светового поля
x = np.linspace(-a, a, N)
y = inputFunc(x)
renderPlot(x, y, 'f(x) = exp(2πix) + exp(-5πix)', 'f(x) = exp(2πix) + exp(-5πix)', 'Амплитуда', 'Фаза')

# Графики амплитуды и фазы финитного преобразования Фурье
x = np.linspace(-a, a, N, endpoint=False)
y = inputFunc(x)
Y_fin_fft_lf, interval = fastFiniteFourierTransform(y, a, -a, M, N)
x = np.linspace(-interval, interval, N)
renderPlot(x, Y_fin_fft_lf, 'f(x) = exp(2πix) + exp(-5πix) БПФ', 'f(x) = exp(2πix) + exp(-5πix) БПФ',
           'Амплитуда', 'Фаза')

# Графики амплитуды и фазы результата преобразования Фурье (аналитически)
x = np.linspace(-interval, interval, N, endpoint=False)
y = analytic(x)
renderPlot(x, y, 'f(x) = exp(2πix) + exp(-5πix) Аналитика', 'f(x) = exp(2πix) + exp(-5πix) Аналитика',
           'Амплитуда', 'Фаза')

# Графики амплитуды и фазы БПФ и аналитического решения
renderPlots(x, Y_fin_fft_lf, y, 'Амплитуда exp(2πix) + exp(-5πix)', 'Фаза exp(2πix) + exp(-5πix)', 'БПФ', 'Аналитика')

# Графики для двумерного случая
lspace = np.linspace(-a, a, N, endpoint=False)
X, Y = np.meshgrid(lspace, lspace)
Z = gauss2D(X, Y)
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
amp = arr[0].imshow(np.absolute(Z), cmap='hot', interpolation='nearest')
arr[0].set_title('exp(-x^2-y^2) Амплитуда')
phase = arr[1].imshow(np.angle(Z), cmap='hot', interpolation='nearest')
arr[1].set_title('exp(-x^2-y^2) Фаза')
fig.colorbar(phase, ax=arr[1])
lspace = np.linspace(-a, a, N, endpoint=False)
X, Y = np.meshgrid(lspace, lspace)
Z = gauss2D(X, Y).astype(np.complex128)
Z_fin_fft, area = fastFiniteFourierTransform2D(Z, -a, a, N, M)
lspace = np.linspace(-area, area, N, endpoint=False)
X, Y = np.meshgrid(lspace, lspace)
plt.show()
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
amp = arr[0].imshow(np.absolute(Z_fin_fft), cmap='hot', interpolation='nearest')
arr[0].set_title('exp(-x^2-y^2) Амплитуда: БПФ')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(Z_fin_fft), cmap='hot', interpolation='nearest')
arr[1].set_title('exp(-x^2-y^2) Фаза: БПФ')
fig.colorbar(phase, ax=arr[1])
lspace = np.linspace(-a, a, N, endpoint=False)
X, Y = np.meshgrid(lspace, lspace)
Z = inputFunc2D(X, Y)
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
amp = arr[0].imshow(np.absolute(Z), cmap='hot', interpolation='nearest')
arr[0].set_title('exp(2πi(x+y)) + exp(-5πi(x+y)) Амплитуда')
fig.colorbar(amp, ax=arr[0])
arr[1].imshow(np.angle(Z), cmap='hot', interpolation='nearest')
arr[1].set_title('exp(2πi(x+y)) + exp(-5πi(x+y)) Фаза')
fig.colorbar(phase, ax=arr[1])
lspace = np.linspace(-a, a, N, endpoint=False)
X, Y = np.meshgrid(lspace, lspace)
Z = inputFunc2D(X, Y).astype(np.complex128)
Z_fin_fft, area = fastFiniteFourierTransform2D(Z, -a, a, N, M)
lspace = np.linspace(-area, area, N, endpoint=False)
X, Y = np.meshgrid(lspace, lspace)
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
amp = arr[0].imshow(np.absolute(Z_fin_fft), cmap='hot', interpolation='nearest')
arr[0].set_title('exp(2πi(x+y)) + exp(-5πi(x+y)) Abs: БПФ')
fig.colorbar(amp, ax=arr[0])
phase = arr[1].imshow(np.angle(Z_fin_fft), cmap='hot', interpolation='nearest')
arr[1].set_title('exp(2πi(x+y)) + exp(-5πi(x+y)) Фаза: БПФ')
fig.colorbar(phase, ax=arr[1])
lspace = np.linspace(-area, area, N, endpoint=False)
X, Y = np.meshgrid(lspace, lspace)
Z = analytic2D(X, Y)
fig, arr = plt.subplots(1, 2, figsize=(10, 5))
arr[0].imshow(np.absolute(Z), cmap='hot', interpolation='nearest')
arr[0].set_title('Амплитуда аналитики')
fig.colorbar(amp, ax=arr[0])
arr[1].imshow(np.angle(Z), cmap='hot', interpolation='nearest')
arr[1].set_title('Фаза аналитики')
fig.colorbar(phase, ax=arr[1])
plt.show()
