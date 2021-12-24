from functions import *

if __name__ == '__main__':
    # параметры размерности N, оптический вихрь m, степень полинома Цернике n и радиус R
    N = 1000
    M = 65536
    m = -3
    n = 5
    R = 5
    # получаем отрезки r и радиальную фунцию f(r)
    h, r, f = discretize(R, m, n, N)
    # Амплитуда и фаза входной функции
    plot(r, f, "полином Цернике f(r)")
    # Построим теперь двумерную функцию
    # Получаем f(, φ)
    f_2D = radial_vortex_function_2d(f, m)
    plot_2d(f_2D, "полинома Цернике f(r, φ)")
    # Преобразование Ханкеля входной функции
    start_hankel_time = time.time()
    F = np.array([hankel_transform(r[i], r, f, m, h) for i in range(len(r))])
    F_2D = radial_vortex_function_2d(F, m)
    end_hankel_time = time.time()
    plot(r, F, "преобразование Ханкеля F(ρ)")
    plot_2d(F_2D, "преобразование Ханкеля F(ρ, θ)")
    print("Скорость преобразования Ханкеля, с = " + str(end_hankel_time - start_hankel_time))
    start_fft_time = time.time()
    fft = fft_2d(f_2D, h, 2 * N, M)
    end_fft_time = time.time()
    plot_2d(fft, "БПФ F(ρ, θ)")
    print("Скорость БПФ, с = " + str(end_fft_time - start_fft_time))
