from functions import *
import time

if __name__ == '__main__':
    # размерности N и M, завихрение m и степени полинома Цернике p и n
    N = 1000
    M = 2048
    m = 2
    p = -3
    n = 5
    R = 2
    h, r, f_1D = create_discretized_function(N, n, p, R, z)
    plot(r, f_1D, "полином Цернике (1D)")
    f_2D = create_discretized_function_2d(m, f_1D)
    plot_2d(f_2D, "полинома Цернике (2D)")
    start_time = time.time()
    F_1D = np.array([hankel(f_1D, m, h, r[i], r) for i in range(len(r))])
    F_2D = create_discretized_function_2d(m, F_1D)
    end_time = time.time()
    plot(r, F_1D, "преобразование Ханкеля (1D)")
    plot_2d(F_2D, "преобразования Ханкеля (2D)")
    print("Время преобразования Ханкеля, с = " + str(end_time - start_time))
    start_time = time.time()
    fft = fft_2d(2 * N, M, h, f_2D)
    end_time = time.time()
    plot_2d(fft, "БПФ (2D)")
    print("Время БПФ, с = " + str(end_time - start_time))
