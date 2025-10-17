import numpy as np
import math
import matplotlib.pyplot as plt

# ====== кусочная CDF ======
def F(x: float) -> float:
    if x <= 0.0:   return 0.0
    if x < 0.5:    return x*x
    if x < 1.0:    return 1.1*x - 0.3
    if x < 1.5:    return 0.4*x + 0.4
    return 1.0

# обратная функция (метод обратных функций)
def Finv(u: float) -> float:
    if u < 0.25:
        return math.sqrt(u)
    elif u < 0.8:
        return (u + 0.3)/1.1
    else:
        return (u - 0.4)/0.4

# генерация выборки
def generate_sample(N: int) -> np.ndarray:
    U = np.random.rand(N)
    return np.array([Finv(u) for u in U], dtype=float)

# χ² Пирсона
def pearson_chi2(sample: np.ndarray, K: int, xmin: float, xmax: float):
    counts, edges = np.histogram(sample, bins=K, range=(xmin, xmax))
    N = len(sample)
    p = np.array([F(edges[i+1]) - F(edges[i]) for i in range(K)], dtype=float)
    expected = N * p
    mask = expected > 0
    chi2 = float(np.sum((counts[mask] - expected[mask])**2 / expected[mask]))
    df = np.count_nonzero(mask) - 1
    return chi2, df, counts, edges, expected

# Колмогоров
def kolmogorov(sample: np.ndarray):
    N = len(sample)
    x = np.sort(sample)
    i = np.arange(1, N+1)
    Fn_right = i / N
    Fn_left  = (i-1) / N
    F_theor  = np.array([F(xi) for xi in x])
    Dplus  = np.max(Fn_right - F_theor)
    Dminus = np.max(F_theor - Fn_left)
    D = max(Dplus, Dminus)
    lam = math.sqrt(N)*D
    return D, Dplus, Dminus, lam

# ====== Основная программа ======
if __name__ == "__main__":
    N = 2000
    K = 25
    xmin, xmax = 0.0, 1.5

    x = generate_sample(N)

    mean = float(np.mean(x))
    var  = float(np.var(x, ddof=0))

    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("==================")
    print(f"Объём выборки: {N}")
    print(f"Число участков разбиения: {K}")
    print(f"Математическое ожидание (выборочное): {mean:.6f}")
    print(f"Дисперсия (выборочная): {var:.6f}\n")

    # --- Пирсон ---
    chi2, df, counts, edges, expected = pearson_chi2(x, K, xmin, xmax)
    print("КРИТЕРИЙ ПИРСОНА")
    print(f"χ² = {chi2:.4f}, df = {df}")
    crit_chi2 = 37.65  # для df=24 и α=0.05, если нет scipy
    print(f"χ²_кр(0.05, df={df}) ≈ {crit_chi2}")
    print("Итог:", "не отвергаем H0" if chi2 < crit_chi2 else "отвергаем H0")
    print()

    # --- Колмогоров ---
    D, Dplus, Dminus, lam = kolmogorov(x)
    Dcrit = 1.36 / math.sqrt(N)
    print("КРИТЕРИЙ КОЛМОГОРОВА")
    print(f"D = {D:.6f} (D+={Dplus:.6f}, D-={Dminus:.6f}), λ={lam:.6f}")
    print(f"D_кр(0.05) ≈ {Dcrit:.6f}")
    print("Итог:", "не отвергаем H0" if D < Dcrit else "отвергаем H0")
    print()

    # --- Табличка гистограммы ---
    print("ДАННЫЕ ГИСТОГРАММЫ")
    print("Интервал           O_i   E_i(теор)   Отн.частота   Накопл.O_i")
    print("-------------------------------------------------------------")
    rel = counts / N
    cum = np.cumsum(counts)
    for i in range(K):
        a, b = edges[i], edges[i+1]
        print(f"[{a:5.3f}; {b:5.3f})  {counts[i]:5d}   {expected[i]:10.2f}     {rel[i]:10.4f}     {cum[i]:6d}")

    # --- Графики ---
    # Гистограмма частот
    plt.hist(x, bins=K, density=True, edgecolor="black", alpha=0.7, color="blue")
    plt.title("Гистограмма частот")
    plt.xlabel("x")
    plt.ylabel("Частота")
    plt.show()

    # Эмпирическая и теоретическая функции распределения
    x_sorted = np.sort(x)
    F_emp = np.arange(1, N+1) / N
    F_theor = [F(val) for val in x_sorted]

    plt.step(x_sorted, F_emp, where="post", label="Эмпирическая", color="blue")
    plt.plot(x_sorted, F_theor, "r--", label="Теоретическая")
    plt.title("Эмпирическая и теоретическая функции распределения")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    plt.show()
