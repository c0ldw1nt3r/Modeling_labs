import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Параметры распределения
# =========================
MU = 3.5
SIG2 = 0.9
SIG = math.sqrt(SIG2)

N = 2000      # объём выборки (>= 1000)
K = 25        # число интервалов (15 или 25)
METHOD = "clt"       # "clt" или "approx"

# =========================
# Теоретические функции
# =========================
SQRT2 = math.sqrt(2.0)

def phi(z: float) -> float:
    """CDF стандартного нормального Φ(z)."""
    return 0.5 * (1.0 + math.erf(z / SQRT2))

def normal_cdf(x: float, mu: float, sig: float) -> float:
    return phi((x - mu) / sig)

# --- Инверсная стандартная нормальная (аппроксимация Асклама) ---
def inv_phi(p: float) -> float:
    """Приближение Асклама инверсии Φ^{-1}(p) для p∈(0,1)."""
    if p <= 0.0:
        return -1e308
    if p >= 1.0:
        return 1e308

    a1 = -3.969683028665376e+01
    a2 =  2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 =  1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 =  2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 =  1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 =  6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 =  4.374664141464968e+00
    c6 =  2.938163982698783e+00

    d1 =  7.784695709041462e-03
    d2 =  3.224671290700398e-01
    d3 =  2.445134137142996e+00
    d4 =  3.754408661907416e+00

    plow  = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / \
               ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / \
               (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)
    q = math.sqrt(-2*math.log(1-p))
    return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / \
             ((((d1*q+d2)*q+d3)*q+d4)*q+1)

# =========================
# Генераторы N(mu, sig^2)
# =========================
def normal_clt(mu: float, sig: float, n: int) -> np.ndarray:
    """ЦПТ: sum_{i=1}^{12} U_i - 6 ~ N(0,1)."""
    u = np.random.rand(n, 12).sum(axis=1) - 6.0
    return mu + sig * u

def normal_approx(mu: float, sig: float, n: int) -> np.ndarray:
    """Метод аппроксимации: x = mu + sig * Φ^{-1}(U)."""
    U = np.random.rand(n)
    Z = np.array([inv_phi(p) for p in U])
    return mu + sig * Z

# =========================
# Критерии
# =========================
def pearson_chi2(sample: np.ndarray, K: int, mu: float, sig: float):
    """χ² Пирсона для нормального распределения (равные по ширине бины)."""
    # берём диапазон ±4σ, чтобы покрыть почти всё
    xmin, xmax = mu - 4*sig, mu + 4*sig
    counts, edges = np.histogram(sample, bins=K, range=(xmin, xmax))
    N = len(sample)
    # Теоретические вероятности по CDF
    p = np.array([normal_cdf(edges[i+1], mu, sig) - normal_cdf(edges[i], mu, sig)
                  for i in range(K)], dtype=float)
    expected = N * p
    mask = expected > 0
    chi2 = float(np.sum((counts[mask] - expected[mask])**2 / expected[mask]))
    df = np.count_nonzero(mask) - 1  # параметры известны (mu,sigma заданы)
    return chi2, df, counts, edges, expected

def kolmogorov(sample: np.ndarray, mu: float, sig: float):
    N = len(sample)
    x = np.sort(sample)
    i = np.arange(1, N+1)
    Fn_right = i / N
    Fn_left  = (i-1) / N
    F_theor  = np.array([normal_cdf(xi, mu, sig) for xi in x])
    Dplus  = float(np.max(Fn_right - F_theor))
    Dminus = float(np.max(F_theor - Fn_left))
    D = max(Dplus, Dminus)
    lam = math.sqrt(N) * D
    return D, Dplus, Dminus, lam

# =========================
# Основной сценарий
# =========================
if __name__ == "__main__":
    if METHOD == "clt":
        sample = normal_clt(MU, SIG, N)
    else:
        sample = normal_approx(MU, SIG, N)

    mean = float(np.mean(sample))
    var  = float(np.var(sample, ddof=0))

    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("==================")
    print(f"Объём выборки: {N}")
    print(f"Число интервалов: {K}")
    print(f"Математическое ожидание (выбор.): {mean:.6f}  (теор. {MU})")
    print(f"Дисперсия (выбор.):              {var:.6f}  (теор. {SIG2})\n")

    # --- Пирсон ---
    chi2, df, counts, edges, expected = pearson_chi2(sample, K, MU, SIG)
    print("Критерий Пирсона (хи-квадрат)")
    print(f"χ² = {chi2:.4f}, df = {df}")
    # если нет scipy, используем табличное значение сами
    if K == 25:
        chi2crit = 36.415  # χ²_кр(0.05, df=24)
    else:
        chi2crit = 24.996  # χ²_кр(0.05, df=15)
    print(f"χ²_кр(α=0.05, df={df}) ≈ {chi2crit}")
    print("Итог:", "не отвергаем H0" if chi2 < chi2crit else "отвергаем H0")
    print()

    # --- Колмогоров ---
    D, Dplus, Dminus, lam = kolmogorov(sample, MU, SIG)
    Dcrit = 1.36 / math.sqrt(N)  # приближение для α≈0.05
    print("Критерий Колмогорова")
    print(f"D = {D:.6f}  (D+={Dplus:.6f}, D-={Dminus:.6f});  λ = √n·D = {lam:.6f}")
    print(f"D_кр(α≈0.05) ≈ {Dcrit:.6f}")
    print("Итог:", "не отвергаем H0" if D < Dcrit else "отвергаем H0")
    print()

    # --- Таблица гистограммы (если нужна в отчёт) ---
    print("Данные гистограммы (интервал, O_i, E_i)")
    for i in range(K):
        a, b = edges[i], edges[i+1]
        print(f"[{a:6.3f}; {b:6.3f})  {counts[i]:5d}   {expected[i]:9.2f}")
    print()

    # --- Графики ---
    # Гистограмма частот (синие столбцы) + теоретическая плотность
    xmin, xmax = MU - 4*SIG, MU + 4*SIG
    xs = np.linspace(xmin, xmax, 500)
    pdf = (1/(SIG*math.sqrt(2*math.pi))) * np.exp(-0.5*((xs-MU)/SIG)**2)

    plt.figure(figsize=(8,5))
    plt.hist(sample, bins=K, range=(xmin, xmax), density=True, edgecolor="black", color="blue", alpha=0.85)
    plt.plot(xs, pdf, linewidth=2, label="Теоретическая плотность")
    plt.title("Гистограмма частот и теоретическая плотность N(3.5, 0.9)")
    plt.xlabel("x")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

    # Эмпирическая и теоретическая функции распределения
    sample_sorted = np.sort(sample)
    F_emp = np.arange(1, N+1) / N
    F_theor = [normal_cdf(v, MU, SIG) for v in sample_sorted]

    plt.figure(figsize=(8,5))
    plt.step(sample_sorted, F_emp, where="post", label="Эмпирическая F_n(x)", color="blue")
    plt.plot(sample_sorted, F_theor, "r--", label="Теоретическая F(x)")
    plt.title("Эмпирическая и теоретическая функции распределения")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
