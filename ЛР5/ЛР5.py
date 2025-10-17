import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, weibull_min, kstest
from scipy.special import gamma as gamma_func

# ====== ПАРАМЕТРЫ ======
n = 1000     # объем выборки
k = 20       # количество интервалов

# --- Параметры ГАММА-распределения ---
alpha = 2.0   # параметр формы
beta = 2.0    # параметр масштаба

# --- Параметры ВЕЙБУЛЛА ---
c = 1.5       # параметр формы
scale = 2.0   # параметр масштаба

# ============================================================
#                     ГАММА-РАСПРЕДЕЛЕНИЕ
# ============================================================

gamma_sample = gamma.rvs(a=alpha, scale=beta, size=n)

# Эмпирические оценки
mean_gamma = np.mean(gamma_sample)
var_gamma = np.var(gamma_sample, ddof=1)

# Теоретические значения
theor_mean_gamma = alpha * beta
theor_var_gamma = alpha * (beta ** 2)

# Проверка согласия (Колмогоров)
ks_gamma = kstest(gamma_sample, 'gamma', args=(alpha, 0, beta))

# ====== ВЫВОД РЕЗУЛЬТАТОВ ======
print("=== ГАММА-РАСПРЕДЕЛЕНИЕ ===")
print(f"Параметры: α = {alpha}, β = {beta}")
print(f"Теоретическое мат. ожидание: {theor_mean_gamma:.4f}")
print(f"Теоретическая дисперсия:     {theor_var_gamma:.4f}")
print(f"Эмпирическое мат. ожидание:  {mean_gamma:.4f}")
print(f"Эмпирическая дисперсия:      {var_gamma:.4f}")
print(f"Критерий Колмогорова: статистика = {ks_gamma.statistic:.4f}, p-value = {ks_gamma.pvalue:.4f}")
print()

# ====== ГИСТОГРАММА ======
plt.figure("Гистограмма Гамма-распределения")
plt.hist(gamma_sample, bins=k, density=True, alpha=0.6, color='skyblue', label='Эмпирическая гистограмма')
x = np.linspace(min(gamma_sample), max(gamma_sample), 200)
plt.plot(x, gamma.pdf(x, a=alpha, scale=beta), 'r-', lw=2, label='Теоретическая плотность')
plt.title('Гамма-распределение')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.show(block=False)

# ====== ЭМПИРИЧЕСКАЯ И ТЕОРЕТИЧЕСКАЯ ФР ======
plt.figure("Функция распределения (Гамма)")
sorted_gamma = np.sort(gamma_sample)
emp_cdf_gamma = np.arange(1, n + 1) / n
plt.plot(sorted_gamma, emp_cdf_gamma, label='Эмпирическая ФР')
plt.plot(x, gamma.cdf(x, a=alpha, scale=beta), 'r--', label='Теоретическая ФР')
plt.title('Эмпирическая и теоретическая функции распределения (Гамма)')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.show(block=False)

# ============================================================
#                     РАСПРЕДЕЛЕНИЕ ВЕЙБУЛЛА
# ============================================================

weibull_sample = weibull_min.rvs(c=c, scale=scale, size=n)

# Эмпирические оценки
mean_weibull = np.mean(weibull_sample)
var_weibull = np.var(weibull_sample, ddof=1)

# Теоретические значения
theor_mean_weibull = scale * gamma_func(1 + 1 / c)
theor_var_weibull = scale**2 * (gamma_func(1 + 2 / c) - gamma_func(1 + 1 / c)**2)

# Проверка согласия (Колмогоров)
ks_weibull = kstest(weibull_sample, 'weibull_min', args=(c, 0, scale))

# ====== ВЫВОД РЕЗУЛЬТАТОВ ======
print("=== РАСПРЕДЕЛЕНИЕ ВЕЙБУЛЛА ===")
print(f"Параметры: c = {c}, scale = {scale}")
print(f"Теоретическое мат. ожидание: {theor_mean_weibull:.4f}")
print(f"Теоретическая дисперсия:     {theor_var_weibull:.4f}")
print(f"Эмпирическое мат. ожидание:  {mean_weibull:.4f}")
print(f"Эмпирическая дисперсия:      {var_weibull:.4f}")
print(f"Критерий Колмогорова: статистика = {ks_weibull.statistic:.4f}, p-value = {ks_weibull.pvalue:.4f}")
print()

# ====== ГИСТОГРАММА ======
plt.figure("Гистограмма распределения Вейбулла")
plt.hist(weibull_sample, bins=k, density=True, alpha=0.6, color='lightgreen', label='Эмпирическая гистограмма')
x = np.linspace(min(weibull_sample), max(weibull_sample), 200)
plt.plot(x, weibull_min.pdf(x, c=c, scale=scale), 'r-', lw=2, label='Теоретическая плотность')
plt.title('Распределение Вейбулла')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.show(block=False)

# ====== ЭМПИРИЧЕСКАЯ И ТЕОРЕТИЧЕСКАЯ ФР ======
plt.figure("Функция распределения (Вейбулл)")
sorted_weibull = np.sort(weibull_sample)
emp_cdf_weibull = np.arange(1, n + 1) / n
plt.plot(sorted_weibull, emp_cdf_weibull, label='Эмпирическая ФР')
plt.plot(x, weibull_min.cdf(x, c=c, scale=scale), 'r--', label='Теоретическая ФР')
plt.title('Эмпирическая и теоретическая функции распределения (Вейбулл)')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.legend()
plt.show()
