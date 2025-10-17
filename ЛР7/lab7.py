

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import warnings
warnings.filterwarnings("ignore")

# ---------- Параметры (можно менять) ----------
a = 30          # правая граница (integer > 0)
x0 = 10         # стартовая позиция (0 < x0 < a)
n_sim = 5000    # число имитаций (рекомендуется >= 1000)
rng = np.random.default_rng(20251016)

# ---------- Функция одной симуляции (итерационная) ----------
def simulate_one(a, x0, rng, max_steps=10**7):
    x = x0
    t = 0
    while True:
        # условие остановки: достигнут 0 или a
        if x == 0 or x == a:
            return t
        # шаг: +-1 с prob 1/2
        step = 1 if rng.random() < 0.5 else -1
        x += step
        t += 1
        if t >= max_steps:
            # Защита от бесконечной петли (на практике не должно срабатывать)
            return t

# ---------- Моделирование (в цикле) ----------
taus = np.empty(n_sim, dtype=int)
for i in range(n_sim):
    taus[i] = simulate_one(a, x0, rng)

# ---------- Эмпирические характеристики ----------
mean_tau = np.mean(taus)
var_tau = np.var(taus, ddof=1)
median_tau = np.median(taus)
q25, q75 = np.percentile(taus, [25, 75])

# Теоретическое мат. ожидание для простого симметричного блуждания с абсорбцией:
# E[τ] = x0 * (a - x0)
theor_mean = x0 * (a - x0)

# ---------- Фит распределений ----------
# Для fitting используем:
# - нормальное (normal): fit -> (mu, sigma)
# - экспоненциальное (expon): fit -> (loc, scale) but for positive tau loc~0
# - лог-нормальное (lognorm): fit -> (s, loc, scale) where s = sigma of log, mu = ln(scale)

# Приведем данные к float
data = taus.astype(float)

# Нормальное приближение
norm_mu, norm_sigma = stats.norm.fit(data)
# Экспоненциальное приближение (скажем, с loc=0)
exp_loc, exp_scale = stats.expon.fit(data, floc=0)
# Логнормальное приближение (данные >0, tau может быть 0; добавим 1 для логарифма, но лучше сместить):
# Если есть нули, сдвинем на +1 (реально tau=0 возможен только если x0==0 или x0==a; здесь x0 внутренняя точка => tau>=1)
if np.any(data == 0):
    shift = 1.0
    data_for_logn = data + shift
else:
    shift = 0.0
    data_for_logn = data.copy()

ln_s, ln_loc, ln_scale = stats.lognorm.fit(data_for_logn, floc=0)  # фиксируем loc=0 для стабильности

# ---------- Проверки Колмогорова-Смирнова (kstest) для fitted parameters ----------
# Для K-S теста используем распределения с подогнанными параметрами.
ks_norm = stats.kstest(data, 'norm', args=(norm_mu, norm_sigma))
ks_exp = stats.kstest(data, 'expon', args=(exp_loc, exp_scale))
# Для lognormal используем преобразование обратно: тест к fitted lognorm
ks_logn = stats.kstest(data_for_logn, 'lognorm', args=(ln_s, ln_loc, ln_scale))

# ---------- Построение графиков: гистограмма + наложенные теоретические плотности ----------
fig1 = plt.figure("Гистограмма времен поглощения τ")
ax1 = fig1.add_subplot(1,1,1)
n_bins = int(math.sqrt(n_sim))  # правило выборки биннинга (примерное)
counts, bins, patches = ax1.hist(taus, bins=n_bins, density=True, alpha=0.6, label='Гистограмма (эмпир.)', edgecolor='black')

# координаты для плотностей
x = np.linspace(min(taus), max(taus), 1000)

# нормальная плотность
pdf_norm = stats.norm.pdf(x, loc=norm_mu, scale=norm_sigma)
ax1.plot(x, pdf_norm, 'r-', lw=2, label=f'Норм., μ={norm_mu:.2f}, σ={norm_sigma:.2f}')

# экспоненциальная плотность
pdf_exp = stats.expon.pdf(x, loc=exp_loc, scale=exp_scale)
ax1.plot(x, pdf_exp, 'g--', lw=2, label=f'Экспон., scale={exp_scale:.2f}')

# логнорм плотность (помним о shift)
x_logn = x + shift
pdf_logn = stats.lognorm.pdf(x_logn, ln_s, ln_loc, ln_scale)
ax1.plot(x, pdf_logn, 'm-.', lw=2, label=f'Логнорм., s={ln_s:.2f}')

ax1.set_xlabel('τ (в шагах)')
ax1.set_ylabel('Плотность (прибл.)')
ax1.set_title(f'Гистограмма τ, a={a}, x0={x0}, n={n_sim}')
ax1.legend()
ax1.grid(True)
plt.show(block=False)

# ---------- Эмпирическая функция распределения + теоретические cdf наложенные ----------
fig2 = plt.figure("Эмпирическая ФР τ")
ax2 = fig2.add_subplot(1,1,1)
# эмпирическая CDF
sorted_data = np.sort(taus)
ecdf_y = np.arange(1, n_sim+1) / n_sim
ax2.step(sorted_data, ecdf_y, where='post', label='Эмпирическая ФР')

# теоретические CDF по fitted параметрам
cdf_norm = stats.norm.cdf(x, loc=norm_mu, scale=norm_sigma)
cdf_exp = stats.expon.cdf(x, loc=exp_loc, scale=exp_scale)
cdf_logn = stats.lognorm.cdf(x + shift, ln_s, ln_loc, ln_scale)

ax2.plot(x, cdf_norm, 'r-', lw=2, label='Норм. CDF (fit)')
ax2.plot(x, cdf_exp, 'g--', lw=2, label='Эксп. CDF (fit)')
ax2.plot(x, cdf_logn, 'm-.', lw=2, label='Логнорм. CDF (fit)')

ax2.set_xlabel('τ (в шагах)')
ax2.set_ylabel('F(τ)')
ax2.set_title('Эмпирическая и приближённые функции распределения')
ax2.legend()
ax2.grid(True)
plt.show(block=False)

# ---------- Вывод таблицы результатов и интерпретация аппроксимаций ----------
print("Параметры моделирования: a =", a, ", x0 =", x0, ", n_sim =", n_sim)
print("\nЭмпирические характеристики τ:")
print(f"  Среднее (эмпир.): {mean_tau:.4f}")
print(f"  Дисперсия (эмпир.): {var_tau:.4f}")
print(f"  Медиана: {median_tau:.4f}, квантиль 25%={q25:.4f}, квантиль 75%={q75:.4f}")
print(f"\nТеоретическое E[τ] = x0*(a - x0) = {theor_mean:.4f}")
print(f"Отклонение эмпирического среднего от теоретического: {mean_tau - theor_mean:.4f}")

print("\nРезультаты подгонки распределений и K-S тестов:")
print(f" Нормальное: μ={norm_mu:.4f}, σ={norm_sigma:.4f}, K-S stat={ks_norm.statistic:.4f}, p-value={ks_norm.pvalue:.4f}")
print(f" Экспоненциальное (loc fixed=0): scale={exp_scale:.4f}, K-S stat={ks_exp.statistic:.4f}, p-value={ks_exp.pvalue:.4f}")
print(f" Логнормальное (fit with loc=0): s={ln_s:.4f}, scale={ln_scale:.4f}, K-S stat={ks_logn.statistic:.4f}, p-value={ks_logn.pvalue:.4f}")

# Интерпретация простая (на уровне 0.05)
alpha = 0.05
def interpret_ks(pval):
    if pval > alpha:
        return "может быть допущено (p > 0.05)"
    else:
        return "отвергается (p ≤ 0.05)"

print("\nИнтерпретация K-S (уровень значимости 0.05):")
print(" Нормальное распределение:", interpret_ks(ks_norm.pvalue))
print(" Экспоненциальное распределение:", interpret_ks(ks_exp.pvalue))
print(" Логнормальное распределение:", interpret_ks(ks_logn.pvalue))

# ---------- Доп. график: boxplot и плотность ядер (для наглядности) ----------
fig3 = plt.figure("Boxplot и KDE τ")
ax3 = fig3.add_subplot(1,1,1)
ax3.boxplot(taus, vert=False, widths=0.6)
ax3.set_xlabel('τ (в шагах)')
ax3.set_title('Boxplot для τ')
plt.show()
fig4 = plt.figure("KDE τ")
ax4 = fig4.add_subplot(1,1,1)

kde = stats.gaussian_kde(taus)
xx = np.linspace(min(taus), max(taus), 1000)
ax4.plot(xx, kde(xx), label='KDE (гладкая плотность)')
ax4.hist(taus, bins=40, density=True, alpha=0.3, edgecolor='black')
ax4.set_title('Плотность (KDE) и гистограмма')
ax4.set_xlabel('τ')
ax4.legend()
plt.show()
