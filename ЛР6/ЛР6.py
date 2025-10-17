# -*- coding: utf-8 -*-
"""
Монтe-Карло + аналитика для задачи воздушного боя (вариант в задаче).
Зависимости: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from math import isclose

# ---------------- ПАРАМЕТРЫ моделирования ----------------
n_sim = 100000     # число моделирований (рекомендуется >= 10000; можно увеличить)
alpha_conf = 0.05  # уровень значимости -> доверительная вероятность beta = 0.95

# Параметры задачи (вводи свои p1 и p2)
p1 = 0.4  # вероятность, что бомбардировщик сбивает данного истребителя при выстреле
p2 = 0.6  # вероятность, что выживший истребитель сбивает бомбардировщик при ответном выстреле

rng = np.random.default_rng(12345)  # генератор случайных чисел (фиксируем seed для воспроизводимости)


# ---------------- Аналитические формулы ----------------
def analytic_probabilities(p1, p2):
    # Обозначения:
    # X_i = 1 если i-й истребитель сбит (бомбардировщиком), P(X_i=1) = p1, они независимы
    # Если X_i = 0, то i-й истребитель стреляет по бомбардировщику и сбивает его с вероятностью p2 (независимо)
    # A — сбит бомбардировщик
    # B — сбиты оба истребителя
    # C — сбит хотя бы один истребитель
    # D — сбит хотя бы один самолет (любой из трёх)
    # E — сбит ровно один истребитель
    # F — сбит ровно один самолет (из трёх)

    # Вероятности для X_i:
    # P(X_i=1) = p1, P(X_i=0) = 1-p1

    # A: бомбардировщик сбит iff хотя бы один из выживших истребителей попадёт в него.
    # Вероятность того, что i-й истребитель НЕ сбьёт бомбардировщик = p1 + (1-p1)*(1-p2) = 1 - (1-p1)*p2
    prob_A = 1 - (1 - (1 - p1) * p2) ** 2
    # Альтернативная запись: prob_A = 1 - [1 - (1-p1)*p2]^2

    # B: оба истребителя сбиты => оба X_i = 1
    prob_B = p1 ** 2

    # C: хотя бы один истребитель сбит = 1 - P(ни один не сбит) = 1 - (1-p1)^2
    prob_C = 1 - (1 - p1) ** 2

    # D: хотя бы один самолет (из трёх) сбит.
    # Противоположное: ни один не сбит -> оба истребителя не сбиты (X1=X2=0) и при этом ни один из выживших не сбил бомбардировщика:
    prob_none = (1 - p1) ** 2 * (1 - p2) ** 2
    prob_D = 1 - prob_none

    # E: ровно один истребитель сбит -> биноминальное распределение для X1+X2:
    prob_E = 2 * p1 * (1 - p1)

    # F: ровно один самолет (из трёх) сбит.
    # Случаи:
    # 1) ровно один истребитель сбит и бомбардировщик выжил (т.е. surviving fighter не попал):
    #    2 * p1 * (1-p1) * (1-p2)
    # 2) бомбардировщик сбит, и оба истребителя выжили (X1=X2=0) -> (1-p1)^2 * [1 - (1-p2)^2]
    prob_F = 2 * p1 * (1 - p1) * (1 - p2) + (1 - p1) ** 2 * (1 - (1 - p2) ** 2)

    return {
        'A': prob_A,
        'B': prob_B,
        'C': prob_C,
        'D': prob_D,
        'E': prob_E,
        'F': prob_F
    }


# ---------------- Моделирование Монте-Карло ----------------
def simulate_once(p1, p2, rng):
    # Возвращает кортеж булевых значений (A,B,C,D,E,F) для одного моделирования
    # 1) бомбардировщик стреляет по обоим истребителям:
    x1 = rng.random() < p1  # True если 1-й истребитель сбит
    x2 = rng.random() < p1  # True если 2-й истребитель сбит

    # 2) если истребитель не сбит (x= False), то он стреляет и может сбить бомбардировщик с prob p2
    y1 = False
    y2 = False
    if not x1:
        y1 = rng.random() < p2
    if not x2:
        y2 = rng.random() < p2

    bomber_down = (y1 or y2)
    fighters_down_count = int(x1) + int(x2)
    total_down_count = int(bomber_down) + fighters_down_count

    A = bomber_down
    B = (fighters_down_count == 2)
    C = (fighters_down_count >= 1)
    D = (total_down_count >= 1)
    E = (fighters_down_count == 1)
    F = (total_down_count == 1)

    return A, B, C, D, E, F


def monte_carlo(n, p1, p2, rng):
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
    # Векторизированный вариант для скорости:
    # Сначала моделируем выстрелы бомбардировщика по двум истребителям:
    u1 = rng.random(n)
    u2 = rng.random(n)
    X1 = u1 < p1
    X2 = u2 < p1

    # Для тех, кто выжил (X==False) моделируем его выстрел по бомберу
    # Сгенерируем матрицу случайных чисел для Y (независимые)
    v1 = rng.random(n)
    v2 = rng.random(n)
    Y1 = np.logical_and(~X1, v1 < p2)
    Y2 = np.logical_and(~X2, v2 < p2)

    bomber_down = np.logical_or(Y1, Y2)
    fighters_down_count = X1.astype(int) + X2.astype(int)
    total_down_count = bomber_down.astype(int) + fighters_down_count

    counts['A'] = np.count_nonzero(bomber_down)
    counts['B'] = np.count_nonzero(fighters_down_count == 2)
    counts['C'] = np.count_nonzero(fighters_down_count >= 1)
    counts['D'] = np.count_nonzero(total_down_count >= 1)
    counts['E'] = np.count_nonzero(fighters_down_count == 1)
    counts['F'] = np.count_nonzero(total_down_count == 1)

    # Частоты и доли
    estimates = {k: counts[k] / n for k in counts}
    return counts, estimates


# ---------------- Доверительный интервал Clopper-Pearson ----------------
def clopper_pearson(k, n, conf=0.95):
    # Возвращает (lower, upper) для вероятности при k успехах в n испытаниях
    alpha = 1 - conf
    if n == 0:
        return 0.0, 1.0
    if k == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lower, upper


# ---------------- Запуск моделирования ----------------
analytic = analytic_probabilities(p1, p2)
counts, estimates = monte_carlo(n_sim, p1, p2, rng)

# Вывод результатов + доверительные интервалы
print(f"Параметры: p1 = {p1}, p2 = {p2}, n_sim = {n_sim}, доверительная вероятность = {1-alpha_conf:.2f}\n")
print("{:>3} {:>12} {:>12} {:>20} {:>10}".format("Мер", "Оценка", "К-во успехов", "95%-ДИ", "Аналит."))
for key in ['A','B','C','D','E','F']:
    k_succ = counts[key]
    est = estimates[key]
    lo, hi = clopper_pearson(k_succ, n_sim, conf=1-alpha_conf)
    anal = analytic[key]
    inside = lo <= anal <= hi
    print("{:>3} {:12.6f} {:12d}  [{:8.6f}, {:8.6f}]   {:10.6f}   Попадание величины в ДИ={}".format(
        key, est, k_succ, lo, hi, anal, inside
    ))

# ---------------- Графики в отдельных окнах ----------------
# 1) гистограмма числа сбитых самолётов (0..3)
# Подсчёт количества сбитых: total_down_count
plt.figure("Гистограмма: число сбитых самолётов")
# повторим векторизацию, чтобы получить массив total_down_count (повтор, но недолго)
u1 = rng.random(n_sim)
u2 = rng.random(n_sim)
X1 = u1 < p1
X2 = u2 < p1
v1 = rng.random(n_sim)
v2 = rng.random(n_sim)
Y1 = np.logical_and(~X1, v1 < p2)
Y2 = np.logical_and(~X2, v2 < p2)
bomber_down = np.logical_or(Y1, Y2)
total_down_count = bomber_down.astype(int) + X1.astype(int) + X2.astype(int)
values, bins_edges = np.histogram(total_down_count, bins=np.arange(-0.5, 4.5, 1))
centers = (bins_edges[:-1] + bins_edges[1:]) / 2
plt.bar(centers, values, width=0.8)
plt.xticks([0,1,2,3])
plt.xlabel("Число сбитых самолётов")
plt.ylabel("Частота")
plt.title("Гистограмма числа сбитых самолётов (по моделированию)")
plt.grid(True)
plt.show(block=False)

# 2) столбчатая диаграмма эмпирических vs аналитических вероятностей для событий A..F
plt.figure("Эмпирические vs Аналитические вероятности A..F")
labels = ['A','B','C','D','E','F']
emp = [estimates[l] for l in labels]
an = [analytic[l] for l in labels]
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, emp, width, label='Эмпирическим')
plt.bar(x + width/2, an, width, label='Аналитическим', alpha=0.7)
plt.xticks(x, labels)
plt.ylabel("Вероятность")
plt.title("Эмпирические оценки vs Аналитические значения")
plt.legend()
plt.grid(axis='y')
plt.show()

