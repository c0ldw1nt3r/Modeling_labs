import math
from collections import Counter

# ===== ПАРАМЕТРЫ =====
A, B, C, M, Y0 = 6, 7, 3, 4096, 4001  # параметры генератора
I = 12
N = 7000        # объём выборки
K_BINS = 16     # число интервалов для Пирсона (только для χ², без вывода таблицы)
POKER_K = 8     # ваш вариант: покер-тест по 8 цифрам
ALPHA = 0.05

# ===== Генератор (ККМ) =====
def gen_qcg(A, B, C, M, Y0, n):
    y = Y0
    xs = []
    for _ in range(n):
        y = (A*y*y + B*y + C) % M
        xs.append(y / M)
    return xs

# ===== Пирсон (только χ², без выводов интервалов) =====
def chi2_pearson(xs, k_bins):
    cnt = [0] * k_bins
    for x in xs:
        j = min(int(x * k_bins), k_bins - 1)
        cnt[j] += 1
    n = len(xs)
    exp = n / k_bins
    chi2 = sum((c - exp) ** 2 / exp for c in cnt)
    df = k_bins - 1
    return chi2, df

# ===== Колмогоров =====
def kolmogorov(xs):
    n = len(xs)
    s = sorted(xs)
    D = 0.0
    for i, x in enumerate(s, 1):
        D = max(D, abs(i/n - x), abs(x - (i-1)/n))
    lam = (n ** 0.5) * D
    return D, lam

# ===== Покер-тест (k=8) по числу различных цифр r =====
def _stirling2_table(n):
    S = [[0] * (n + 1) for _ in range(n + 1)]
    S[0][0] = 1
    for i in range(1, n + 1):
        for r in range(1, i + 1):
            S[i][r] = r * S[i - 1][r] + S[i - 1][r - 1]
    return S

def _perm(n, k):
    p = 1
    for t in range(n, n - k, -1):
        p *= t
    return p

def poker_test(xs, k_digits=8):
    # «слово» — первые k десятичных цифр после точки
    words = [f"{int(x * (10**k_digits)):0{k_digits}d}" for x in xs]
    # r = количество разных цифр в слове
    obs_by_r = Counter(len(set(w)) for w in words)
    n = len(words)

    # теоретические вероятности p_r = 10P_r * S(k,r) / 10^k
    S = _stirling2_table(k_digits)
    probs = {r: _perm(10, r) * S[k_digits][r] / (10 ** k_digits)
             for r in range(1, min(10, k_digits) + 1)}

    # χ² по категориям r
    chi2 = 0.0
    for r, p in probs.items():
        exp = n * p
        obs = obs_by_r.get(r, 0)
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp

    df = len(probs) - 1
    return obs_by_r, probs, chi2, df

# ===== Основной запуск =====
if __name__ == "__main__":
    xs = gen_qcg(A, B, C, M, Y0, N)

    # Выборочные характеристики
    mean = sum(xs) / N
    var  = sum((x - mean) ** 2 for x in xs) / N  # смещённая дисперсия, как в теории U(0,1)

    # 1) Пирсон
    chi2p, df_p = chi2_pearson(xs, K_BINS)
    chi2_crit_95 = 36.415  # df=24 для K=25; при K=16 df=15 → 24.996 (при необходимости подставьте своё)
    # если K_BINS=16, лучше используйте 24.996; здесь оставлено как пример фиксированного порога

    # 2) Колмогоров
    D, lam = kolmogorov(xs)
    Dcrit_approx = 1.36 / math.sqrt(N)  # приближённый критический уровень для α≈0.05

    # 3) Покер-тест k=8
    obs_r, probs_r, chi2_poker, df_poker = poker_test(xs, POKER_K)
    chi2_poker_crit_95 = 16.919  # приблизительно для df=8 (уточните по вашей таблице, зависит от k)

    # ===== Вывод =====
    print("=== Результаты статистической проверки ===\n")
    print(f"Объём выборки: {N}")
    print(f"Среднее:   {mean:.6f}  (теор. 0.5)")
    print(f"Дисперсия: {var:.6f}  (теор. 1/12 ≈ 0.083333)\n")

    print(f"1) Пирсон (K={K_BINS}, df={df_p}): χ² = {chi2p:.4f}")
    print(f"   Сравнение с критическим значением: используйте таблицу χ² для df={df_p}")
    # При K_BINS=16 можно ориентироваться на χ²_кр(0.05,15) ≈ 24.996
    if K_BINS == 16:
        chi2_crit_95 = 24.996
        print(f"   χ²_кр(α=0.05, df=15) ≈ {chi2_crit_95}")
        print(f"   Решение: {'не отвергаем H0' if chi2p < chi2_crit_95 else 'отвергаем H0'}")
    print()

    print(f"2) Колмогоров: D = {D:.6f}, λ = {lam:.6f}")
    print(f"   Критическое (прибл.) D_кр ≈ 1.36/√n = {Dcrit_approx:.6f}")
    print(f"   Решение: {'не отвергаем H0' if D < Dcrit_approx else 'отвергаем H0'}\n")

    print(f"3) Покер-тест (k={POKER_K}, по числу различных цифр r):")
    print("   r |  O_r (набл) |  E_r (ожид) |   p_r")
    for r in range(1, min(10, POKER_K) + 1):
        p = probs_r[r]
        e = N * p
        o = obs_r.get(r, 0)
        print(f"  {r:1d} | {o:12d} | {e:11.3f} | {p:8.6f}")
    print(f"   χ² (покер) = {chi2_poker:.4f}, df ≈ {df_poker}")
    print(f"   Сравнение с χ²_кр(α=0.05, df≈{df_poker}) — по вашей таблице критических значений.")
