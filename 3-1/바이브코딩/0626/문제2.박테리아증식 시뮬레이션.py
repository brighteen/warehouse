import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 2-1. 박테리아 수 계산 함수
def N(t, N0=100, r=0.15):
    """시간 t(시간) 후 박테리아 수를 계산"""
    return N0 * math.exp(r * t)

# 2-2. 목표 개체수(N_target)가 되는 시간 계산 함수 (로그 이용)
def time_to_reach(N_target, N0=100, r=0.15):
    """
    N(t) = N_target 이 되는 t를 계산.
    해석적 해: t = ln(N_target/N0) / r
    """
    return math.log(N_target / N0) / r

# 2-3. 성장률 비교 및 시각화
def compare_growth(N0=100, r1=0.15, r2=0.10, N_target=300):
    # 5시간, 10시간 후 수
    n5 = N(5, N0, r1)
    n10 = N(10, N0, r1)
    print(f"2-1. 5시간 후 개체수: {n5:.2f}")
    print(f"     10시간 후 개체수: {n10:.2f}\n")

    # 목표치(3배) 달성 시간
    t1 = time_to_reach(N_target, N0, r1)
    t2 = time_to_reach(N_target, N0, r2)
    print(f"2-2. 성장률 {r1*100:.0f}% 일 때 300마리가 되는 시간: {t1:.2f}시간")
    print(f"2-3. 성장률 {r2*100:.0f}% 일 때 300마리가 되는 시간: {t2:.2f}시간")
    print(f"     더 걸리는 시간: {t2 - t1:.2f}시간\n")

    # 시각화
    t_max = max(t1, t2) * 1.1  # 최대 시간의 110%로 여유를 둠
    ts = np.linspace(0, t_max, 500)
    Ns1 = [N(t, N0, r1) for t in ts]
    Ns2 = [N(t, N0, r2) for t in ts]

    plt.figure(figsize=(8, 5))
    plt.plot(ts, Ns1, label=f"r = {r1*100:.0f}%")
    plt.plot(ts, Ns2, '--', label=f"r = {r2*100:.0f}%")
    plt.axhline(N_target, color='gray', linewidth=1, linestyle=':')
    plt.scatter([t1, t2], [N_target, N_target], color=['blue','orange'])
    plt.text(t1, N_target*1.02, f"{t1:.2f}h", ha='center', color='blue')
    plt.text(t2, N_target*1.02, f"{t2:.2f}h", ha='center', color='orange')

    plt.title("박테리아 지수적 성장 비교")
    plt.xlabel("시간 t (시간)")
    plt.ylabel("박테리아 수 N(t)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    compare_growth()
