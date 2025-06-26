import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

from math import sqrt

# 1. 공의 높이를 이차함수로 정의하는 함수
def h(t, a, b, c):
    return a * t**2 + b * t + c

# 1-1. 최고 높이와 그때의 시간(샘플링 방법)
def find_peak_sampling(a, b, c, t_start=0.0, t_end=8.0, dt=0.1):
    times = np.arange(t_start, t_end + dt, dt)
    heights = h(times, a, b, c)
    idx_max = np.argmax(heights)
    return times[idx_max], heights[idx_max]

# 1-2. 땅(높이 0)에 닿는 시간 계산 (근의 공식)
def find_roots(a, b, c):
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return []
    t1 = (-b + sqrt(discriminant)) / (2*a)
    t2 = (-b - sqrt(discriminant)) / (2*a)
    # 물리적으로 의미 있는 t>0인 해만 반환
    return sorted([t for t in (t1, t2) if t > 0])

# 1-3. 두 함수 비교 및 시각화
def compare_and_plot(params1, params2, t_max=10.0, dt=0.01):
    a1, b1, c1 = params1
    a2, b2, c2 = params2

    # 1-1, 1-2 결과 구하기
    t_peak1, h_peak1 = find_peak_sampling(a1, b1, c1)
    t_peak2, h_peak2 = find_peak_sampling(a2, b2, c2)

    roots1 = find_roots(a1, b1, c1)
    roots2 = find_roots(a2, b2, c2)

    print("원래 함수: h(t) = {:.2f}t² + {:.2f}t + {:.2f}".format(a1, b1, c1))
    print("  최고점: t = {:.2f} s, h = {:.2f} m".format(t_peak1, h_peak1))
    print("  지면 도달 시간:", ", ".join("{:.2f} s".format(t) for t in roots1))

    print("\n변경된 함수: h(t) = {:.2f}t² + {:.2f}t + {:.2f}".format(a2, b2, c2))
    print("  최고점: t = {:.2f} s, h = {:.2f} m".format(t_peak2, h_peak2))
    print("  지면 도달 시간:", ", ".join("{:.2f} s".format(t) for t in roots2))

    # 시각화
    ts = np.arange(0, t_max, dt)
    plt.figure(figsize=(8, 5))
    plt.plot(ts, h(ts, a1, b1, c1), label="원래 함수")
    plt.plot(ts, h(ts, a2, b2, c2), label="힘 강해진 함수", linestyle="--")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.scatter([t_peak1, t_peak2], [h_peak1, h_peak2], color=['blue', 'orange'])
    for t in roots1:
        plt.scatter(t, 0, color='blue')
    for t in roots2:
        plt.scatter(t, 0, color='orange')
    plt.title("축구공 높이 곡선 비교")
    plt.xlabel("시간 t (s)")
    plt.ylabel("높이 h(t) (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

# 실행 예시
if __name__ == "__main__":
    # 원래 함수 계수
    params_orig = (-0.5, 4.0, 1.0)
    # 힘이 강해진 경우 계수
    params_strong = (-0.3, 5.0, 1.5)

    compare_and_plot(params_orig, params_strong, t_max=8.0)
