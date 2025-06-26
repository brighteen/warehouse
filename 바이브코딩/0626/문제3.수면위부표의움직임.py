import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise1
import math
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False
# --- 파라미터 설정 ---
# 원래 모델 파라미터 (3-1, 3-2)
A1 = 0.5
omega1 = 0.8 * np.pi
phi = 4 * np.pi
offset = 1.2

# 변경 모델 파라미터 (3-3)
A2 = A1 * 2         # 진폭 2배
omega2 = omega1 * 2 # 각진동수 2배

# 감쇠 + Perlin 노이즈 파라미터 (3-4)
decay_rate = 0.1     # 감쇠 계수 λ
noise_scale = 0.5    # Perlin 노이즈 스케일
noise_initial = 0.1  # 초기 노이즈 진폭

# 시간 범위 (원래 모델 주기의 4배)
T1 = 2 * np.pi / omega1
t_max = T1 * 4
ts = np.linspace(0, t_max, 1000)

# --- 함수 정의 ---
def H(t, A, omega, phi, offset):
    """부표 높이 계산"""
    return A * np.sin(omega * t + phi) + offset

def period(omega):
    """주기 계산"""
    return 2 * np.pi / omega

def extrema(A, offset):
    """최대/최소 높이 계산"""
    return offset + A, offset - A

# --- 3-1. 특정 시점 높이 출력 ---
print("3-1. 특정 시점 부표 높이")
for t in (1, 3, 5):
    h_val = H(t, A1, omega1, phi, offset)
    print(f"  t = {t}s → H(t) = {h_val:.4f} m")
print()

# --- 3-2. 원래 모델 주기 및 최대/최소 높이 출력 ---
T_orig = period(omega1)
H_max_orig, H_min_orig = extrema(A1, offset)
print("3-2. 원래 모델의 주기 및 극값")
print(f"  주기 T₁ = {T_orig:.4f} s")
print(f"  최대 높이 Hₘₐₓ = {H_max_orig:.4f} m")
print(f"  최소 높이 Hₘᵢₙ = {H_min_orig:.4f} m")
print()

# --- 3-3. 변경 모델 주기 및 최대/최소 높이 출력 ---
T_mod = period(omega2)
H_max_mod, H_min_mod = extrema(A2, offset)
print("3-3. 변경 모델의 주기 및 극값")
print(f"  주기 T₂ = {T_mod:.4f} s")
print(f"  최대 높이 Hₘₐₓ' = {H_max_mod:.4f} m")
print(f"  최소 높이 Hₘᵢₙ' = {H_min_mod:.4f} m")
print()

# --- 3-4. 감쇠 + Perlin 노이즈 시뮬레이션용 배경 출력 ---
# 진폭 및 노이즈 진폭의 초기/최종 값을 계산하여 출력
env_start = A1
env_end = A1 * math.exp(-decay_rate * t_max)
noise_env_start = noise_initial
noise_env_end = noise_initial * math.exp(-decay_rate * t_max)
print("3-4. 감쇠 및 노이즈 진폭 변화")
print(f"  초기 진폭 A(0) = {env_start:.4f}, 최종 진폭 A({t_max:.1f}) ≈ {env_end:.4f}")
print(f"  초기 노이즈 진폭 N(0) = {noise_env_start:.4f}, "
      f"최종 노이즈 진폭 N({t_max:.1f}) ≈ {noise_env_end:.4f}")
print()

# --- 그래프 그리기 ---
# 3-3: 원래 모델, 변경 모델
H_orig_vals = H(ts, A1, omega1, phi, offset)
H_mod_vals  = H(ts, A2, omega2, phi, offset)

# 3-4: 감쇠 + Perlin 노이즈
H_damped = []
for t in ts:
    A_decay = A1 * np.exp(-decay_rate * t)
    noise_val = pnoise1(t * noise_scale)
    noise_env = noise_initial * np.exp(-decay_rate * t)
    H_damped.append(A_decay * np.sin(omega1 * t + phi)
                    + offset
                    + noise_env * noise_val)
H_damped = np.array(H_damped)

plt.figure(figsize=(10, 5))
plt.plot(ts, H_orig_vals, label="원래 모델 (A=0.5, ω=0.8π)", color='tab:blue')
plt.plot(ts, H_mod_vals, '--', label="변경 모델 (A=1.0, ω=1.6π)", color='tab:orange')
plt.plot(ts, H_damped, alpha=0.7, label="감쇠 + Perlin 노이즈", color='tab:green')

plt.title("부표 움직임 비교: 원래·변경·감쇠+노이즈 모델")
plt.xlabel("시간 t (초)")
plt.ylabel("부표 높이 H(t) (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
