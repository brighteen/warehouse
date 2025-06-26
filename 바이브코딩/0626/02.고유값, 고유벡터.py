import numpy as np
import matplotlib.pyplot as plt

# 행렬과 벡터 정의
M = np.array([[2, 3], [2, 1]])
x = np.array([1, 1.5])
v = np.array([-1, 1])

# 행렬-벡터 곱셈
Mx = M @ x
Mv = M @ v

# 고유값과 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(M)
print("고유값 (Eigenvalues):", eigenvalues)
print("고유벡터 (Eigenvectors):", eigenvectors)
print(Mv)

# 시각화
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# 첫 번째 그래프: x와 Mx
axs[0].quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, color='blue', label='x')
axs[0].quiver(0, 0, Mx[0], Mx[1], angles='xy', scale_units='xy', scale=1, color='red', label='Mx')
axs[0].set_xlim(-4, 4)
axs[0].set_ylim(-4, 4)
axs[0].set_aspect('equal')
axs[0].legend()
axs[0].set_title('Mx')
axs[0].grid()

# 두 번째 그래프: v와 Mv
axs[1].quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v')
axs[1].quiver(0, 0, Mv[0], Mv[1], angles='xy', scale_units='xy', scale=1, color='green', label='Mv')
# 고유벡터 추가 (정규화된 형태로 표시)
for i in range(len(eigenvalues)):
    eig_vec = eigenvectors[:, i] * 2  # 시각화를 위해 크기 조정
    axs[1].quiver(0, 0, eig_vec[0], eig_vec[1], angles='xy', scale_units='xy', scale=1, color='purple', alpha=0.5, label=f'Eigenvector {i+1}' if i == 0 else None)
axs[1].set_xlim(-4, 4)
axs[1].set_ylim(-4, 4)
axs[1].set_aspect('equal')
axs[1].legend()
axs[1].set_title('Mv with Eigenvectors')
axs[1].grid()

plt.tight_layout()
plt.show()
