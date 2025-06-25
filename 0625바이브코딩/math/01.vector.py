
def vector_example():
    """
    벡터 정의
    """
    a_list = [1,2,3] # 리스트
    a_array = np.array([1,2,3]) # 넘파이 배열
    row_vector = np.array([[1,2,3]]) # 행 벡터
    column_vector = np.array([[1],[2],[3]]) # 열 벡터

    print(f"a_list: {a_list}, type: {type(a_list)}, shape: {np.shape(a_list)}")
    print(f"a_array: {a_array}, type: {type(a_array)}, shape: {np.shape(a_array)}")
    print(f"row_vector: {row_vector}, type: {type(row_vector)}, shape: {np.shape(row_vector)}")
    print(f"column_vector: {column_vector}, type: {type(column_vector)}, shape: {np.shape(column_vector)}")
    pass

def vector_calc1():
    """
    간단한 벡터 연산
    """
    v = np.array([1,2])
    w = np.array([4,-6])
    u = np.array([0,3,6,9])
    print(f"v: {v}, w: {w}, u: {u}")
    print(f"v + w: {v + w}")
    pass

def vector_calc2():
    """
    행벡터와 열벡터의 연산
    """
    v = np.array([[4,5,6]])  # 행 벡터
    w = np.array([[10],[20],[30]])  # 열 벡터
    print(f"v: {v},  type: {type(v)}, shape: {np.shape(v)}")
    print(f"w: {w},  type: {type(w)}, shape: {np.shape(w)}")
    print(f"\nv + w: {v + w}")  # 브로드캐스팅 연산
    print(f"\nv * w: {v * w}")
    pass

def vector_visualization():
    """
    두 벡터의 연산 시각화
    """
    v = np.array([1, 2])
    w = np.array([5, -4])
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='v')
    plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='b', label='w')
    plt.quiver(0, 0, v[0] + w[0], v[1] + w[1], angles='xy', scale_units='xy', scale=1, color='g', label='v + w')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('Vector Addition Visualization')
    plt.show()
    pass

def scalar_product():
    """
    스칼라와 벡터의 곱 연산 후 시각화
    """
    scalar = -2
    v = np.array([1,1])
    print(f"scalar: {scalar}, v: {v}")
    print(f"scalar * v: {scalar * v}")  # 스칼라와 벡터의 곱

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='v')
    plt.quiver(0, 0, scalar * v[0], scalar * v[1], angles='xy', scale_units='xy', scale=1, color='b', label='scalar * v')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('Scalar Multiplication Visualization')
    plt.show()
    pass

def vector_transpose():
    """
    벡터의 전치 연산
    """
    v = np.array([[1, 2, 3]])  # 행 벡터
    print(f"v: {v}, type: {type(v)}, shape: {np.shape(v)}")
    # 전치 연산
    v_transpose = v.T
    print(f"v_transpose: {v_transpose}, type: {type(v_transpose)}, shape: {np.shape(v_transpose)}")
    # 두번 전치
    v_transpose_twice = v_transpose.T
    print(f"v_transpose_twice: {v_transpose_twice}, type: {type(v_transpose_twice)}, shape: {np.shape(v_transpose_twice)}")
    pass

if __name__ == "__main__":
    import numpy as np

    # vector_example()
    # vector_calc1()
    # vector_calc2()
    # vector_visualization()
    # scalar_product()
    vector_transpose()