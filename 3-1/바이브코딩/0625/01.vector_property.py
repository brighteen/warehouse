
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
    print(f"\nv_transpose: {v_transpose}, type: {type(v_transpose)}, shape: {np.shape(v_transpose)}")
    # 두번 전치
    v_transpose_twice = v_transpose.T
    print(f"\nv_transpose_twice: {v_transpose_twice}, type: {type(v_transpose_twice)}, shape: {np.shape(v_transpose_twice)}")
    pass

def vector_distance():
    """
    벡터의 크기 계산
    """
    v = np.array([1,2,3,7,8,9])
    print(f"v: {v}, type: {type(v)}, shape: {np.shape(v)}, 길이: {len(v)}")
    v_length = np.linalg.norm(v)  # 벡터의 크기
    print(f"v_length: {v_length}")

def my_calculate_distance(v):
    """
    넘파이를 사용하지 않고 벡터의 크기를 계산하는 함수
    """
    v_length = 0
    for i in v:
        v_length += i ** 2 # 제곱합
    v_length = v_length ** 0.5 # 루트 취하기
    return v_length

def unit_vector_calc():
    """
    단위 벡터 계산
    """
    v = [1, 2, 3]
    print(f"원본 벡터 v: {v}")
    
    v_length = my_calculate_distance(v)  # 벡터의 크기
    print(f"벡터의 크기: {v_length}")
    
    # 단위 벡터 계산 (각 성분을 벡터의 크기로 나누기)
    unit_vector = [component / v_length for component in v]
    print(f"단위 벡터: {unit_vector}")
    
    # 단위벡터의 크기 확인
    unit_vector_length = my_calculate_distance(unit_vector)
    print(f"단위 벡터의 크기: {unit_vector_length}")
    
    return unit_vector

def vector_product():
    """
    벡터의 내적 계산
    """
    s = -1
    v = np.array([1, 2, 3, 4])
    w = np.array([5,6,7,8])
    print(f"v: {v}, w: {w}")
    v = v * s
    print(f"v: {v}, w: {w}")
    dot_product = np.dot(v, w)  # 벡터의 내적
    print(f"v . w (dot product): {dot_product}")

def vector_분배법칙():
    """
    벡터의 분배법칙
    """
    v = np.array([ 0,1,2 ])
    w = np.array([ 3,5,8 ])
    u = np.array([ 13,21,34 ])
    res1 = np.dot( v, w+u )
    res2 = np.dot( v,w ) + np.dot( v,u )
    print(f"v: {v}, w: {w}, u: {u}")
    print(f"v . (w + u): {res1}, v . w + v . u: {res2}")
    pass

def vector_similirarity():
    """
    벡터의 유사도 계산
    """
    v = np.array([0,1,2])
    w = np.array([3,5,8])
    print(f"v: {v}, w: {w}")

    dot_product = np.dot(v, w)  # 내적
    print(f"v . w (dot product): {dot_product}")
    v_length = np.linalg.norm(v)  # 벡터 v의 크기
    w_length = np.linalg.norm(w)  # 벡터 w의 크기
    print(f"v_length: {v_length}, w_length: {w_length}")

    similarity = dot_product / (v_length * w_length)  # 코사인 유사도
    print(f"코사인 유사도: {similarity}")

if __name__ == "__main__":
    import numpy as np

    # vector_example()
    # vector_calc1()
    # vector_calc2()
    # vector_visualization()
    # scalar_product()
    # vector_transpose()
    # vector_distance()
    # my_calculate_distance()
    # unit_vector_calc()
    # vector_product()
    # vector_분배법칙()
    vector_similirarity()