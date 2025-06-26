def check_linear_independence_T():
    """
    벡터 집합 T의 선형 독립성 판별
    T = {[8, -4, 14, 6], [4, 6, 0, 3], [14, 2, 4, 7], [13, 2, 9, 8]}
    """
    # 벡터 집합 T 정의 (열벡터들을 행렬의 열로 배치)
    T = np.array([
        [8, 4, 14, 13],
        [-4, 6, 2, 2],
        [14, 0, 4, 9],
        [6, 3, 7, 8]
    ])
    
    print("벡터 집합 T:")
    print(T)
    print(f"T의 형태: {T.shape}")
    
    # 방법 1: 행렬식(determinant) 계산
    det_T = np.linalg.det(T)
    print(f"\n행렬 T의 행렬식: {det_T}")
    
    if abs(det_T) < 1e-10:  # 부동소수점 오차 고려
        print("행렬식이 0이므로 벡터들은 선형 종속입니다.")
        linear_dependent = True
    else:
        print("행렬식이 0이 아니므로 벡터들은 선형 독립입니다.")
        linear_dependent = False
    
    # 방법 2: 계수(rank) 계산
    rank_T = np.linalg.matrix_rank(T)
    print(f"\n행렬 T의 계수(rank): {rank_T}")
    print(f"벡터의 개수: {T.shape[1]}")
    
    if rank_T < T.shape[1]:
        print("계수가 벡터 개수보다 작으므로 벡터들은 선형 종속입니다.")
        linear_dependent = True
    else:
        print("계수가 벡터 개수와 같으므로 벡터들은 선형 독립입니다.")
        linear_dependent = False
    
    # 방법 3: 동차 선형 방정식 Tx = 0의 해 확인
    print("\n동차 선형 방정식 Tx = 0의 해 확인:")
    try:
        # 영벡터가 아닌 해가 존재하는지 확인
        eigenvalues = np.linalg.eigvals(T)
        zero_eigenvalues = np.sum(np.abs(eigenvalues) < 1e-10)
        print(f"0에 가까운 고유값의 개수: {zero_eigenvalues}")
        
        if zero_eigenvalues > 0:
            print("영벡터가 아닌 해가 존재하므로 벡터들은 선형 종속입니다.")
        else:
            print("자명한 해만 존재하므로 벡터들은 선형 독립입니다.")
    except:
        print("고유값 계산 중 오류 발생")
    
    return not linear_dependent

def find_linear_combination_coefficients():
    """
    선형 종속인 경우, 선형 결합 계수 찾기
    c1*v1 + c2*v2 + c3*v3 + c4*v4 = 0 (모든 계수가 0이 아닌 경우)
    """
    T = np.array([
        [8, 4, 14, 13],
        [-4, 6, 2, 2],
        [14, 0, 4, 9],
        [6, 3, 7, 8]
    ])
    
    print("선형 결합 계수 찾기:")
    print("c1*v1 + c2*v2 + c3*v3 + c4*v4 = 0")
    
    # SVD를 사용하여 null space 찾기
    U, s, Vt = np.linalg.svd(T)
    
    # 특이값이 0에 가까운 경우의 개수 확인
    tolerance = 1e-10
    null_space_dim = np.sum(s < tolerance)
    
    if null_space_dim > 0:
        print(f"Null space의 차원: {null_space_dim}")
        # 가장 작은 특이값에 해당하는 오른쪽 특이벡터가 해
        coefficients = Vt[-1, :]  # 마지막 행이 null space의 기저
        print(f"선형 결합 계수: {coefficients}")
        
        # 검증
        result = T @ coefficients
        print(f"검증 (T * coefficients): {result}")
        print(f"결과의 노름: {np.linalg.norm(result)}")
        
        return coefficients
    else:
        print("선형 독립이므로 자명하지 않은 선형 결합이 존재하지 않습니다.")
        return None

if __name__ == "__main__":
    import numpy as np
    # 벡터 집합 T의 선형 독립성 판별
    print("=== 벡터 집합 T의 선형 독립성 판별 ===")
    is_independent = check_linear_independence_T()
    
    print(f"\n최종 결과: 벡터들은 {'선형 독립' if is_independent else '선형 종속'}입니다.")
    
    # 선형 종속인 경우 계수 찾기
    if not is_independent:
        print("\n=== 선형 결합 계수 찾기 ===")
        coefficients = find_linear_combination_coefficients()