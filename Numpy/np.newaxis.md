`np.newaxis`는 NumPy에서 배열의 차원을 늘릴 때 사용하는 특수한 인덱싱 객체이다.  
이 기능을 이용하면 기존 배열에 새로운 축(axis)을 추가하여, 배열의 형태(shape)를 쉽게 바꿀 수 있다.

---

### 1. 기본 개념

- `np.newaxis`는 파이썬에서 슬라이싱에 사용하는 콜론(`:`)처럼 인덱싱에 쓸 수 있다.
- 배열의 원하는 위치에 `np.newaxis`를 넣으면, 해당 위치에 새로운 차원이 추가된다.

---

### 2. 주요 특징

- **값 자체로는 None과 같다.** 즉, `np.newaxis` 대신 `None`을 써도 동작은 동일하다.
- **배열의 shape을 명확하게 조작**할 때 유용하다.
- **reshape 함수와 달리, 원하는 위치에 차원을 추가할 수 있다.**

---

### 3. 사용 예시

#### (1) 1차원 배열을 2차원 행 벡터로 변환  
```python
import numpy as np
a = np.array([1, 2, 3, 4])       # shape: (4,)
row_vec = a[np.newaxis, :]       # shape: (1, 4)
# 결과: [[1 2 3 4]]
```
  
#### (2) 1차원 배열을 2차원 열 벡터로 변환  
```python
col_vec = a[:, np.newaxis]       # shape: (4, 1)
# 결과:
# [[1]
#  [2]
#  [3]
#  [4]]
```

#### (3) 2차원 배열을 3차원으로 변환  
```python
b = np.array([[1, 2], [3, 4]])   # shape: (2, 2)
b3 = b[:, np.newaxis, :]         # shape: (2, 1, 2)
```

---

### 4. 시각적 설명

- 원본 배열: `[1, 2, 3, 4]` (shape: (4,))
- `a[np.newaxis, :]` → `[[1, 2, 3, 4]]` (shape: (1, 4))
- `a[:, np.newaxis]` →  
  ```
  [[1],
   [2],
   [3],
   [4]]
  ```
  (shape: (4, 1))

---

### 5. 언제 사용하는가?

- 딥러닝에서 배치 차원, 채널 차원을 추가할 때 자주 사용한다.
- 벡터/행렬 곱셈에서 shape 맞추기가 필요할 때 쓰인다.
- `broadcasting`을 활용할 때 명확한 shape 조작이 필요할 경우 사용한다.

---

### 6. 주의 사항

- 기존 데이터는 변하지 않고, 뷰(view)만 바뀐다.
- 너무 많은 newaxis 사용은 코드 가독성을 떨어뜨릴 수 있다. 필요할 때만 쓰는 것이 좋다.

---

### 결론

`np.newaxis`는 배열의 차원 조작을 간단하고 직관적으로 해주는 도구이다.  
특히 복잡한 데이터 처리나 딥러닝 전처리에서 자주 쓰이며, 원하는 위치에 쉽게 차원을 추가할 수 있다는 점이 큰 장점이다.