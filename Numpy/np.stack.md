`np.stack()` 함수는 넘파이에서 여러 개의 배열(array)을 **새로운 축(axis)으로 쌓아** 하나의 배열로 만들어주는 함수이다.

---

## 1. 기본 구조 및 역할

**문법**  
```python
np.stack(arrays, axis=0)
```
- `arrays`: 쌓을 배열들의 리스트(또는 튜플), 모든 배열의 shape가 같아야 한다.
- `axis`: 새로 생성될 축의 위치(기본값 0, 즉 맨 앞에 축 추가)

---

## 2. 예제 코드 해설

```python
horizontal_filter = np.array([[-1, -1, -1],
                              [ 2,  2,  2],
                              [-1, -1, -1]])
vertical_filter = np.array([[-1, 2, -1],
                            [-1, 2, -1],
                            [-1, 2, -1]])

W = np.stack([horizontal_filter, vertical_filter])  # (2, 3, 3)
```

- `horizontal_filter`와 `vertical_filter`는 shape이 (3, 3)인 2차원 배열이다.
- `np.stack([horizontal_filter, vertical_filter])`는 이 두 개의 (3, 3) 배열을 **새로운 축(0번 축)**을 따라 쌓아서 (2, 3, 3)짜리 3차원 배열을 만든다.
    - **즉,** 두 개의 3x3 행렬이 한 배열의 0번 축(맨 앞)에 나란히 붙는다.

### 시각적으로
```
W[0] = horizontal_filter
W[1] = vertical_filter
```
즉,
```
W = [
      [[-1, -1, -1],
       [ 2,  2,  2],
       [-1, -1, -1]],

      [[-1,  2, -1],
       [-1,  2, -1],
       [-1,  2, -1]]
    ]
```
- shape: (2, 3, 3)

---

## 3. stack과 concatenate, vstack, hstack의 차이

- `np.concatenate`는 기존 축을 따라 배열을 붙인다(새 축 X).
- `np.stack`은 **새 축을 만든다**.
- `np.vstack`, `np.hstack`은 각각 수직, 수평 방향으로 배열을 붙인다(일부 경우에만 가능).

---

## 4. axis 파라미터 설명

- `axis=0`이면 맨 앞에 새 축이 생긴다(여러 장의 사진이 한 뭉치로).
- `axis=1`이면 두 번째 축에 쌓인다(행이 늘어남).
- `axis=2`이면 세 번째 축에 쌓인다(열이 늘어남).

예시:
```python
a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
print(np.stack([a,b], axis=0).shape) # (2, 2, 2)
print(np.stack([a,b], axis=1).shape) # (2, 2, 2)
print(np.stack([a,b], axis=2).shape) # (2, 2, 2)
```
※ 2차원 배열 2개를 stack하면 axis 위치에 따라 shape이 달라진다.

---

## 5. 결과

- W의 type은 `<class 'numpy.ndarray'>`
- W의 shape은 (2, 3, 3)

---

**요약:**  
`np.stack()`은 여러 개의 같은 shape 배열을 새로운 축으로 쌓아 다차원 배열을 만드는 함수이다.  
딥러닝에서 여러 필터(kernel) 또는 이미지 뭉치를 하나의 텐서로 다룰 때 자주 사용된다.
