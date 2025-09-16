# 분류(Classification) 알고리즘 설명 코드

이 폴더는 머신러닝의 분류(Classification) 알고리즘에 대한 종합적인 설명과 실습 코드를 포함합니다.

## 파일 구성

### 1. `classification_explanation.py`
- **주요 파일**: 분류 알고리즘의 이론과 실제 구현을 포함하는 종합 설명 파일
- **포함 내용**:
  - 분류의 이론적 개념 설명
  - 6가지 주요 분류 알고리즘 구현 (KNN, Decision Tree, SVM, Random Forest, Logistic Regression, Naive Bayes)
  - 데이터 전처리 및 시각화
  - 모델 성능 평가 및 비교
  - 교차 검증
  - 특성 중요도 분석

### 2. `classification_example.py`
- **간단한 사용 예제**: 기본적인 사용 방법을 보여주는 예제 파일
- **빠른 시작**을 위한 데모 코드

## 사용 방법

### 필수 패키지 설치
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 기본 사용법

#### 1. 간단한 예제 실행
```bash
python classification_example.py
```

#### 2. 전체 설명 및 실습 실행
```bash
python classification_explanation.py
```

#### 3. 코드에서 직접 사용
```python
from classification_explanation import ClassificationExplainer

# 분류 설명자 생성
classifier = ClassificationExplainer()

# 데이터 로드
classifier.create_sample_data('iris')  # 'iris', 'wine', 'synthetic' 중 선택

# 특정 알고리즘 실행
knn_model, accuracy = classifier.demonstrate_knn(k=3)
dt_model, accuracy = classifier.demonstrate_decision_tree()
svm_model, accuracy = classifier.demonstrate_svm()

# 알고리즘 비교
results = classifier.compare_algorithms()
```

## 포함된 분류 알고리즘

1. **K-최근접 이웃(KNN)**
   - 거리 기반 분류
   - 간단하고 직관적
   - 비선형 경계 처리 가능

2. **결정 트리(Decision Tree)**
   - 규칙 기반 분류
   - 해석하기 쉬움
   - 특성 중요도 제공

3. **서포트 벡터 머신(SVM)**
   - 마진 최대화 기반
   - 고차원에서 효과적
   - 커널 트릭 활용

4. **랜덤 포레스트(Random Forest)**
   - 앙상블 기법
   - 과적합 방지
   - 안정적 성능

5. **로지스틱 회귀(Logistic Regression)**
   - 확률 기반 분류
   - 선형 모델
   - 해석 가능

6. **나이브 베이즈(Naive Bayes)**
   - 베이즈 정리 기반
   - 빠른 훈련 및 예측
   - 텍스트 분류에 효과적

## 주요 기능

### 📊 데이터 처리
- 다양한 데이터셋 지원 (Iris, Wine, 인공 데이터)
- 자동 데이터 분할 (훈련/테스트)
- 데이터 표준화

### 🔍 모델 평가
- 정확도(Accuracy) 계산
- 분류 리포트 (정밀도, 재현율, F1-score)
- 혼동 행렬(Confusion Matrix)
- 교차 검증(Cross Validation)

### 📈 시각화
- 데이터 분포 시각화
- 특성 중요도 그래프
- 혼동 행렬 히트맵

### ⚖️ 모델 비교
- 여러 알고리즘 성능 비교
- 최적 모델 자동 선택
- 성능 지표별 순위

## 학습 목표

이 코드를 통해 다음을 학습할 수 있습니다:

1. **이론 이해**: 분류의 기본 개념과 각 알고리즘의 원리
2. **실습 경험**: 실제 데이터를 사용한 모델 훈련 및 평가
3. **비교 분석**: 다양한 알고리즘의 장단점 비교
4. **성능 최적화**: 교차 검증과 하이퍼파라미터 튜닝
5. **실무 적용**: 실제 문제 해결에 필요한 전체 파이프라인

## 추가 정보

### 데이터셋 정보
- **Iris**: 붓꽃 분류 (150개 샘플, 4개 특성, 3개 클래스)
- **Wine**: 와인 분류 (178개 샘플, 13개 특성, 3개 클래스)
- **Synthetic**: 인공 생성 데이터 (1000개 샘플, 4개 특성, 3개 클래스)

### 성능 지표 설명
- **정확도(Accuracy)**: 전체 예측 중 맞춘 비율
- **정밀도(Precision)**: 양성으로 예측한 것 중 실제 양성 비율
- **재현율(Recall)**: 실제 양성 중 양성으로 예측한 비율
- **F1-score**: 정밀도와 재현율의 조화평균

---

**작성자**: Warehouse Repository  
**최종 수정**: 2024년  
**용도**: 교육 및 학습 목적