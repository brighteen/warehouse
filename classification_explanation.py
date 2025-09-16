"""
분류(Classification) 알고리즘 설명 및 구현 예제

이 파일은 머신러닝에서 가장 중요한 분야 중 하나인 분류(Classification)에 대해 
이론적 설명과 실제 구현 예제를 포함합니다.

작성자: Warehouse Repository
날짜: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (matplotlib에서 한글 표시)
plt.rcParams['font.family'] = 'DejaVu Sans'

class ClassificationExplainer:
    """
    분류 알고리즘을 설명하고 실습할 수 있는 클래스
    """
    
    def __init__(self):
        """초기화"""
        print("분류(Classification) 설명 클래스를 초기화했습니다.")
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def explain_classification_theory(self):
        """
        분류의 이론적 개념 설명
        """
        print("=" * 60)
        print("분류(Classification)란?")
        print("=" * 60)
        print("""
        분류는 지도학습(Supervised Learning)의 한 종류로, 
        주어진 데이터를 미리 정의된 클래스나 카테고리로 분류하는 작업입니다.
        
        주요 특징:
        1. 타겟 변수가 범주형(categorical) 데이터
        2. 입력 데이터를 기반으로 클래스 예측
        3. 정확도, 정밀도, 재현율 등으로 성능 평가
        
        분류의 종류:
        - 이진 분류(Binary Classification): 2개의 클래스 (예: 스팸/정상 메일)
        - 다중 분류(Multi-class Classification): 3개 이상의 클래스 (예: 꽃의 종류)
        - 다중 레이블 분류(Multi-label Classification): 하나의 샘플이 여러 클래스에 속함
        
        주요 알고리즘:
        1. K-최근접 이웃(K-Nearest Neighbors, KNN)
        2. 결정 트리(Decision Tree)
        3. 서포트 벡터 머신(Support Vector Machine, SVM)
        4. 로지스틱 회귀(Logistic Regression)
        5. 나이브 베이즈(Naive Bayes)
        6. 랜덤 포레스트(Random Forest)
        7. 신경망(Neural Networks)
        """)
        
    def create_sample_data(self, dataset_type='synthetic'):
        """
        샘플 데이터 생성 또는 로드
        
        Args:
            dataset_type (str): 'synthetic', 'iris', 'wine' 중 선택
        """
        print(f"\n{dataset_type} 데이터셋을 로드합니다...")
        
        if dataset_type == 'synthetic':
            # 인공 데이터 생성
            self.X, self.y = make_classification(
                n_samples=1000,
                n_features=4,
                n_informative=3,
                n_redundant=1,
                n_classes=3,
                random_state=42
            )
            feature_names = ['특성1', '특성2', '특성3', '특성4']
            target_names = ['클래스A', '클래스B', '클래스C']
            
        elif dataset_type == 'iris':
            # 아이리스 데이터셋
            data = load_iris()
            self.X, self.y = data.data, data.target
            feature_names = data.feature_names
            target_names = data.target_names
            
        elif dataset_type == 'wine':
            # 와인 데이터셋
            data = load_wine()
            self.X, self.y = data.data, data.target
            feature_names = data.feature_names
            target_names = data.target_names
            
        # 훈련/테스트 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"데이터 형태: {self.X.shape}")
        print(f"클래스 개수: {len(np.unique(self.y))}")
        print(f"특성 이름: {feature_names}")
        print(f"클래스 이름: {target_names}")
        
        return self.X, self.y
        
    def visualize_data(self):
        """
        데이터 시각화
        """
        if self.X is None or self.y is None:
            print("먼저 데이터를 로드해주세요.")
            return
            
        # 특성이 2개 이상인 경우, 처음 2개 특성으로 산점도 그리기
        if self.X.shape[1] >= 2:
            plt.figure(figsize=(12, 4))
            
            # 산점도
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='viridis', alpha=0.7)
            plt.xlabel('특성 1')
            plt.ylabel('특성 2')
            plt.title('데이터 분포 (처음 2개 특성)')
            plt.colorbar(scatter)
            
            # 클래스별 히스토그램
            plt.subplot(1, 2, 2)
            unique_classes = np.unique(self.y)
            for class_label in unique_classes:
                mask = self.y == class_label
                plt.hist(self.X[mask, 0], alpha=0.7, label=f'클래스 {class_label}')
            plt.xlabel('특성 1 값')
            plt.ylabel('빈도')
            plt.title('클래스별 특성 1 분포')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
    def demonstrate_knn(self, k=3):
        """
        K-최근접 이웃 알고리즘 시연
        
        Args:
            k (int): 이웃의 개수
        """
        print(f"\n{'='*50}")
        print(f"K-최근접 이웃(KNN) 알고리즘 (k={k})")
        print(f"{'='*50}")
        print("""
        KNN의 원리:
        1. 새로운 데이터 포인트와 가장 가까운 k개의 이웃을 찾음
        2. 이웃들의 클래스 중 가장 많은 클래스로 분류
        3. 거리 기반 알고리즘 (유클리드 거리 주로 사용)
        
        장점: 간단하고 직관적, 비선형 경계 처리 가능
        단점: 계산 비용이 높음, 노이즈에 민감
        """)
        
        # 데이터 표준화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # KNN 모델 훈련
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, self.y_train)
        
        # 예측
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"정확도: {accuracy:.4f}")
        print("\n분류 리포트:")
        print(classification_report(self.y_test, y_pred))
        
        return knn, accuracy
        
    def demonstrate_decision_tree(self, max_depth=3):
        """
        결정 트리 알고리즘 시연
        
        Args:
            max_depth (int): 트리의 최대 깊이
        """
        print(f"\n{'='*50}")
        print(f"결정 트리(Decision Tree) 알고리즘")
        print(f"{'='*50}")
        print("""
        결정 트리의 원리:
        1. 데이터를 가장 잘 분할하는 특성과 임계값을 찾음
        2. 재귀적으로 하위 노드에서 분할을 반복
        3. 리프 노드에서 클래스 결정
        
        장점: 해석하기 쉬움, 전처리가 거의 필요 없음
        단점: 과적합 경향, 불안정함
        """)
        
        # 결정 트리 모델 훈련
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt.fit(self.X_train, self.y_train)
        
        # 예측
        y_pred = dt.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"정확도: {accuracy:.4f}")
        print("\n분류 리포트:")
        print(classification_report(self.y_test, y_pred))
        
        # 특성 중요도
        if hasattr(dt, 'feature_importances_'):
            print("\n특성 중요도:")
            for i, importance in enumerate(dt.feature_importances_):
                print(f"특성 {i+1}: {importance:.4f}")
                
        return dt, accuracy
        
    def demonstrate_svm(self, kernel='rbf'):
        """
        서포트 벡터 머신 알고리즘 시연
        
        Args:
            kernel (str): 커널 함수 ('linear', 'rbf', 'poly')
        """
        print(f"\n{'='*50}")
        print(f"서포트 벡터 머신(SVM) 알고리즘 (kernel={kernel})")
        print(f"{'='*50}")
        print("""
        SVM의 원리:
        1. 클래스를 분리하는 최적의 결정 경계(hyperplane) 찾기
        2. 마진을 최대화하는 서포트 벡터 활용
        3. 커널 트릭으로 비선형 문제 해결
        
        장점: 고차원에서 효과적, 과적합 제어 가능
        단점: 큰 데이터셋에서 느림, 파라미터 튜닝 필요
        """)
        
        # 데이터 표준화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # SVM 모델 훈련
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train_scaled, self.y_train)
        
        # 예측
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"정확도: {accuracy:.4f}")
        print("\n분류 리포트:")
        print(classification_report(self.y_test, y_pred))
        
        return svm, accuracy
        
    def demonstrate_random_forest(self, n_estimators=100):
        """
        랜덤 포레스트 알고리즘 시연
        
        Args:
            n_estimators (int): 트리의 개수
        """
        print(f"\n{'='*50}")
        print(f"랜덤 포레스트(Random Forest) 알고리즘")
        print(f"{'='*50}")
        print("""
        랜덤 포레스트의 원리:
        1. 여러 개의 결정 트리를 랜덤하게 생성
        2. 각 트리에서 다른 특성 부분집합 사용
        3. 투표를 통해 최종 예측 결정
        
        장점: 과적합 감소, 안정적 성능, 특성 중요도 제공
        단점: 해석하기 어려움, 메모리 사용량 많음
        """)
        
        # 랜덤 포레스트 모델 훈련
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(self.X_train, self.y_train)
        
        # 예측
        y_pred = rf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"정확도: {accuracy:.4f}")
        print("\n분류 리포트:")
        print(classification_report(self.y_test, y_pred))
        
        # 특성 중요도
        print("\n특성 중요도:")
        for i, importance in enumerate(rf.feature_importances_):
            print(f"특성 {i+1}: {importance:.4f}")
            
        return rf, accuracy
        
    def compare_algorithms(self):
        """
        여러 알고리즘 성능 비교
        """
        print(f"\n{'='*50}")
        print("알고리즘 성능 비교")
        print(f"{'='*50}")
        
        # 데이터 표준화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # 알고리즘 정의
        algorithms = {
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            # 표준화가 필요한 알고리즘과 그렇지 않은 알고리즘 구분
            if name in ['KNN', 'SVM', 'Logistic Regression']:
                algorithm.fit(X_train_scaled, self.y_train)
                y_pred = algorithm.predict(X_test_scaled)
            else:
                algorithm.fit(self.X_train, self.y_train)
                y_pred = algorithm.predict(self.X_test)
                
            accuracy = accuracy_score(self.y_test, y_pred)
            results[name] = accuracy
            
        # 결과 출력
        print("알고리즘별 정확도:")
        print("-" * 30)
        for name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20}: {accuracy:.4f}")
            
        return results
        
    def evaluate_model_performance(self, model, X_test_scaled=None):
        """
        모델 성능 상세 평가
        
        Args:
            model: 훈련된 모델
            X_test_scaled: 표준화된 테스트 데이터 (필요한 경우)
        """
        if X_test_scaled is not None:
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(self.X_test)
            
        # 혼동 행렬
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('혼동 행렬 (Confusion Matrix)')
        plt.xlabel('예측 클래스')
        plt.ylabel('실제 클래스')
        plt.show()
        
        # 분류 리포트
        print("\n상세 분류 리포트:")
        print(classification_report(self.y_test, y_pred))
        
    def cross_validation_example(self):
        """
        교차 검증 예제
        """
        print(f"\n{'='*50}")
        print("교차 검증(Cross Validation) 예제")
        print(f"{'='*50}")
        print("""
        교차 검증의 목적:
        1. 모델의 일반화 성능을 더 정확히 평가
        2. 과적합 여부 확인
        3. 모델 선택 및 하이퍼파라미터 튜닝
        """)
        
        # 랜덤 포레스트로 교차 검증
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf, self.X, self.y, cv=5, scoring='accuracy')
        
        print(f"5-Fold 교차 검증 결과:")
        print(f"각 폴드 점수: {cv_scores}")
        print(f"평균 정확도: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    def feature_importance_analysis(self):
        """
        특성 중요도 분석
        """
        print(f"\n{'='*50}")
        print("특성 중요도 분석")
        print(f"{'='*50}")
        
        # 랜덤 포레스트로 특성 중요도 계산
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        
        # 특성 중요도 시각화
        feature_importance = rf.feature_importances_
        indices = np.argsort(feature_importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("특성 중요도")
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.xlabel("특성 인덱스")
        plt.ylabel("중요도")
        plt.xticks(range(len(feature_importance)), [f"특성 {i+1}" for i in indices])
        plt.show()
        
        print("특성별 중요도:")
        for i in indices:
            print(f"특성 {i+1}: {feature_importance[i]:.4f}")


def main():
    """
    메인 실행 함수 - 분류 알고리즘 전체 데모
    """
    print("분류(Classification) 알고리즘 종합 설명 및 실습")
    print("=" * 60)
    
    # 분류 설명 객체 생성
    classifier = ClassificationExplainer()
    
    # 1. 이론 설명
    classifier.explain_classification_theory()
    
    # 2. 데이터 로드 및 시각화
    print("\n" + "="*50)
    print("실습 데이터 준비")
    print("="*50)
    
    # 아이리스 데이터셋 사용
    classifier.create_sample_data('iris')
    classifier.visualize_data()
    
    # 3. 다양한 알고리즘 시연
    knn_model, knn_acc = classifier.demonstrate_knn(k=3)
    dt_model, dt_acc = classifier.demonstrate_decision_tree(max_depth=3)
    svm_model, svm_acc = classifier.demonstrate_svm(kernel='rbf')
    rf_model, rf_acc = classifier.demonstrate_random_forest(n_estimators=100)
    
    # 4. 알고리즘 비교
    comparison_results = classifier.compare_algorithms()
    
    # 5. 교차 검증
    classifier.cross_validation_example()
    
    # 6. 특성 중요도 분석
    classifier.feature_importance_analysis()
    
    # 7. 상세 평가 (최고 성능 모델)
    best_algorithm = max(comparison_results.items(), key=lambda x: x[1])
    print(f"\n최고 성능 알고리즘: {best_algorithm[0]} (정확도: {best_algorithm[1]:.4f})")
    
    print("\n" + "="*60)
    print("분류 알고리즘 실습이 완료되었습니다!")
    print("="*60)


if __name__ == "__main__":
    # 필요한 라이브러리 설치 확인
    try:
        import sklearn
        import matplotlib
        import seaborn
        print("모든 필요한 라이브러리가 설치되어 있습니다.")
        print("실습을 시작합니다...\n")
        main()
    except ImportError as e:
        print(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
        print("다음 명령어로 설치해주세요:")
        print("pip install scikit-learn matplotlib seaborn pandas numpy")