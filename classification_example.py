"""
분류(Classification) 알고리즘 사용 예제

이 파일은 classification_explanation.py의 간단한 사용 예제를 보여줍니다.
"""

# 비주얼 출력을 위해 백엔드 설정 (서버 환경에서는 주석 처리)
import matplotlib
matplotlib.use('Agg')  # GUI가 없는 환경에서 사용

from classification_explanation import ClassificationExplainer

def simple_classification_demo():
    """간단한 분류 데모"""
    print("=== 분류 알고리즘 간단 데모 ===\n")
    
    # 분류 설명자 생성
    classifier = ClassificationExplainer()
    
    # 데이터 로드 (아이리스 데이터셋)
    print("1. 데이터 로드:")
    classifier.create_sample_data('iris')
    
    # 여러 알고리즘 비교
    print("\n2. 알고리즘 성능 비교:")
    results = classifier.compare_algorithms()
    
    # 가장 좋은 알고리즘 찾기
    best_algo = max(results.items(), key=lambda x: x[1])
    print(f"\n최고 성능: {best_algo[0]} (정확도: {best_algo[1]:.4f})")
    
    # 교차 검증으로 안정성 확인
    print("\n3. 교차 검증:")
    classifier.cross_validation_example()

def algorithm_specific_demo():
    """특정 알고리즘 상세 데모"""
    print("\n=== 특정 알고리즘 상세 데모 ===\n")
    
    classifier = ClassificationExplainer()
    classifier.create_sample_data('wine')  # 와인 데이터셋 사용
    
    print("랜덤 포레스트 알고리즘 상세 분석:")
    rf_model, accuracy = classifier.demonstrate_random_forest(n_estimators=50)
    
    print(f"\n모델 정확도: {accuracy:.4f}")

if __name__ == "__main__":
    # 간단한 데모 실행
    simple_classification_demo()
    
    # 상세 데모 실행
    algorithm_specific_demo()
    
    print("\n=== 데모 완료 ===")
    print("더 자세한 내용은 classification_explanation.py의 main() 함수를 실행해보세요!")