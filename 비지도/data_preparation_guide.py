# 소스 이미지 데이터 수집 가이드
"""
Image-to-Image 모델 훈련을 위한 소스 이미지 데이터 수집 스크립트

이 스크립트는 다양한 방법으로 소스 이미지를 수집하는 방법을 제공합니다.
"""

import os
import requests
import urllib.request
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import random

def create_source_data_directory():
    """소스 데이터 디렉토리 구조 생성"""
    base_dir = r'C:\Users\brigh\Documents\GitHub\warehouse\비지도'
    source_dir = os.path.join(base_dir, 'source_images')
    
    categories = [
        'landscapes',    # 풍경
        'portraits',     # 인물
        'nature',        # 자연
        'architecture',  # 건축
        'still_life',    # 정물
        'general'        # 일반
    ]
    
    for category in categories:
        category_path = os.path.join(source_dir, category)
        os.makedirs(category_path, exist_ok=True)
        print(f"디렉토리 생성: {category_path}")
    
    return source_dir

def download_sample_images():
    """
    샘플 이미지 다운로드 (무료 이미지 소스 활용)
    
    주의: 실제 사용 시 저작권을 확인하세요.
    """
    
    # Unsplash API를 통한 샘플 이미지 URL들 (예시)
    sample_urls = [
        # 풍경 이미지들
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
        "https://images.unsplash.com/photo-1447675325282-09b8a2f7e060?w=800",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
        
        # 자연 이미지들
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800",
        "https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?w=800",
        
        # 일반 이미지들
        "https://images.unsplash.com/photo-1604537529428-15bcbeecfe4d?w=800",
    ]
    
    source_dir = create_source_data_directory()
    
    print("샘플 이미지 다운로드 중...")
    
    for i, url in enumerate(sample_urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filename = f"sample_{i:03d}.jpg"
                filepath = os.path.join(source_dir, 'general', filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"다운로드 완료: {filename}")
                time.sleep(1)  # API 제한 방지
            
        except Exception as e:
            print(f"다운로드 실패 {url}: {e}")
    
    print("샘플 이미지 다운로드 완료")

def create_synthetic_source_images():
    """
    합성 소스 이미지 생성 (테스트용)
    
    실제 사진 대신 간단한 패턴 이미지를 생성하여
    모델 구조를 테스트할 수 있습니다.
    """
    
    source_dir = create_source_data_directory()
    
    print("합성 소스 이미지 생성 중...")
    
    # 다양한 패턴의 이미지 생성
    patterns = [
        'gradient',     # 그라디언트
        'noise',        # 노이즈
        'geometric',    # 기하학적 패턴
        'texture'       # 텍스처
    ]
    
    for pattern_idx, pattern in enumerate(patterns):
        for i in range(20):  # 각 패턴당 20개 이미지
            # 256x256 RGB 이미지 생성
            img_array = np.zeros((256, 256, 3), dtype=np.uint8)
            
            if pattern == 'gradient':
                # 그라디언트 패턴
                for x in range(256):
                    for y in range(256):
                        img_array[y, x] = [
                            int(255 * x / 256),
                            int(255 * y / 256),
                            int(255 * (x + y) / 512)
                        ]
            
            elif pattern == 'noise':
                # 랜덤 노이즈
                img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            
            elif pattern == 'geometric':
                # 기하학적 패턴
                center_x, center_y = 128, 128
                for x in range(256):
                    for y in range(256):
                        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if distance < 50 + 30 * np.sin(distance * 0.1):
                            img_array[y, x] = [255, 100, 100]
                        else:
                            img_array[y, x] = [100, 100, 255]
            
            elif pattern == 'texture':
                # 텍스처 패턴
                for x in range(256):
                    for y in range(256):
                        value = int(128 + 127 * np.sin(x * 0.1) * np.cos(y * 0.1))
                        img_array[y, x] = [value, value // 2, 255 - value]
            
            # PIL 이미지로 변환 및 저장
            img = Image.fromarray(img_array)
            filename = f"{pattern}_{i:03d}.jpg"
            filepath = os.path.join(source_dir, 'general', filename)
            img.save(filepath)
        
        print(f"{pattern} 패턴 이미지 생성 완료 (20개)")
    
    print("합성 소스 이미지 생성 완료")

def validate_dataset():
    """데이터셋 검증"""
    base_dir = r'C:\Users\brigh\Documents\GitHub\warehouse\비지도'
    source_dir = os.path.join(base_dir, 'source_images')
    target_dir = os.path.join(base_dir, 'artwork_data')
    
    print("데이터셋 검증 중...")
    print("="*50)
    
    # 소스 이미지 확인
    if os.path.exists(source_dir):
        source_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    source_files.append(os.path.join(root, file))
        
        print(f"✅ 소스 이미지: {len(source_files)}개")
        
        if len(source_files) == 0:
            print("❌ 소스 이미지가 없습니다!")
            return False
    else:
        print("❌ source_images 폴더가 없습니다!")
        return False
    
    # 타겟 이미지 확인
    if os.path.exists(target_dir):
        target_files = [
            f for f in os.listdir(target_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"✅ 타겟 이미지: {len(target_files)}개")
        
        if len(target_files) == 0:
            print("❌ 타겟 이미지가 없습니다!")
            return False
    else:
        print("❌ artwork_data 폴더가 없습니다!")
        return False
    
    # 이미지 품질 검사 (샘플)
    print("\n이미지 품질 검사:")
    sample_files = source_files[:5] + [os.path.join(target_dir, f) for f in target_files[:5]]
    
    for filepath in sample_files:
        try:
            img = Image.open(filepath)
            print(f"✅ {os.path.basename(filepath)}: {img.size}, {img.mode}")
        except Exception as e:
            print(f"❌ {os.path.basename(filepath)}: 오류 - {e}")
    
    print("\n데이터셋 검증 완료")
    return True

def get_data_preparation_recommendations():
    """데이터 준비 권장사항"""
    print("📋 소스 이미지 데이터 준비 권장사항")
    print("="*50)
    
    recommendations = [
        "🎯 이미지 수량:",
        "   - 최소 500장 이상 (타겟 이미지와 비슷한 수량)",
        "   - 더 많을수록 좋은 결과 (1000-5000장 권장)",
        "",
        "🎨 이미지 종류:",
        "   - 풍경 사진 (반 고흐가 많이 그린 주제)",
        "   - 인물 사진 (초상화 스타일 학습용)",
        "   - 정물 사진 (꽃, 과일 등)",
        "   - 건축물 사진",
        "   - 자연 사진 (나무, 들판 등)",
        "",
        "📏 이미지 품질:",
        "   - 해상도: 최소 256x256 이상",
        "   - 형식: JPG, PNG",
        "   - 품질: 선명하고 노이즈가 적은 이미지",
        "",
        "🚫 피해야 할 이미지:",
        "   - 너무 어둡거나 밝은 이미지",
        "   - 블러 처리된 이미지",
        "   - 저해상도 이미지",
        "   - 워터마크가 있는 이미지",
        "",
        "📁 추천 데이터 소스:",
        "   - Unsplash (무료, 고품질)",
        "   - Pexels (무료)",
        "   - Pixabay (무료)",
        "   - 개인 촬영 사진",
        "",
        "⚖️ 저작권 주의사항:",
        "   - 상업적 이용 가능한 이미지만 사용",
        "   - Creative Commons 라이선스 확인",
        "   - 개인 프로젝트용으로만 사용"
    ]
    
    for rec in recommendations:
        print(rec)

def main():
    """메인 함수"""
    print("🎨 Image-to-Image 모델용 소스 데이터 준비")
    print("="*60)
    
    while True:
        print("\n선택하세요:")
        print("1. 디렉토리 구조 생성")
        print("2. 합성 이미지 생성 (테스트용)")
        print("3. 데이터셋 검증")
        print("4. 데이터 준비 권장사항 보기")
        print("5. 종료")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == '1':
            create_source_data_directory()
            print("✅ 디렉토리 구조가 생성되었습니다.")
            print("이제 source_images 폴더에 일반 사진들을 추가하세요.")
        
        elif choice == '2':
            create_synthetic_source_images()
            print("✅ 테스트용 합성 이미지가 생성되었습니다.")
            print("실제 훈련 전에 모델 구조를 테스트할 수 있습니다.")
        
        elif choice == '3':
            if validate_dataset():
                print("✅ 데이터셋이 준비되었습니다!")
                print("이제 image_to_image_finetuning_detailed_guide.py에서")
                print("main_training() 함수를 실행할 수 있습니다.")
            else:
                print("❌ 데이터셋 준비가 필요합니다.")
        
        elif choice == '4':
            get_data_preparation_recommendations()
        
        elif choice == '5':
            print("프로그램을 종료합니다.")
            break
        
        else:
            print("잘못된 선택입니다. 1-5 중에서 선택하세요.")

if __name__ == "__main__":
    main()
