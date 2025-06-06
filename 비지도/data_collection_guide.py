# Image-to-Image 훈련용 데이터 수집 가이드

"""
현재 상황: 반 고흐 작품 250+개 보유
목표: Style Transfer 모델 훈련

필요한 추가 데이터:
1. 소스 이미지 (일반 사진들)
2. 충분한 양과 다양성
"""

import os
import requests
import zipfile
from PIL import Image
import matplotlib.pyplot as plt

# ================================================================
# 1. 무료 데이터셋 다운로드 옵션들
# ================================================================

def download_coco_subset():
    """COCO 데이터셋 일부 다운로드 (일반 사진용)"""
    print("🔽 COCO 데이터셋 다운로드 시작...")
    
    # COCO 2017 validation set (작은 크기)
    url = "http://images.cocodataset.org/zips/val2017.zip"
    
    # 다운로드 디렉토리 생성
    download_dir = r'C:\Users\brigh\Documents\GitHub\warehouse\비지도\source_images'
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"다운로드 위치: {download_dir}")
    print("⚠️  주의: 1GB 정도 용량이므로 시간이 걸릴 수 있습니다.")
    
    """
    # 실제 다운로드 코드 (필요시 활성화)
    response = requests.get(url, stream=True)
    zip_path = os.path.join(download_dir, "val2017.zip")
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # 압축 해제
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    
    print("✅ 다운로드 완료!")
    """

def alternative_datasets():
    """대안 데이터셋 옵션들"""
    print("🎯 추천 데이터셋 옵션들:")
    print()
    
    print("1️⃣ **Places365** (풍경 사진)")
    print("   - 365개 장소 카테고리")
    print("   - 고화질 풍경/건물 사진")
    print("   - 다운로드: http://places2.csail.mit.edu/download.html")
    print()
    
    print("2️⃣ **ImageNet** (일반 객체)")
    print("   - 다양한 객체 클래스")
    print("   - 고품질 이미지")
    print("   - 다운로드: https://www.image-net.org/")
    print()
    
    print("3️⃣ **Flickr 크리에이티브 커먼즈**")
    print("   - 실제 사용자 촬영 사진")
    print("   - 자연스러운 구도와 색감")
    print("   - API 또는 수동 수집")
    print()
    
    print("4️⃣ **Unsplash API** (고품질)")
    print("   - 전문가급 사진")
    print("   - 다양한 주제")
    print("   - API 키 필요")

def create_data_structure():
    """훈련용 데이터 폴더 구조 생성"""
    base_dir = r'C:\Users\brigh\Documents\GitHub\warehouse\비지도'
    
    # 폴더 구조 생성
    folders = [
        'training_data/source_images',      # 일반 사진 (소스)
        'training_data/target_images',      # 반 고흐 작품 (타겟)
        'training_data/validation',         # 검증용
        'trained_models',                   # 훈련된 모델 저장
        'results'                          # 결과 이미지 저장
    ]
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"📁 생성됨: {folder_path}")
    
    # 기존 반 고흐 작품들을 target_images로 복사하는 코드 제안
    artwork_source = os.path.join(base_dir, 'artwork_data')
    artwork_target = os.path.join(base_dir, 'training_data', 'target_images')
    
    print(f"\n📋 다음 단계:")
    print(f"1. {artwork_source}의 이미지들을 {artwork_target}로 복사")
    print(f"2. 일반 사진들을 training_data/source_images에 추가")
    print(f"3. 이미지 크기 및 품질 확인")

# ================================================================
# 2. 데이터 전처리 및 검증
# ================================================================

def validate_dataset(source_dir, target_dir):
    """데이터셋 유효성 검사"""
    print("🔍 데이터셋 검증 중...")
    
    # 이미지 개수 확인
    source_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    target_images = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"소스 이미지 수: {len(source_images)}")
    print(f"타겟 이미지 수: {len(target_images)}")
    
    # 권장 비율 확인
    if len(source_images) < len(target_images):
        print("⚠️  경고: 소스 이미지가 타겟보다 적습니다.")
        print("   권장: 소스 >= 타겟 (최소 1:1, 이상적으로는 3:1)")
    
    # 이미지 크기 분석
    print("\n📐 이미지 크기 분석:")
    sample_sizes = []
    
    for img_file in source_images[:10]:  # 샘플 10개만 확인
        try:
            img_path = os.path.join(source_dir, img_file)
            img = Image.open(img_path)
            sample_sizes.append(img.size)
        except Exception as e:
            print(f"오류 파일: {img_file} - {e}")
    
    if sample_sizes:
        avg_width = sum(size[0] for size in sample_sizes) / len(sample_sizes)
        avg_height = sum(size[1] for size in sample_sizes) / len(sample_sizes)
        print(f"평균 크기: {avg_width:.0f} x {avg_height:.0f}")
        
        if avg_width < 256 or avg_height < 256:
            print("⚠️  경고: 이미지 해상도가 낮습니다. 256x256 이상 권장")

def preprocess_images(input_dir, output_dir, target_size=(256, 256)):
    """이미지 전처리 (크기 조정, 포맷 통일)"""
    print(f"🔄 이미지 전처리 시작: {input_dir} → {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for i, img_file in enumerate(image_files):
        try:
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"processed_{i:04d}.jpg")
            
            # 이미지 로드 및 리사이즈
            img = Image.open(input_path).convert('RGB')
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # 저장
            img_resized.save(output_path, 'JPEG', quality=95)
            
            if i % 100 == 0:
                print(f"처리 완료: {i}/{len(image_files)}")
                
        except Exception as e:
            print(f"오류 파일: {img_file} - {e}")
    
    print(f"✅ 전처리 완료: {len(image_files)}개 이미지")

# ================================================================
# 3. 실제 실행 예제
# ================================================================

def setup_data_for_training():
    """훈련용 데이터 설정 완전 가이드"""
    print("="*60)
    print("🎨 Image-to-Image 훈련 데이터 설정 가이드")
    print("="*60)
    
    # 1. 폴더 구조 생성
    print("\n1️⃣ 폴더 구조 생성...")
    create_data_structure()
    
    # 2. 데이터셋 옵션 안내
    print("\n2️⃣ 데이터셋 수집 옵션:")
    alternative_datasets()
    
    # 3. 다음 단계 안내
    print("\n3️⃣ 다음 단계:")
    print("   a) 일반 사진 데이터 수집 (위 옵션 중 선택)")
    print("   b) 데이터 전처리 실행")
    print("   c) 훈련 시작")
    
    print("\n4️⃣ 예상 훈련 시간:")
    print("   - 데이터 1,000장: 2-4시간 (GPU 기준)")
    print("   - 데이터 10,000장: 1-2일")
    print("   - 에포크 100회 기준")

if __name__ == "__main__":
    setup_data_for_training()
