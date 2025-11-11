import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

def cutout(image, n_holes=3, length=80):
    h, w = image.shape[:2]
    image_cutout = image.copy()
    cutout_regions = []
    
    for _ in range(n_holes):  # n_holes: Cutout 사각형의 개수
        y = random.randint(0, h - length)  # length: cutout 사각형의 한 변 길이
        x = random.randint(0, w - length)               
        image_cutout[y:y+length, x:x+length] = 0  # 검은색으로 마스킹        
        cutout_regions.append((x, y, length, length))
    
    return image_cutout, cutout_regions


def mixup(image1, image2, alpha=0.5):

    h1, w1 = image1.shape[:2]
    image2_resized = cv2.resize(image2, (w1, h1))
    
    lam = np.random.beta(alpha, alpha)  # Beta(α, α) 분포에서 lambda 샘플링    
    mixed_image = (lam * image1 + (1 - lam) * image2_resized).astype(np.uint8) # 가중치 혼합
    
    return mixed_image, lam


def cutmix(image1, image2, beta=1.0):

    h1, w1 = image1.shape[:2]                     # image1: 배경 이미지
    image2_resized = cv2.resize(image2, (w1, h1)) # image2: 붙여넣을 이미지
    
    # Lambda 값 샘플링 (Beta 분포)
    lam = np.random.beta(beta, beta) # beta: CutMix 비율 조정 파라미터
    
    # CutMix할 사각형 크기
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(w1 * cut_ratio)
    cut_h = int(h1 * cut_ratio)
    
    # 랜덤 위치 선택
    cx = np.random.randint(w1)
    cy = np.random.randint(h1)
    
    # 경계 처리
    x1 = np.clip(cx - cut_w // 2, 0, w1)
    x2 = np.clip(cx + cut_w // 2, 0, w1)
    y1 = np.clip(cy - cut_h // 2, 0, h1)
    y2 = np.clip(cy + cut_h // 2, 0, h1)
    
    # CutMix 적용
    cutmix_image = image1.copy()
    cutmix_image[y1:y2, x1:x2] = image2_resized[y1:y2, x1:x2]
    
    # 실제 혼합 비율 계산
    actual_lam = 1 - ((x2 - x1) * (y2 - y1) / (w1 * h1))
    
    return cutmix_image, actual_lam, (x1, y1, x2, y2)


def visualize_cutout(original, cutout_applied, cutout_regions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Cutout applied image
    axes[1].imshow(cv2.cvtColor(cutout_applied, cv2.COLOR_BGR2RGB))
    axes[1].set_title('After Cutout', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_mixup(image1, image2, mixed_image, alpha):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image 1', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Image 2', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Mixup (α={alpha:.2f})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_cutmix(image1, image2, cutmix_image, lam, region):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))   
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Background Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Patch Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # CutMix 영역 표시
    cutmix_vis = cutmix_image.copy()
    x1, y1, x2, y2 = region
    cv2.rectangle(cutmix_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    axes[2].imshow(cv2.cvtColor(cutmix_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'CutMix (λ={lam:.2f})', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


image1 = cv2.imread('cutecat.jpg')
print(f"imgae1 shape: {image1.shape}")
h, w = image1.shape[:2]
total_pixels = h * w

print(f"imgae1 1 size: {image1.shape}")
image2 = cv2.imread('cutedog.jpg')

# 1. Cutout 
print("\n1. Cutout")
cutout_image, regions = cutout(image1, n_holes=3, length=100)
print(f"Number of cutout regions: {len(regions)}")
for idx, (x, y, w, h) in enumerate(regions, 1):
    print(f"  Region {idx}: position=({x}, {y}), size={w}x{h}")

fig1 = visualize_cutout(image1, cutout_image, regions)
plt.savefig('cutout_result.png', dpi=150, bbox_inches='tight')

# 2. Mixup
print("-" * 60)
print("\n2. Mixup")
mixed_image, alpha = mixup(image1, image2, alpha=0.5)

print(f"Mixup ratio (α): {alpha:.2f}")
print(f"Mixup image1 ratio: {alpha*100:.1f}%")
print(f"Mixup image2 ratio: {(1-alpha)*100:.1f}%")
    
fig2 = visualize_mixup(image1, image2, mixed_image, alpha)
plt.savefig('mixup_result.png', dpi=150, bbox_inches='tight')

# 3. Cutmix
print("-" * 60)
print("\n3. Cutmix")

cutmix_image, lam, region = cutmix(image1, image2, beta=1.0)
x1, y1, x2, y2 = region
print(f"Cutmix lambda: {lam:.2f}")
print(f"Cut region: ({x1}, {y1}) to ({x2}, {y2})")
print(f"Cut size: {x2-x1}x{y2-y1} pixels")

cutmix_pixels = (x2 - x1) * (y2 - y1)
cutmix_ratio = (cutmix_pixels / total_pixels) * 100
print(f"Cutmix patch ratio: {cutmix_ratio:.2f}%")
print(f"Cutmix background ratio: {100-cutmix_ratio:.2f}%")

fig3 = visualize_cutmix(image1, image2, cutmix_image, lam, region)
plt.savefig('cutmix_result.png', dpi=150, bbox_inches='tight')        