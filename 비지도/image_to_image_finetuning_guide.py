# Image-to-Image 모델 파인튜닝 가이드
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt

# ================================================================
# 1. 환경 설정 및 라이브러리 설치
# ================================================================

"""
필요한 라이브러리 설치:
pip install torch torchvision
pip install pillow
pip install matplotlib
pip install opencv-python

선택적 (고급 기능):
pip install pytorch-fid  # FID 스코어 계산
pip install lpips        # 지각적 손실 계산
"""

# ================================================================
# 2. 데이터셋 클래스 정의
# ================================================================

class ImageToImageDataset(Dataset):
    """Image-to-Image 변환을 위한 데이터셋 클래스"""
    
    def __init__(self, source_dir, target_dir, transform=None, paired=False):
        """
        Args:
            source_dir: 입력 이미지 디렉토리 (일반 사진)
            target_dir: 타겟 이미지 디렉토리 (반 고흐 작품)
            transform: 이미지 전처리
            paired: True면 같은 파일명끼리 매칭, False면 랜덤 매칭
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        self.paired = paired
        
        # 이미지 파일 목록 수집
        self.source_images = [f for f in os.listdir(source_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.target_images = [f for f in os.listdir(target_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # paired가 False면 더 많은 조합 생성 가능
        self.length = max(len(self.source_images), len(self.target_images))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 소스 이미지 선택
        source_idx = idx % len(self.source_images)
        source_path = os.path.join(self.source_dir, self.source_images[source_idx])
        source_img = Image.open(source_path).convert('RGB')
        
        # 타겟 이미지 선택
        if self.paired:
            # 같은 인덱스 사용 (paired 데이터)
            target_idx = source_idx % len(self.target_images)
        else:
            # 랜덤 선택 (unpaired 데이터)
            target_idx = torch.randint(0, len(self.target_images), (1,)).item()
        
        target_path = os.path.join(self.target_dir, self.target_images[target_idx])
        target_img = Image.open(target_path).convert('RGB')
        
        # 전처리 적용
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
        
        return source_img, target_img

# ================================================================
# 3. 모델 아키텍처 정의 (U-Net 기반)
# ================================================================

class UNetBlock(nn.Module):
    """U-Net의 기본 블록"""
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() if down else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) if use_dropout else None
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class Generator(nn.Module):
    """이미지 생성기 (U-Net 구조)"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder (다운샘플링)
        self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)  # 128x128
        self.down2 = UNetBlock(64, 128, down=True)         # 64x64
        self.down3 = UNetBlock(128, 256, down=True)        # 32x32
        self.down4 = UNetBlock(256, 512, down=True)        # 16x16
        self.down5 = UNetBlock(512, 512, down=True)        # 8x8
        self.down6 = UNetBlock(512, 512, down=True)        # 4x4
        
        # Bottleneck
        self.bottleneck = UNetBlock(512, 512, down=True)   # 2x2
        
        # Decoder (업샘플링)
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)     # 4x4
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)    # 8x8
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)    # 16x16
        self.up4 = UNetBlock(1024, 256, down=False)                      # 32x32
        self.up5 = UNetBlock(512, 128, down=False)                       # 64x64
        self.up6 = UNetBlock(256, 64, down=False)                        # 128x128
        
        # 최종 출력층
        self.final = nn.ConvTranspose2d(128, out_channels, 4, 2, 1)      # 256x256
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        # Bottleneck
        bottleneck = self.bottleneck(d6)
        
        # Decoder with skip connections
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))
        
        # 최종 출력
        output = self.final(torch.cat([u6, d1], 1))
        return self.tanh(output)

class Discriminator(nn.Module):
    """판별기 (PatchGAN)"""
    def __init__(self, in_channels=6):  # 입력 + 타겟 이미지
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, y):
        # 입력과 타겟을 연결
        combined = torch.cat([x, y], 1)
        return self.model(combined)

# ================================================================
# 4. 훈련 설정 및 함수
# ================================================================

def setup_training():
    """훈련 환경 설정"""
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 초기화
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # 손실 함수
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    
    # 최적화기
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] 범위로 정규화
    ])
    
    return {
        'device': device,
        'generator': generator,
        'discriminator': discriminator,
        'criterion_GAN': criterion_GAN,
        'criterion_L1': criterion_L1,
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'transform': transform
    }

def train_one_epoch(dataloader, models_dict, epoch):
    """한 에포크 훈련"""
    device = models_dict['device']
    generator = models_dict['generator']
    discriminator = models_dict['discriminator']
    criterion_GAN = models_dict['criterion_GAN']
    criterion_L1 = models_dict['criterion_L1']
    optimizer_G = models_dict['optimizer_G']
    optimizer_D = models_dict['optimizer_D']
    
    generator.train()
    discriminator.train()
    
    running_loss_G = 0.0
    running_loss_D = 0.0
    
    for batch_idx, (source, target) in enumerate(dataloader):
        source, target = source.to(device), target.to(device)
        batch_size = source.size(0)
        
        # 레이블 생성
        real_label = torch.ones(batch_size, 1, 30, 30).to(device)  # PatchGAN 출력 크기에 맞춤
        fake_label = torch.zeros(batch_size, 1, 30, 30).to(device)
        
        # =====================================
        # 판별기 훈련
        # =====================================
        optimizer_D.zero_grad()
        
        # 실제 이미지 쌍 판별
        real_pred = discriminator(source, target)
        loss_D_real = criterion_GAN(real_pred, real_label)
        
        # 가짜 이미지 생성 및 판별
        fake_target = generator(source)
        fake_pred = discriminator(source, fake_target.detach())
        loss_D_fake = criterion_GAN(fake_pred, fake_label)
        
        # 판별기 총 손실
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()
        
        # =====================================
        # 생성기 훈련
        # =====================================
        optimizer_G.zero_grad()
        
        # 판별기를 속이는 손실
        fake_pred = discriminator(source, fake_target)
        loss_G_GAN = criterion_GAN(fake_pred, real_label)
        
        # L1 손실 (픽셀 단위 유사성)
        loss_G_L1 = criterion_L1(fake_target, target)
        
        # 생성기 총 손실
        lambda_L1 = 100  # L1 손실 가중치
        loss_G = loss_G_GAN + lambda_L1 * loss_G_L1
        loss_G.backward()
        optimizer_G.step()
        
        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()
        
        # 진행상황 출력
        if batch_idx % 50 == 0:
            print(f'Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], '
                  f'Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}')
    
    avg_loss_G = running_loss_G / len(dataloader)
    avg_loss_D = running_loss_D / len(dataloader)
    
    return avg_loss_G, avg_loss_D

# ================================================================
# 5. 메인 훈련 함수
# ================================================================

def main_training():
    """메인 훈련 실행"""
    
    # 경로 설정 (현재 상황에 맞게 수정)
    current_artwork_dir = r'C:\Users\brigh\Documents\GitHub\warehouse\비지도\artwork_data'
    # source_dir = "path/to/general/photos"  # 일반 사진 디렉토리 (추가 필요)
    
    print("="*60)
    print("🎨 Image-to-Image 모델 파인튜닝 시작")
    print("="*60)
    
    # 현재 반 고흐 작품만 있는 상황
    print(f"현재 반 고흐 작품 수: {len(os.listdir(current_artwork_dir))}")
    print("\n⚠️  주의: 완전한 훈련을 위해서는 일반 사진 데이터가 추가로 필요합니다.")
    print("   Style Transfer의 경우: 소스 이미지(일반 사진) + 타겟 이미지(반 고흐 작품)")
    print("   현재는 반 고흐 작품만 있어서 style transfer가 아닌 다른 방법 고려 필요")
    
    # 훈련 설정
    config = setup_training()
    
    # 하이퍼파라미터
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0002
    
    print(f"\n📋 훈련 설정:")
    print(f"   - 에포크 수: {EPOCHS}")
    print(f"   - 배치 크기: {BATCH_SIZE}")
    print(f"   - 학습률: {LEARNING_RATE}")
    print(f"   - 디바이스: {config['device']}")
    
    # 실제 데이터셋이 준비되면 아래 코드 활성화
    """
    # 데이터셋 및 데이터로더 생성
    dataset = ImageToImageDataset(
        source_dir=source_dir,
        target_dir=current_artwork_dir,
        transform=config['transform'],
        paired=False  # unpaired 학습
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2
    )
    
    # 훈련 실행
    for epoch in range(EPOCHS):
        avg_loss_G, avg_loss_D = train_one_epoch(dataloader, config, epoch)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}] - Avg Loss_G: {avg_loss_G:.4f}, Avg Loss_D: {avg_loss_D:.4f}')
        
        # 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': config['generator'].state_dict(),
                'discriminator_state_dict': config['discriminator'].state_dict(),
                'optimizer_G_state_dict': config['optimizer_G'].state_dict(),
                'optimizer_D_state_dict': config['optimizer_D'].state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pth')
    """

if __name__ == "__main__":
    main_training()
