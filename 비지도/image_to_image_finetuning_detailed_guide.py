# Image-to-Image 모델 파인튜닝 상세 가이드
"""
이미지 투 이미지 모델 파인튜닝 완전 가이드

이 가이드는 반 고흐 스타일 변환을 위한 Image-to-Image 모델 파인튜닝 과정을 
단계별로 설명합니다.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# ================================
# 1. 데이터셋 준비 및 구성
# ================================

class ImageToImageDataset(Dataset):
    """
    Image-to-Image 변환을 위한 데이터셋 클래스
    
    구조:
    - source_images/: 원본 이미지들 (일반 사진, 풍경 등)
    - target_images/: 타겟 스타일 이미지들 (반 고흐 작품들)
    """
    
    def __init__(self, source_dir, target_dir, transform=None, image_size=256):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        self.image_size = image_size
        
        # 이미지 파일 목록 가져오기
        self.source_images = self._get_image_files(source_dir)
        self.target_images = self._get_image_files(target_dir)
        
        # 데이터셋 크기는 더 작은 쪽에 맞춤
        self.dataset_size = min(len(self.source_images), len(self.target_images))
        
        print(f"Source images: {len(self.source_images)}")
        print(f"Target images: {len(self.target_images)}")
        print(f"Dataset size: {self.dataset_size}")
    
    def _get_image_files(self, directory):
        """디렉토리에서 이미지 파일 목록 가져오기"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend([
                os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if f.lower().endswith(ext)
            ])
        
        return sorted(image_files)
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        # 인덱스가 범위를 벗어나면 순환
        source_idx = idx % len(self.source_images)
        target_idx = idx % len(self.target_images)
        
        # 이미지 로드
        source_path = self.source_images[source_idx]
        target_path = self.target_images[target_idx]
        
        source_image = Image.open(source_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')
        
        # 크기 조정
        source_image = source_image.resize((self.image_size, self.image_size))
        target_image = target_image.resize((self.image_size, self.image_size))
        
        # 변환 적용
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        
        return {
            'source': source_image,
            'target': target_image,
            'source_path': source_path,
            'target_path': target_path
        }

# ================================
# 2. 모델 아키텍처 정의
# ================================

class ResidualBlock(nn.Module):
    """잔차 블록 - 스타일 변환에 효과적"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    """
    Generator 네트워크 - U-Net 기반 구조
    
    구조:
    1. Encoder: 이미지를 낮은 해상도 특성맵으로 변환
    2. Residual Blocks: 스타일 변환 수행
    3. Decoder: 원래 해상도로 복원
    """
    
    def __init__(self, input_channels=3, output_channels=3, ngf=64, n_residual_blocks=9):
        super(Generator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # 첫 번째 레이어
            nn.Conv2d(input_channels, ngf, 7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            # 다운샘플링 레이어들
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(ngf * 4) for _ in range(n_residual_blocks)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # 업샘플링 레이어들
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            # 최종 출력 레이어
            nn.Conv2d(ngf, output_channels, 7, padding=3),
            nn.Tanh()  # [-1, 1] 범위로 출력
        )
    
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Residual blocks
        residual = self.residual_blocks(encoded)
        
        # Decoder
        output = self.decoder(residual)
        
        return output

class Discriminator(nn.Module):
    """
    Discriminator 네트워크 - PatchGAN 구조
    
    이미지 전체가 아닌 패치 단위로 진짜/가짜를 판별하여
    더 세밀한 텍스처와 디테일을 학습할 수 있습니다.
    """
    
    def __init__(self, input_channels=3, ndf=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 첫 번째 레이어 (배치 정규화 없음)
            nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 중간 레이어들
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 최종 출력 레이어
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# ================================
# 3. 손실 함수 정의
# ================================

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss - VGG 네트워크의 특성맵을 이용한 손실
    
    픽셀 단위 차이보다는 의미적 유사성을 측정하여
    더 자연스러운 스타일 변환을 가능하게 합니다.
    """
    
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 사전 훈련된 VGG19 모델 사용
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        
        # VGG의 특정 레이어들만 사용
        self.features = nn.Sequential(*list(vgg.features)[:36]).eval()
        
        # 파라미터 고정
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, generated, target):
        # VGG 특성맵 추출
        generated_features = self.features(generated)
        target_features = self.features(target)
        
        # MSE 손실 계산
        loss = nn.MSELoss()(generated_features, target_features)
        return loss

def gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    WGAN-GP를 위한 Gradient Penalty 계산
    
    더 안정적인 GAN 학습을 위해 사용됩니다.
    """
    batch_size = real_samples.size(0)
    
    # 랜덤 보간 계수
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # 실제와 가짜 샘플 사이의 보간
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Discriminator를 통과
    d_interpolates = discriminator(interpolates)
    
    # 그래디언트 계산
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Gradient penalty 계산
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ================================
# 4. 학습 설정 및 실행
# ================================

class StyleTransferTrainer:
    """Image-to-Image 스타일 변환 모델 훈련 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # 옵티마이저 초기화
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config['g_lr'],
            betas=(0.5, 0.999)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config['d_lr'],
            betas=(0.5, 0.999)
        )
        
        # 손실 함수 초기화
        self.adversarial_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        
        # 학습 기록
        self.losses = {
            'g_total': [],
            'g_adversarial': [],
            'g_l1': [],
            'g_perceptual': [],
            'd_real': [],
            'd_fake': []
        }
    
    def train_epoch(self, dataloader, epoch):
        """한 에폭 훈련"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {key: 0.0 for key in self.losses.keys()}
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            source_images = batch['source'].to(self.device)
            target_images = batch['target'].to(self.device)
            batch_size = source_images.size(0)
            
            # ============================
            # Discriminator 훈련
            # ============================
            
            self.d_optimizer.zero_grad()
            
            # 진짜 이미지에 대한 판별
            real_pred = self.discriminator(target_images)
            real_labels = torch.ones_like(real_pred)
            d_real_loss = self.adversarial_loss(real_pred, real_labels)
            
            # 가짜 이미지 생성 및 판별
            with torch.no_grad():
                fake_images = self.generator(source_images)
            fake_pred = self.discriminator(fake_images.detach())
            fake_labels = torch.zeros_like(fake_pred)
            d_fake_loss = self.adversarial_loss(fake_pred, fake_labels)
            
            # Discriminator 총 손실
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            self.d_optimizer.step()
            
            # ============================
            # Generator 훈련
            # ============================
            
            self.g_optimizer.zero_grad()
            
            # 가짜 이미지 생성
            fake_images = self.generator(source_images)
            
            # Adversarial Loss
            fake_pred = self.discriminator(fake_images)
            real_labels = torch.ones_like(fake_pred)
            g_adversarial_loss = self.adversarial_loss(fake_pred, real_labels)
            
            # L1 Loss (픽셀 단위 차이)
            g_l1_loss = self.l1_loss(fake_images, target_images)
            
            # Perceptual Loss (의미적 유사성)
            g_perceptual_loss = self.perceptual_loss(fake_images, target_images)
            
            # Generator 총 손실
            g_total_loss = (
                self.config['lambda_adv'] * g_adversarial_loss +
                self.config['lambda_l1'] * g_l1_loss +
                self.config['lambda_perceptual'] * g_perceptual_loss
            )
            
            g_total_loss.backward()
            self.g_optimizer.step()
            
            # 손실 기록
            epoch_losses['g_total'] += g_total_loss.item()
            epoch_losses['g_adversarial'] += g_adversarial_loss.item()
            epoch_losses['g_l1'] += g_l1_loss.item()
            epoch_losses['g_perceptual'] += g_perceptual_loss.item()
            epoch_losses['d_real'] += d_real_loss.item()
            epoch_losses['d_fake'] += d_fake_loss.item()
            
            # 진행 상황 업데이트
            progress_bar.set_postfix({
                'G_loss': f'{g_total_loss.item():.4f}',
                'D_loss': f'{d_loss.item():.4f}'
            })
        
        # 에폭 평균 손실 계산
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            self.losses[key].append(epoch_losses[key])
        
        return epoch_losses
    
    def save_checkpoint(self, epoch, save_dir):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'losses': self.losses,
            'config': self.config
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_{timestamp}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        print(f"체크포인트 저장됨: {checkpoint_path}")
        return checkpoint_path
    
    def generate_samples(self, dataloader, save_dir, num_samples=5):
        """샘플 이미지 생성 및 저장"""
        self.generator.eval()
        
        with torch.no_grad():
            batch = next(iter(dataloader))
            source_images = batch['source'][:num_samples].to(self.device)
            target_images = batch['target'][:num_samples].to(self.device)
            
            # 이미지 생성
            generated_images = self.generator(source_images)
            
            # 시각화
            fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
            
            for i in range(num_samples):
                # 정규화 해제 [-1, 1] -> [0, 1]
                source = (source_images[i].cpu() + 1) / 2
                target = (target_images[i].cpu() + 1) / 2
                generated = (generated_images[i].cpu() + 1) / 2
                
                # 이미지 표시
                axes[0, i].imshow(source.permute(1, 2, 0))
                axes[0, i].set_title('Source')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(generated.permute(1, 2, 0))
                axes[1, i].set_title('Generated')
                axes[1, i].axis('off')
                
                axes[2, i].imshow(target.permute(1, 2, 0))
                axes[2, i].set_title('Target')
                axes[2, i].axis('off')
            
            plt.tight_layout()
            
            # 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_path = os.path.join(save_dir, f'samples_{timestamp}.png')
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"샘플 이미지 저장됨: {sample_path}")

# ================================
# 5. 실제 사용 예제
# ================================

def setup_training_config():
    """훈련 설정 구성"""
    config = {
        # 데이터
        'source_dir': r'C:\Users\brigh\Documents\GitHub\warehouse\비지도\source_images',
        'target_dir': r'C:\Users\brigh\Documents\GitHub\warehouse\비지도\artwork_data',
        'image_size': 256,
        'batch_size': 4,
        
        # 훈련
        'num_epochs': 100,
        'g_lr': 0.0002,
        'd_lr': 0.0002,
        
        # 손실 함수 가중치
        'lambda_adv': 1.0,      # Adversarial loss 가중치
        'lambda_l1': 100.0,     # L1 loss 가중치
        'lambda_perceptual': 10.0,  # Perceptual loss 가중치
        
        # 저장
        'save_dir': r'C:\Users\brigh\Documents\GitHub\warehouse\비지도\training_results',
        'save_interval': 10,    # 몇 에폭마다 체크포인트 저장
        'sample_interval': 5    # 몇 에폭마다 샘플 생성
    }
    
    return config

def create_directories(config):
    """필요한 디렉토리 생성"""
    directories = [
        config['source_dir'],
        config['target_dir'],
        config['save_dir'],
        os.path.join(config['save_dir'], 'checkpoints'),
        os.path.join(config['save_dir'], 'samples')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"디렉토리 확인/생성: {directory}")

def setup_data_transforms(image_size):
    """데이터 전처리 변환 설정"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]로 정규화
    ])
    
    return transform

def main_training():
    """메인 훈련 함수"""
    print("🎨 Image-to-Image 스타일 변환 모델 훈련 시작")
    print("="*70)
    
    # 설정 로드
    config = setup_training_config()
    
    # 디렉토리 생성
    create_directories(config)
    
    # 디바이스 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 데이터 전처리 설정
    transform = setup_data_transforms(config['image_size'])
    
    # 데이터셋 및 데이터로더 생성
    try:
        dataset = ImageToImageDataset(
            source_dir=config['source_dir'],
            target_dir=config['target_dir'],
            transform=transform,
            image_size=config['image_size']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2 if device.type == 'cuda' else 0
        )
        
        print(f"데이터셋 크기: {len(dataset)}")
        print(f"배치 크기: {config['batch_size']}")
        print(f"배치 수: {len(dataloader)}")
        
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        print("\n해결 방법:")
        print("1. source_images 폴더에 일반 사진들을 추가하세요")
        print("2. artwork_data 폴더에 반 고흐 작품들이 있는지 확인하세요")
        return
    
    # 훈련 객체 생성
    trainer = StyleTransferTrainer(config)
    
    # 훈련 실행
    print(f"\n🚀 훈련 시작 - {config['num_epochs']} 에폭")
    print("="*70)
    
    for epoch in range(config['num_epochs']):
        # 에폭 훈련
        epoch_losses = trainer.train_epoch(dataloader, epoch)
        
        # 손실 출력
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} 완료")
        print(f"Generator Loss: {epoch_losses['g_total']:.4f}")
        print(f"Discriminator Loss: {(epoch_losses['d_real'] + epoch_losses['d_fake'])/2:.4f}")
        
        # 체크포인트 저장
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_dir = os.path.join(config['save_dir'], 'checkpoints')
            trainer.save_checkpoint(epoch, checkpoint_dir)
        
        # 샘플 생성
        if (epoch + 1) % config['sample_interval'] == 0:
            sample_dir = os.path.join(config['save_dir'], 'samples')
            trainer.generate_samples(dataloader, sample_dir)
    
    print("\n🎉 훈련 완료!")
    
    # 최종 모델 저장
    final_checkpoint_dir = os.path.join(config['save_dir'], 'checkpoints')
    final_checkpoint = trainer.save_checkpoint(config['num_epochs']-1, final_checkpoint_dir)
    
    print(f"최종 모델 저장됨: {final_checkpoint}")

# ================================
# 6. 모델 사용 및 추론
# ================================

def load_trained_model(checkpoint_path, device):
    """훈련된 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 생성
    generator = Generator().to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    print(f"모델 로드 완료: {checkpoint_path}")
    print(f"훈련 에폭: {checkpoint['epoch'] + 1}")
    
    return generator

def style_transfer_inference(generator, input_image_path, output_path, device):
    """스타일 변환 추론"""
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 이미지 로드 및 전처리
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # 스타일 변환
    with torch.no_grad():
        output_tensor = generator(input_tensor)
        
        # 정규화 해제
        output_tensor = (output_tensor + 1) / 2
        output_tensor = torch.clamp(output_tensor, 0, 1)
        
        # PIL 이미지로 변환
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
        
        # 저장
        output_image.save(output_path)
        
        print(f"스타일 변환 완료: {output_path}")
        
        return output_image

if __name__ == "__main__":
    print("Image-to-Image 모델 파인튜닝 가이드")
    print("="*50)
    print()
    print("이 파일은 반 고흐 스타일 변환을 위한")
    print("Image-to-Image 모델 파인튜닝 가이드입니다.")
    print()
    print("실행하려면:")
    print("1. source_images 폴더에 일반 사진들 추가")
    print("2. artwork_data 폴더에 반 고흐 작품들 확인")
    print("3. main_training() 함수 실행")
    print()
    print("주의: GPU가 권장되며, 충분한 저장공간이 필요합니다.")
