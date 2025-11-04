import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 데이터 로드
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=64, shuffle=False
)

# LeNet-5 모델
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1   = nn.Linear(in_features=16*5*5, out_features=120, bias=True)
        self.fc2   = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc3   = nn.Linear(in_features=84, out_features=10, bias=True)        
    
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

model = LeNet5()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 & 테스트
for epoch in range(1, 11):
    model.train()
    processed = 0  # 누적 샘플 수 추적
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        processed += len(data)  # 실제 처리한 샘플 수 누적
        
        # 매 iteration마다 출력
        if (batch_idx + 1) % 1 == 0:
            print(f'Epoch: {epoch} [{processed}/{len(train_loader.dataset)} '
                  f'({100. * processed / len(train_loader.dataset):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
            # print(model.conv1.weight[0],'\n')
            # print(model.conv1.weight[1],'\n')
            # print(model.conv1.weight[2],'\n')
            # print(model.conv1.weight[3],'\n')
            # print(model.conv1.weight[4],'\n')
            # print(model.conv1.weight[5],'\n')
    
    # 테스트
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            correct += (model(data).argmax(1) == target).sum().item()
    
    print(f'\n=== Epoch {epoch} Test Accuracy: {100*correct/len(test_loader.dataset):.2f}% ===\n')