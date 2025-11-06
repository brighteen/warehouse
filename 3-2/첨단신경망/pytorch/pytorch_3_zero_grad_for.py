import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

SEED = 42
torch.manual_seed(SEED)

x = torch.tensor([[1.0, 2.0, 3.0]])
model = nn.Linear(in_features=3, out_features=2, bias=True)
print('W_before:', model.weight, '\n')
print('b_before:', model.bias, '\n')

target = torch.tensor([[1.0, 0.0]])   

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for i in range(1, 30):  
    # 1) grad 초기화
    optimizer.zero_grad()
    print(f"[iter {i}] grad(before) weight:", model.weight.grad)
    print(f"[iter {i}] grad(before) bias  :", model.bias.grad, '\n')

    # 2) forward & loss
    output = model(x)                  # y = W x + b
    loss = criterion(output, target)
    print(f"[iter {i}] output =", output)
    print(f"[iter {i}] loss =", loss.item(), '\n')

    # 3) backward → grad 계산
    loss.backward()
    print(f"[iter {i}] grad(after) weight:", model.weight.grad)
    print(f"[iter {i}] grad(after) bias  :", model.bias.grad, '\n')

    # 4) step → 파라미터 업데이트
    optimizer.step()
    print(f"[iter {i}] W:", model.weight, '\n')
    print(f"[iter {i}] b:", model.bias, '\n')

    