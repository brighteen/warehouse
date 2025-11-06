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

optimizer.zero_grad()
print("model.weight.grad1", model.weight.grad, '\n')
print("model.bias.grad1", model.bias.grad, '\n')

output = model(x)                            # y = Wx + b
loss = criterion(output, target)
print("output1 =", output)
print("loss1 =", loss.item(), '\n')

loss.backward()
print("model.weight.grad1", model.weight.grad, '\n')
print("model.bias.grad1", model.bias.grad, '\n')

optimizer.step()                               # θ ← θ - lr * grad
print("W_after1", model.weight, '\n')
print("b_after1", model.bias, '\n')

optimizer.zero_grad()
print("model.weight.grad2", model.weight.grad, '\n')
print("model.bias.grad2", model.bias.grad, '\n')

output  = model(x)
loss = criterion(output, target)
print("output2 =", output)
print("loss2 =", loss.item())

loss.backward()
print("model.weight.grad2", model.weight.grad, '\n')
print("model.bias.grad2", model.bias.grad, '\n')

optimizer.step()                               # θ ← θ - lr * grad
print("W_after2", model.weight, '\n')
print("b_after2", model.bias, '\n')
