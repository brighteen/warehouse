import torch

x = torch.tensor([3.0], requires_grad=True)
a = x ** 2
b = 3 * a
c = 4 * x
y = b * c + 2 # y = 3x^2 * 4x + 2

a.retain_grad()
b.retain_grad()
c.retain_grad()
y.retain_grad()
print(f"각 변수 확인")
print(f"x: {x}, a: {a}, b: {b}, c: {c}, y: {y} \n")

y.backward()

print("x.data:", x.data)
print("x.grad:", x.grad)
print("x.grad_fn:", x.grad_fn, "\n")

print("a.data:", a.data)
print("a.grad:", a.grad)
print("a.grad_fn:", a.grad_fn, "\n")

print("b.data:", b.data)
print("b.grad:", b.grad)
print("b.grad_fn:", b.grad_fn, "\n")

print("c.data:", c.data)
print("c.grad:", c.grad)
print("c.grad_fn:", c.grad_fn, "\n")

print("y.data:", y.data)
print("y.grad:", y.grad)