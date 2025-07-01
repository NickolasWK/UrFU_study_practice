# Создайте тензоры x, y, z с requires_grad=True
import torch
from math import pi
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2 * x * y * z
# Найдите градиенты по всем переменным
f.backward()
# Проверьте результат аналитически
print("№2.1")
print(x.grad) # Производная 2x+2yz = 2*2+2*3*4 = 28
print(y.grad) # Производная 2x+2yz = 2*3+2*2*4 = 22
print(z.grad) # Производная 2x+2yz = 2*4+2*3*2 = 20

x = torch.tensor([2.0, 3.0, 4.0, 7.0], requires_grad=True)
y_true = torch.tensor([2.0, 4.0, 5.0, 8.0], requires_grad=True)
n = 4

w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

y_pred = w * x + b
MSE = torch.sum((y_pred - y_true)**2)/n
MSE.backward()
print("№2.2")
print(w.grad)
print(b.grad)

# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
# Проверьте результат с помощью torch.autograd.grad

def get_grad_test(x_test):
    f_test = torch.sin(x_test**2 + 1)
    print(torch.autograd.grad(f_test, x_test))

x = torch.tensor(pi, requires_grad=True)
u = x**2 + 1
f = torch.sin(u)
f.backward()

print("№2.3")
print(x.grad)
get_grad_test(torch.tensor(pi, requires_grad=True))