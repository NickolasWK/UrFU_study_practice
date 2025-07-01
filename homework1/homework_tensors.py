import torch

# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
basic_tensor1 = torch.randn(3, 4)
# - Тензор размером 2x3x4, заполненный нулями
basic_tensor2 = torch.zeros(2, 3, 4)
# - Тензор размером 5x5, заполненный единицами
basic_tensor3 = torch.ones(5, 5)
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
arr = torch.arange(16.)
basic_tensor4 = torch.reshape(arr, (4, 4))

print("№1.1")
print(basic_tensor1)
print(basic_tensor2)
print(basic_tensor3)
print(basic_tensor4)


# Дано: тензор A размером 3x4 и тензор B размером 4x3
A = torch.randn(3, 4)
B = torch.randn(4, 3)
# Выполните:
# - Транспонирование тензора A
A_tr = torch.transpose(A, 0, 1)
# - Матричное умножение A и B
mult1 = A @ B
# - Поэлементное умножение A и транспонированного B
mult2 = A * torch.transpose(B, 0, 1)
# - Вычислите сумму всех элементов тензора A
summ = torch.sum(A)

print("№1.2")
print(A_tr)
print(mult1)
print(mult2)
print(summ)

# Создайте тензор размером 5x5x5
index_tensor = torch.reshape(torch.arange(125.), (5, 5, 5))
# Извлеките:
# - Первую строку
index_tensor1 = index_tensor[0]
# - Последний столбец
index_tensor2 = index_tensor[:, -1]
# - Подматрицу размером 2x2 из центра тензора
index_tensor3 = index_tensor[1:3, 1:3, 1:3]
# - Все элементы с четными индексами
index_tensor4 = index_tensor[index_tensor % 2 == 0]

print("№1.3")
print(index_tensor1)
print(index_tensor2)
print(index_tensor3)
print(index_tensor4)

# Создайте тензор размером 24 элемента
reshape_tensor = torch.arange(24.)
# Преобразуйте его в формы:
# - 2x12
reshape_tensor1 = torch.reshape(reshape_tensor, (2, 12))
# - 3x8
reshape_tensor2 = torch.reshape(reshape_tensor, (3, 8))
# - 4x6
reshape_tensor3 = torch.reshape(reshape_tensor, (4, 6))
# - 2x3x4
reshape_tensor4 = torch.reshape(reshape_tensor, (2, 3, 4))
# - 2x2x2x3
reshape_tensor5 = torch.reshape(reshape_tensor, (2, 2, 2, 3))

print("№1.4")
print(reshape_tensor1)
print(reshape_tensor2)
print(reshape_tensor3)
print(reshape_tensor4)
print(reshape_tensor5)