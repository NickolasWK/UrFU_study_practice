import torch
import time

# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
# Заполните их случайными числами
test_matrix1 = torch.randn(64, 1024, 1024)
test_matrix2 = torch.randn(128, 512, 512)
test_matrix3 = torch.randn(256, 256, 256)

# Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU

def GPU_time(functions):
    GPU_timer = []
    starter, ender = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
    data = []
    for _ in range(10): #Разогрев CUDA
        torch.matmul(test_matrix1, test_matrix1)

    for i in range(5):
        starter.record()
        data.append(functions[i](test_matrix1, test_matrix2, test_matrix3))
        ender.record()
        torch.cuda.synchronize()
        GPU_timer.append(starter.elapsed_time(ender))
    return GPU_timer

def CPU_time(functions):
    CPU_timer = []
    for i in range(5):
        start_time = time.time()
        functions[i](test_matrix1, test_matrix2, test_matrix3)
        end_time = time.time()
        CPU_timer.append(end_time - start_time)
    return CPU_timer

# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

def matrix_multiply (test_matrix1, test_matrix2, test_matrix3):
    return(
    torch.matmul(test_matrix1, test_matrix1),
    torch.matmul(test_matrix2, test_matrix2),
    torch.matmul(test_matrix3, test_matrix3)
    )

def elements_sum (test_matrix1, test_matrix2, test_matrix3):
    return (
    test_matrix1 + test_matrix1,
    test_matrix2 + test_matrix2,
    test_matrix3 + test_matrix3)

def elements_multiply (test_matrix1, test_matrix2, test_matrix3):
    return (
    test_matrix1 * test_matrix1,
    test_matrix2 * test_matrix2,
    test_matrix3 * test_matrix3)

def transpose (test_matrix1, test_matrix2, test_matrix3):
    return (
    torch.transpose(test_matrix1, 0, 1),
    torch.transpose(test_matrix2, 0, 1),
    torch.transpose(test_matrix3, 0, 1))

def summary (test_matrix1, test_matrix2, test_matrix3):
    return(
    torch.sum(test_matrix1),
    torch.sum(test_matrix2),
    torch.sum(test_matrix3))

# Для каждой операции:
functions = []
functions.append(matrix_multiply)
functions.append(elements_sum)
functions.append(elements_multiply)
functions.append(transpose)
functions.append(summary)

# 1. Измерьте время на GPU
test_matrix1.cuda()
test_matrix2.cuda()
test_matrix3.cuda()
GPU_logs = GPU_time(functions)

# 2. Измерьте время на CPU
test_matrix1.cpu()
test_matrix2.cpu()
test_matrix3.cpu()
CPU_logs = CPU_time(functions)

# 3. Вычислите ускорение (speedup)
speedup = [max(GPU_logs[i], CPU_logs[i]) / min(GPU_logs[i], CPU_logs[i])
           for i in range(len(GPU_logs))]

# 4. Выведите результаты в табличном виде
print("Операция                 | GPU(ms)    | CPU(ms) | Speedup")
print("Матричное умножение      | {:.5f}    | {:.5f}    | {:.1f} ".format(GPU_logs[0], CPU_logs[0], speedup[0]))
print("Поэлементное сложение    | {:.5f}    | {:.5f}    | {:.1f} ".format(GPU_logs[1], CPU_logs[1], speedup[1]))
print("Поэлементное умножение   | {:.5f}    | {:.5f}    | {:.1f} ".format(GPU_logs[2], CPU_logs[2], speedup[2]))
print("Транспонирование         | {:.5f}    | {:.5f}    | {:.1f} ".format(GPU_logs[3], CPU_logs[3], speedup[3]))
print("Сумма элементов          | {:.5f}    | {:.5f}    | {:.1f} ".format(GPU_logs[4], CPU_logs[4], speedup[4]))

# Значения аномально низкие, решить проблему не успел