import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn as nn

def plot_learning_curves(results):
    plt.figure(figsize=(15, 5))

    # График потерь на тренировочном наборе
    plt.subplot(1, 2, 1)
    for res in results:
        plt.plot(res["train_losses"], label=f'{res["model_name"]} (Loss)')
    plt.title('Кривые потерь на тренировочном наборе')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери (CrossEntropyLoss)')
    plt.legend()
    plt.grid(True)

    # График точности на валидационном наборе
    plt.subplot(1, 2, 2)
    for res in results:
        plt.plot(res["val_accuracies"], label=f'{res["model_name"]} (Accuracy)')
    plt.title('Кривые точности на валидационном наборе')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(all_true_labels, all_preds, num_classes, model_name):
    cm = confusion_matrix(all_true_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title(f'Confusion Matrix для {model_name}')
    plt.show()

def plot_training_history(history):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_weights_distribution(model, model_name):
    # Визуализирует распределение весов для каждого линейного слоя в модели.

    # Если модель обернута в DataParallel, нужно получить доступ к базовой модели
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model

    # Собираем все линейные слои
    linear_layers = []
    for name, module in actual_model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    if not linear_layers:
        print(f"В модели '{model_name}' не найдено линейных слоев для визуализации весов.")
        return

    # Определяем количество графиков
    num_layers = len(linear_layers)
    # Вычисляем оптимальное количество строк и столбцов для сетки графиков
    ncols = 3 # Или другое удобное число
    nrows = (num_layers + ncols - 1) // ncols

    plt.figure(figsize=(ncols * 5, nrows * 4)) # Устанавливаем размер фигуры

    for i, (name, layer) in enumerate(linear_layers):
        plt.subplot(nrows, ncols, i + 1)
        # Получаем веса и переводим их в numpy массив
        # .detach() отсоединяет тензор от графа вычислений, .cpu() перемещает на CPU
        weights = layer.weight.detach().cpu().numpy().flatten()
        sns.histplot(weights, kde=True, bins=50) # kde=True добавляет оценку плотности ядра
        plt.title(f'Распределение весов: {name}')
        plt.xlabel('Значение веса')
        plt.ylabel('Частота')
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout() # Автоматическая корректировка расположения элементов
    plt.suptitle(f'Распределение весов для {model_name}', y=1.02, fontsize=16) # Общий заголовок
    plt.show()