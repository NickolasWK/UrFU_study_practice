import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F

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

def compare_train_models(history, names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(len(history)):
        ax1.plot(history[i]['train_accs'], label=names[i])
    ax1.set_title('Train Accuracy Comparison')
    ax1.legend()

    for i in range(len(history)):
        ax2.plot(history[i]['train_losses'], label=names[i])
    ax2.set_title('Train Loss Comparison')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def compare_test_models(history, names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(len(history)):
        ax1.plot(history[i]['test_accs'], label=names[i])
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()

    for i in range(len(history)):
        ax2.plot(history[i]['test_accs'], label=names[i])
    ax2.set_title('Test Loss Comparison')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, input_image, layer_names=None, model_name="Model"):
    model.eval()

    feature_maps = {}

    def get_features(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()
        return hook

    hooks = []
    actual_model = model.module if isinstance(model, nn.DataParallel) else model

    if layer_names is None:
        for name, module in actual_model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(get_features(name)))
    else:
        for name, module in actual_model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(get_features(name)))

    with torch.no_grad():
        _ = model(input_image)

    for hook in hooks:
        hook.remove()

    for layer_name, features_tensor in feature_maps.items():
        features = features_tensor[0].cpu().numpy()
        num_channels = features.shape[0]
        ncols = min(8, num_channels)
        nrows = (num_channels + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axes = axes.flatten()

        for i in range(num_channels):
            if i < len(axes):
                ax = axes[i]
                ax.imshow(features[i], cmap='viridis')
                ax.axis('off')
        for i in range(num_channels, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'Карты признаков слоя: {layer_name} ({model_name})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def visualize_first_layer_activations(model, input_image, model_name="Model"):
    model.eval()
    activations = {}
    def conv1_hook(module, input, output):
        bn1_layer = None
        if hasattr(model, 'bn1'):
            bn1_layer = model.bn1
        elif hasattr(model.module, 'bn1'):
            bn1_layer = model.module.bn1

        if bn1_layer:
            activated_output = F.relu(bn1_layer(output))
            activations['conv1_activations'] = activated_output.detach()
        else:
            activations['conv1_activations'] = output.detach()

    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    hook_handle = actual_model.conv1.register_forward_hook(conv1_hook)

    with torch.no_grad():
        _ = model(input_image)

    hook_handle.remove()

    if 'conv1_activations' in activations:
        features = activations['conv1_activations'][0].cpu().numpy()  # Берем первый элемент батча
        num_channels = features.shape[0]
        ncols = min(8, num_channels)  # Максимум 8 столбцов
        nrows = (num_channels + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
        axes = axes.flatten()

        for i in range(num_channels):
            if i < len(axes):
                ax = axes[i]
                ax.imshow(features[i], cmap='viridis')
                ax.axis('off')

        for i in range(num_channels, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'Активации первого слоя ({model_name})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print(f"Не удалось получить активации первого слоя для модели {model_name}.")