import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]

    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]

    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow * 2, 2))
    if nrow == 1:
        axes = [axes]

    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_single_augmentation(original_img, augmented_img, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)

    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')

    # Аугментированное изображение
    aug_np = aug_resized.numpy().transpose(1, 2, 0)
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def show_multiple_augmentations(original_img, augmented_imgs, titles):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))

    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)

    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def show_pipeline_aug(pipeline_name, pipeline, original_img, num_samples=5):
    print(pipeline)  # Используем __repr__ пайплайна

    fig, axes = plt.subplots(1, num_samples + 1, figsize=(3 * (num_samples + 1), 3))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i in range(num_samples):
        # Применяем пайплайн. Важно: последние аугментации должны вернуть тензор,
        # если вы хотите отображать его через plt.imshow напрямую
        # Поэтому мы добавим transforms.ToTensor() в пайплайн.

        # Создаем временный пайплайн с transforms.ToTensor() в конце для отображения
        display_pipeline_transforms = list(pipeline.get_augmentations().values())
        if not isinstance(display_pipeline_transforms[-1], transforms.ToTensor):
            # Проверяем, чтобы ToTensor был последним перед отображением
            display_pipeline_transforms.append(transforms.ToTensor())

        temp_compose = transforms.Compose(display_pipeline_transforms)

        augmented_img_tensor = temp_compose(original_img)

        # Для отображения, если это тензор PyTorch, нужно перевести его в numpy (H, W, C)
        img_to_show = augmented_img_tensor.permute(1, 2, 0).cpu().numpy()

        axes[i + 1].imshow(img_to_show)
        axes[i + 1].set_title(f"Aug {i + 1}")
        axes[i + 1].axis('off')

    plt.suptitle(f"{pipeline_name} Augmentations", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def show_time (sizes, processing_times):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sizes, processing_times, marker='o', linestyle='-', color='blue')
    plt.title('Время загрузки и аугментации vs. Размер изображения')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Время (секунды)')
    plt.grid(True)
    plt.xticks(sizes)

def show_memory(sizes, memory_usages):
    plt.subplot(1, 2, 2)
    plt.plot(sizes, memory_usages, marker='o', linestyle='-', color='red')
    plt.title('Потребление памяти vs. Размер изображения')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Использовано памяти (МБ)')
    plt.grid(True)
    plt.xticks(sizes)  # Убедимся, что метки соответствуют нашим размерам

    plt.tight_layout()
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