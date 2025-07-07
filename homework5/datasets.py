import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


class CustomImageDataset(Dataset):
    """Кастомный датасет для работы с папками классов"""

    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            root_dir (str): Путь к папке с классами
            transform: Аугментации для изображений
            target_size (tuple): Размер для ресайза изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        # Получаем список классов (папок)
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Собираем все пути к изображениям
        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')

        # Ресайзим изображение
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Применяем аугментации
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        """Возвращает список имен классов"""
        return self.classes

    def get_path(self, idx):
        return self.images[idx]


class AugmentationPipeline:
    """
    Класс для создания и управления пайплайнами аугментации.
    Позволяет добавлять, удалять и применять аугментации по имени.
    """

    def __init__(self):
        self._augmentations = {}  # Словарь для хранения аугментаций: {name: aug_instance}
        self._transform = None  # Скомбинированный transforms.Compose

    def add_augmentation(self, name: str, aug_instance):
        if not isinstance(name, str) or not name:
            raise ValueError("Имя аугментации должно быть непустой строкой.")
        if not callable(aug_instance) and not isinstance(aug_instance, transforms.Compose):
            print(
                f"Предупреждение: '{name}' не является callable объектом или transforms.Compose. Убедитесь, что это допустимая аугментация.")

        self._augmentations[name] = aug_instance
        self._update_transform()  # Обновляем скомбинированный transforms.Compose

    def remove(self, name: str):
        del self._augmentations[name]
        self._update_transform()  # Обновляем скомбинированный transforms.Compose

    def apply(self, image):
        if self._transform is None:
            self._update_transform()

        if self._transform is None or not self._augmentations:
            return image

        return self._transform(image)

    def get_augmentations(self) -> dict:
        return self._augmentations.copy()  # Возвращаем копию, чтобы избежать внешних изменений

    def _update_transform(self):
        # Сортируем по имени, чтобы обеспечить воспроизводимый порядок применения
        sorted_augs = sorted(self._augmentations.items())

        # Извлекаем только экземпляры аугментаций
        aug_instances = [aug for name, aug in sorted_augs]

        if aug_instances:
            self._transform = transforms.Compose(aug_instances)
        else:
            self._transform = None  # Нет аугментаций, _transform должен быть None

    def __repr__(self):
        """Строковое представление пайплайна."""
        if not self._augmentations:
            return "AugmentationPipeline(Empty)"
        augs_str = ",\n  ".join([f"'{name}': {aug.__class__.__name__}(...)"
                                 for name, aug in self.get_augmentations().items()])
        return f"AugmentationPipeline(\n  {augs_str}\n)"