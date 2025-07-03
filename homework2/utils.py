import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.datasets import make_classification

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def accuracy(y_pred_probs, y_true):
    # Если на вход подаются вероятности (после softmax), берем argmax
    # Если на вход подаются логиты, то для accuracy все равно argmax
    predicted_classes = y_pred_probs.argmax(dim=1)
    correct_predictions = (predicted_classes == y_true).float()
    return correct_predictions.mean().item()

def log_epoch(epoch, avg_loss, metrics=None):
    log_str = f"Эпоха {epoch:03d} | Потери: {avg_loss:.6f}"
    print(log_str)

def make_classification_data(n=200, n_features=2, n_classes=3, n_informative=2, n_redundant=0, n_repeated=0, random_state=42):
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        random_state=random_state,
        n_clusters_per_class=1
    )
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()