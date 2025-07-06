import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn

def run_fc_epoch(model, data_loader, criterion, optimizer=None, device='cuda', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    all_true_labels = []
    all_preds = []

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if is_test:
            all_true_labels.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy().flatten())
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    if is_test:
        return avg_loss, accuracy, all_true_labels, all_preds
    return avg_loss, accuracy

# Тренировка модели
def train_fc_model(model, train_loader, test_loader, L2_reg, epochs=10, lr=0.001, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    if L2_reg:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    final_test_true_labels = []
    final_test_preds = []

    for epoch in range(epochs):
        train_loss, train_acc = run_fc_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc,  epoch_true_labels, epoch_preds = (
            run_fc_epoch(model, test_loader, criterion, None, device, is_test=True))

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)

        if epoch == epochs - 1:
            final_test_true_labels = epoch_true_labels
            final_test_preds = epoch_preds

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_test_true_labels': final_test_true_labels,
        'final_test_preds': final_test_preds
    }


def run_cnn_epoch(model, data_loader, criterion, optimizer=None, device='cuda', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0
    all_true_labels = []
    all_preds = []

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if is_test:
            all_true_labels.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    if is_test:
        return avg_loss, accuracy, all_true_labels, all_preds
    return avg_loss, accuracy

def train_cnn_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    final_test_true_labels = []
    final_test_preds = []

    for epoch in range(epochs):
        train_loss, train_acc = run_cnn_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc, epoch_true_labels, epoch_preds = run_cnn_epoch(model, test_loader, criterion, None, device, is_test=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)

        if epoch == epochs - 1:
            final_test_true_labels = epoch_true_labels
            final_test_preds = epoch_preds

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_test_true_labels': final_test_true_labels,
        'final_test_preds': final_test_preds
    }

def count_parameters(model):
    # Подсчитывает количество параметров модели
    return sum(p.numel() for p in model.parameters() if p.requires_grad)