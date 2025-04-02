import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from gnn_model import GATNet
from graph_smote import GraphSMOTE
from torch.optim import Adam
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter


# 设置随机种子确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设置随机种子
set_seed(345621)

# 加载预处理好的数据集
dataset = torch.load(
    "C:\\Users\\86130\\Desktop\\Code\\Pycharm\\flaky_test_prediction\\data\\processed\\processed_data2.pt")
print("Number of graphs in dataset:", len(dataset))

# 打印数据集类别分布
labels = [data.y.item() for data in dataset]
label_counts = Counter(labels)
print("Class distribution before balancing:", label_counts)

# 在划分数据集前打乱顺序
indices = list(range(len(dataset)))
random.shuffle(indices)
dataset = [dataset[i] for i in indices]

# 划分训练集、验证集和测试集（70%/20%/10%）
num_graphs = len(dataset)
train_num = int(num_graphs * 0.7)
val_num = int(num_graphs * 0.2)
test_num = num_graphs - train_num - val_num

train_dataset = dataset[:train_num]
val_dataset = dataset[train_num:train_num + val_num]
test_dataset = dataset[train_num + val_num:]

# 打印训练集的类别分布
train_labels = [data.y.item() for data in train_dataset]
train_label_counts = Counter(train_labels)
print("Training set class distribution before balancing:", train_label_counts)

# 应用GraphSMOTE仅对训练集进行平衡处理
graphsmote = GraphSMOTE(k=5, ratio=0.75, sigma=0.1, n_connect=3)
balanced_train_dataset = graphsmote(train_dataset)

# 打印平衡后的训练集类别分布
balanced_train_labels = [data.y.item() for data in balanced_train_dataset]
balanced_train_label_counts = Counter(balanced_train_labels)
print("Training set class distribution after balancing:", balanced_train_label_counts)

# 创建DataLoader
train_loader = DataLoader(balanced_train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 超参数设置
in_channels = dataset[0].x.size(1)  # CodeBERT嵌入维度
hidden_channels = 128
num_classes = 5
learning_rate = 0.0001
epochs = 200

# 检查是否有CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型
model = GATNet(in_channels, hidden_channels, num_classes).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)

# 根据类别不平衡计算权重用于损失函数
class_weights = None
if True:  # 设为True启用加权损失函数
    # 计算每个类别的权重（反比于样本数量）
    class_counts = np.bincount(balanced_train_labels, minlength=num_classes)
    class_weights = torch.tensor(
        1.0 / (class_counts + 1e-10),  # 添加小值避免除零
        dtype=torch.float32
    ).to(device)
    # 归一化权重
    class_weights = class_weights / class_weights.sum() * num_classes
    print("Class weights:", class_weights)

# 使用加权交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(balanced_train_dataset)


def evaluate(loader, dataset_len):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    # 收集所有预测和真实标签用于后续计算指标
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
            preds = out.argmax(dim=1)
            correct += (preds == data.y.view(-1)).sum().item()
            total += data.num_graphs
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.view(-1).cpu().numpy())
    avg_loss = total_loss / total
    accuracy = correct / total

    # 计算F1 Weighted指标
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')

    # 计算每个类别的F1分数
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_per_class': f1_per_class,
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_confusion_matrix(labels, preds, class_names=None):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(labels, preds)

    # 归一化混淆矩阵（按行）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))

    # 绘制原始计数混淆矩阵
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Counts)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if class_names:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)

    # 绘制归一化混淆矩阵
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if class_names:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=90)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)

    plt.tight_layout()
    plt.savefig("confusion_matrix_graphsmote5.png")
    plt.close()


def plot_training_history(history):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 8))

    # 绘制Loss曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制Accuracy曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制F1 Weighted曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Train F1 (Weighted)')
    plt.plot(history['val_f1'], label='Validation F1 (Weighted)')
    plt.title('F1 Score (Weighted)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    # 绘制Precision/Recall曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['train_precision'], label='Train Precision')
    plt.plot(history['train_recall'], label='Train Recall')
    plt.plot(history['val_precision'], label='Val Precision')
    plt.plot(history['val_recall'], label='Val Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history_graphsmote5.png")
    plt.close()


def plot_class_f1_scores(test_f1_per_class):
    """绘制每个类别的F1分数"""
    plt.figure(figsize=(12, 6))
    classes = np.arange(len(test_f1_per_class))
    plt.bar(classes, test_f1_per_class)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Class')
    plt.xticks(classes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加数值标签
    for i, v in enumerate(test_f1_per_class):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')

    plt.tight_layout()
    plt.savefig("class_f1_scores_graphsmote5.png")
    plt.close()


if __name__ == "__main__":
    best_val_acc = 0
    best_val_f1 = 0
    patience = 200  # 早停耐心值
    counter = 0  # 早停计数器

    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'train_precision': [],
        'train_recall': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }

    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train()

        # 评估
        train_metrics = evaluate(train_loader, len(balanced_train_dataset))
        val_metrics = evaluate(val_loader, len(val_dataset))

        # 记录历史（使用F1 Weighted）
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_weighted'])
        history['train_precision'].append(train_metrics['precision_macro'])
        history['train_recall'].append(train_metrics['recall_macro'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_weighted'])
        history['val_precision'].append(val_metrics['precision_macro'])
        history['val_recall'].append(val_metrics['recall_macro'])

        # 打印当前epoch的结果（使用F1 Weighted）
        print(f"Epoch: {epoch:03d}, "
              f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Train F1 (Weighted): {train_metrics['f1_weighted']:.4f}, "
              f"Train Prec: {train_metrics['precision_macro']:.4f}, Train Rec: {train_metrics['recall_macro']:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}, "
              f"Val Prec: {val_metrics['precision_macro']:.4f}, Val Rec: {val_metrics['recall_macro']:.4f}")

        # 保存最佳模型（基于验证F1 Weighted指标）
        if val_metrics['f1_weighted'] > best_val_f1:
            best_val_f1 = val_metrics['f1_weighted']
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), "best_model_graphsmote5.pth")
            counter = 0  # 重置早停计数器
        else:
            counter += 1

        # 早停检查
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}. No improvement in validation F1 for {patience} epochs.")
            break

    # 绘制训练历史曲线
    plot_training_history(history)

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load("best_model_graphsmote5.pth"))
    test_metrics = evaluate(test_loader, len(test_dataset))

    # 打印测试结果
    print("\nTest Results:")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"Test Precision: {test_metrics['precision_macro']:.4f}")
    print(f"Test Recall: {test_metrics['recall_macro']:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(test_metrics['labels'], test_metrics['predictions'])

    # 绘制每个类别的F1分数
    plot_class_f1_scores(test_metrics['f1_per_class'])

    # 打印每个类别的F1分数
    print("\nF1 Score for each class:")
    for class_idx, f1 in enumerate(test_metrics['f1_per_class']):
        print(f"Class {class_idx}: {f1:.4f}")
