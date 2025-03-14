# import os
# import torch
# import torch.nn.functional as F
# from torch_geometric.data import DataLoader
# from gnn_model import GATNet
# from torch.optim import Adam
# from sklearn.metrics import f1_score  # Removed Precision import
# import random  # Added for shuffling
#
# # Load the previously processed dataset
# dataset = torch.load("C:\\Users\\86130\\Desktop\\Code\\Pycharm\\flaky_test_prediction\\data\\processed\\processed_data1.pt")
# print("Number of graphs in dataset:", len(dataset))
#
# # Set a fixed random seed for reproducibility
# random.seed(42)
# torch.manual_seed(42)
#
# # Shuffle the dataset before splitting
# random.shuffle(dataset)
#
# # Split into train, validation, and test sets (70%/10%/20%)
# num_graphs = len(dataset)
# train_num = int(num_graphs * 0.6)
# val_num = int(num_graphs * 0.2)
# test_num = num_graphs - train_num - val_num
#
# train_dataset = dataset[:train_num]
# val_dataset = dataset[train_num:train_num + val_num]
# test_dataset = dataset[train_num + val_num:]
#
# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#
# # Hyperparameters
# in_channels = dataset[0].x.size(1)  # CodeBERT embedding dimension (e.g., 768)
# hidden_channels = 128
# num_classes = 13
# learning_rate = 0.0002
# epochs = 250
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = GATNet(in_channels, hidden_channels, num_classes).to(device)
# optimizer = Adam(model.parameters(), lr=learning_rate)
# criterion = torch.nn.CrossEntropyLoss()
#
#
# def train():
#     model.train()
#     total_loss = 0
#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         # Prediction output
#         out = model(data)
#         # Note: Each graph's label is in [label] format, so we need to view it as a 1D tensor
#         loss = criterion(out, data.y.view(-1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * data.num_graphs
#     return total_loss / len(train_dataset)
#
#
# def evaluate(loader):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0
#     # Collect all predictions and true labels for further metric calculation
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             out = model(data)
#             loss = criterion(out, data.y.view(-1))
#             total_loss += loss.item() * data.num_graphs
#             preds = out.argmax(dim=1)
#             correct += (preds == data.y.view(-1)).sum().item()
#             total += data.num_graphs
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(data.y.view(-1).cpu().numpy())
#     avg_loss = total_loss / total
#     accuracy = correct / total
#     # Calculate F1 score using macro averaging
#     f1 = f1_score(all_labels, all_preds, average='macro')
#     return accuracy, avg_loss, f1, all_preds, all_labels
#
#
# if __name__ == "__main__":
#     best_val_acc = 0
#     for epoch in range(1, epochs + 1):
#         loss = train()
#         train_acc, train_loss, train_f1, _, _ = evaluate(train_loader)
#         val_acc, val_loss, val_f1, _, _ = evaluate(val_loader)
#         print(f"Epoch: {epoch:03d}, "
#               f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
#               f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
#         # Save the best model based on validation accuracy
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "best_model.pth")
#
#     # Load the best model and evaluate on the test set
#     model.load_state_dict(torch.load("best_model.pth"))
#     test_acc, test_loss, test_f1, preds, labels = evaluate(test_loader)
#     print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")


import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from gnn_model import GATNet
from torch.optim import Adam
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import random
import matplotlib.pyplot as plt
import seaborn as sns

# 加载预处理好的数据集
dataset = torch.load("C:\\Users\\86130\\Desktop\\Code\\Pycharm\\flaky_test_prediction\\data\\processed\\processed_data1.pt")
print("Number of graphs in dataset:", len(dataset))

# 固定随机种子以保证结果可复现
random.seed(123)
torch.manual_seed(123)

# 在划分数据集前打乱顺序
random.shuffle(dataset)

# 划分训练集、验证集和测试集（60%/20%/20%）
num_graphs = len(dataset)
train_num = int(num_graphs * 0.7)
val_num = int(num_graphs * 0.2)
test_num = num_graphs - train_num - val_num
train_dataset = dataset[:train_num]
val_dataset = dataset[train_num:train_num + val_num]
test_dataset = dataset[train_num + val_num:]

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 超参数设置
in_channels = dataset[0].x.size(1)  # CodeBERT嵌入维度（例如768）
hidden_channels = 128
num_classes = 13
learning_rate = 0.0002
epochs = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATNet(in_channels, hidden_channels, num_classes).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

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
    return total_loss / len(train_dataset)

def evaluate(loader):
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
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    return accuracy, avg_loss, f1, precision, recall, all_preds, all_labels

if __name__ == "__main__":
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train()
        train_acc, train_loss, train_f1, train_precision, train_recall, _, _ = evaluate(train_loader)
        val_acc, val_loss, val_f1, val_precision, val_recall, _, _ = evaluate(val_loader)
        print(f"Epoch: {epoch:03d}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
              f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        # 保存验证集准确率最高的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    # 加载最优模型并在测试集上评估
    model.load_state_dict(torch.load("best_model.pth"))
    test_acc, test_loss, test_f1, test_precision, test_recall, preds, labels = evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, "
          f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

    # 生成多分类结果的混淆矩阵图像
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("multiclass_results.png")
    plt.show()

