import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from gnn_model import GATNet
from torch.optim import Adam

# 加载之前处理好的数据集（保存为 processed_data.pt）
dataset = torch.load("C:\\Users\\86130\\Desktop\\Code\\Pycharm\\flaky_test_prediction\\data\\processed\\processed_data.pt")
print("数据集图数量:", len(dataset))

# 划分训练、验证和测试集（例如 70%/10%/20%）
num_graphs = len(dataset)
train_num = int(num_graphs * 0.6)
val_num = int(num_graphs * 0.2)
test_num = num_graphs - train_num - val_num

train_dataset = dataset[:train_num]
val_dataset = dataset[train_num:train_num + val_num]
test_dataset = dataset[train_num + val_num:]

# 构造 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 超参数设置
in_channels = dataset[0].x.size(1)  # CodeBERT 嵌入维度（例如 768）
hidden_channels = 128
num_classes = 13
learning_rate = 0.001
epochs = 500

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
        # 预测输出
        out = model(data)
        # 注意：由于每个图的标签是形如 [label]，故需 view 成一维张量
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
    # 可选：保存所有预测与真实标签，方便后续分析
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
    return accuracy, avg_loss, all_preds, all_labels


if __name__ == "__main__":
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train()
        train_acc, train_loss, _, _ = evaluate(train_loader)
        val_acc, val_loss, _, _ = evaluate(val_loader)
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        # 保存验证集上最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    # 加载最优模型并在测试集上评估
    model.load_state_dict(torch.load("best_model.pth"))
    test_acc, test_loss, preds, labels = evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
