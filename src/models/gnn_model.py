import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.3, heads=8):
        super(GATNet, self).__init__()
        # 第一层 GAT，使用多头注意力，输出维度为 hidden_channels * heads
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # 第二层 GAT，使用单头注意力（不拼接），输出维度为 hidden_channels
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)

        # 定义一个简单的 gate 网络，用于 Global Attention Pooling
        self.attention_gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )
        # Global Attention Pooling 可以根据 gate 得到的权重对节点特征进行加权聚合
        self.global_pool = GlobalAttention(gate_nn=self.attention_gate)

        # 全连接层输出分类结果
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 第一层 GAT + 激活 + Dropout
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层 GAT + 激活
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # 使用 Global Attention Pooling 将节点级表示聚合为图级表示
        x = self.global_pool(x, batch)

        # 通过全连接层得到分类结果
        x = self.fc(x)
        return x



