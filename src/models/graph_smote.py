import torch
import numpy as np
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_scatter import scatter_add
import random


class GraphSMOTE:
    """
    GraphSMOTE实现，用于解决图级别分类任务中的类别不平衡问题。
    该实现基于论文：GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks
    但已适配为图级别分类任务。
    """

    def __init__(self, k=5, ratio=1.0, sigma=0.1, n_connect=3):
        """
        初始化GraphSMOTE

        参数:
            k (int): 近邻数量
            ratio (float): 过采样比例，1.0表示完全平衡
            sigma (float): 合成样本特征扰动程度
            n_connect (int): 每个合成节点要连接的邻居数量
        """
        self.k = k
        self.ratio = ratio
        self.sigma = sigma
        self.n_connect = n_connect

    def __call__(self, dataset):
        """
        对数据集应用GraphSMOTE

        参数:
            dataset (list): 图数据对象列表

        返回:
            augmented_dataset (list): 增强后的图数据对象列表
        """
        # 统计每个类的数量
        labels = [data.y.item() for data in dataset]
        class_counts = {}
        for label in labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # 找出多数类和少数类
        max_class = max(class_counts, key=class_counts.get)
        max_count = class_counts[max_class]

        # 创建增强后的数据集（从原始数据集开始）
        augmented_dataset = dataset.copy()

        # 对每个少数类进行过采样
        for class_label, count in class_counts.items():
            # 跳过多数类
            if count == max_count:
                continue

            # 计算需要合成的样本数量
            n_samples_to_generate = int((max_count - count) * self.ratio)

            # 获取该类的所有样本
            class_samples = [data for data in dataset if data.y.item() == class_label]

            # 如果样本过少，则无法进行SMOTE
            if len(class_samples) <= 1:
                continue

            # 为少数类生成新样本
            synthesized_samples = self._generate_synthetic_samples(
                class_samples, n_samples_to_generate, class_label
            )

            # 将合成样本添加到增强数据集中
            augmented_dataset.extend(synthesized_samples)

        return augmented_dataset

    def _generate_synthetic_samples(self, samples, n_samples, class_label):
        """
        为给定类生成合成样本

        参数:
            samples (list): 该类的样本列表
            n_samples (int): 要生成的样本数量
            class_label (int): 类别标签

        返回:
            synthetic_samples (list): 合成样本列表
        """
        synthetic_samples = []
        n_samples_per_graph = max(1, n_samples // len(samples))

        # 提取每个图的特征向量（使用全局池化的结果）
        # 这里简单地使用节点特征的平均值作为图的表示
        graph_features = [self._extract_graph_features(graph) for graph in samples]
        graph_features_tensor = torch.stack(graph_features)

        # 计算特征空间中的欧氏距离
        dist_matrix = torch.cdist(graph_features_tensor, graph_features_tensor)

        # 对每个样本找到k个最近邻
        # 排除自身（将对角线设为无穷大）
        dist_matrix.fill_diagonal_(float('inf'))
        _, knn_indices = torch.topk(dist_matrix, min(self.k, len(samples) - 1), largest=False)

        # 为每个样本生成合成图
        for i, base_graph in enumerate(samples):
            for _ in range(n_samples_per_graph):
                # 随机选择一个近邻
                neighbor_idx = knn_indices[i][torch.randint(0, knn_indices.size(1), (1,)).item()]
                neighbor_graph = samples[neighbor_idx]

                # 生成合成图
                synthetic_graph = self._synthesize_graph(base_graph, neighbor_graph, class_label)
                synthetic_samples.append(synthetic_graph)

        return synthetic_samples

    def _extract_graph_features(self, graph):
        """
        从图中提取特征表示

        参数:
            graph (Data): 图数据对象

        返回:
            features (Tensor): 图的特征表示
        """
        # 简单地使用节点特征的平均值作为图的表示
        # 更复杂的方法可以使用预训练模型的编码器部分
        return graph.x.mean(dim=0)

    def _synthesize_graph(self, graph1, graph2, class_label):
        """
        基于两个图生成一个合成图

        参数:
            graph1 (Data): 基础图
            graph2 (Data): 邻居图
            class_label (int): 类别标签

        返回:
            synthetic_graph (Data): 合成图
        """
        # 1. 复制基础图的结构
        synthetic_graph = Data()

        # 2. 节点特征插值
        alpha = random.random()  # 随机插值因子
        # 我们需要考虑两个图可能有不同数量的节点
        min_nodes = min(graph1.x.size(0), graph2.x.size(0))

        # 复制基础图的节点特征
        synthetic_graph.x = graph1.x.clone()

        # 对重叠部分进行插值
        synthetic_graph.x[:min_nodes] = (
                alpha * graph1.x[:min_nodes] + (1 - alpha) * graph2.x[:min_nodes]
        )

        # 3. 添加随机噪声以增加多样性
        noise = torch.randn_like(synthetic_graph.x) * self.sigma
        synthetic_graph.x = synthetic_graph.x + noise

        # 4. 边结构插值（保持基础图的边结构，可能添加一些图2的边）
        synthetic_graph.edge_index = graph1.edge_index.clone()

        # 可能的改进：随机添加图2中的一些边
        if graph2.edge_index.size(1) > 0 and random.random() > 0.5:
            # 随机选择一些图2的边添加到合成图中
            n_edges_to_add = min(
                int(graph2.edge_index.size(1) * 0.3),  # 最多添加30%的边
                synthetic_graph.x.size(0) * self.n_connect  # 每个节点最多添加n_connect条边
            )

            if n_edges_to_add > 0:
                # 随机选择边
                edge_indices = torch.randperm(graph2.edge_index.size(1))[:n_edges_to_add]
                additional_edges = graph2.edge_index[:, edge_indices]

                # 确保添加的边不会超出节点范围
                valid_edges = (additional_edges[0] < synthetic_graph.x.size(0)) & \
                              (additional_edges[1] < synthetic_graph.x.size(0))
                additional_edges = additional_edges[:, valid_edges]

                if additional_edges.size(1) > 0:
                    synthetic_graph.edge_index = torch.cat([
                        synthetic_graph.edge_index, additional_edges
                    ], dim=1)

        # 5. 设置类别标签
        synthetic_graph.y = torch.tensor([class_label], dtype=torch.long)

        # 6. 如果原图有边属性，也进行处理
        if hasattr(graph1, 'edge_attr') and graph1.edge_attr is not None:
            synthetic_graph.edge_attr = graph1.edge_attr.clone()

            # 如果添加了图2的边，也需要添加相应的边属性
            if 'additional_edges' in locals() and additional_edges.size(1) > 0:
                # 为新添加的边随机生成边属性
                if hasattr(graph2, 'edge_attr') and graph2.edge_attr is not None:
                    edge_indices = torch.randperm(graph2.edge_index.size(1))[:n_edges_to_add]
                    additional_edge_attr = graph2.edge_attr[edge_indices][valid_edges]

                    if additional_edge_attr.size(0) > 0:
                        synthetic_graph.edge_attr = torch.cat([
                            synthetic_graph.edge_attr, additional_edge_attr
                        ], dim=0)

        # 7. 复制其他必要的图属性
        if hasattr(graph1, 'pos') and graph1.pos is not None:
            synthetic_graph.pos = graph1.pos.clone()

        # 确保边索引不包含重复边
        synthetic_graph.edge_index, synthetic_graph.edge_attr = self._remove_duplicate_edges(
            synthetic_graph.edge_index, synthetic_graph.edge_attr if hasattr(synthetic_graph, 'edge_attr') else None
        )

        return synthetic_graph

    def _remove_duplicate_edges(self, edge_index, edge_attr=None):
        """
        移除重复的边

        参数:
            edge_index (Tensor): 边索引
            edge_attr (Tensor, optional): 边属性

        返回:
            edge_index (Tensor): 去重后的边索引
            edge_attr (Tensor, optional): 去重后的边属性
        """
        if edge_index.size(1) == 0:
            return edge_index, edge_attr

        # 创建唯一标识符
        edge_ids = edge_index[0] * edge_index.max() + edge_index[1]

        # 找到唯一边的索引
        unique_ids, inverse_indices = torch.unique(edge_ids, return_inverse=True)
        perm = torch.arange(inverse_indices.size(0), device=inverse_indices.device)
        perm = inverse_indices.new_empty(unique_ids.size(0)).scatter_(0, inverse_indices, perm)

        # 应用到边索引
        edge_index = edge_index[:, perm]

        # 如果有边属性，也应用相同的索引
        if edge_attr is not None:
            edge_attr = edge_attr[perm]

        return edge_index, edge_attr
