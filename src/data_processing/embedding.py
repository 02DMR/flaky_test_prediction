import os
import torch
import networkx as nx
import pydot
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel

# 定义不稳定原因与数字标签之间的映射
label_map = {
    "async wait": 0,
    "test order dependency": 1,
    "unordered collections": 2,
    "concurrency": 3,
    "time": 4,
    "network": 5,
    "randomness": 6,
    "test case timeout": 7,
    "resource leak": 8,
    "platform dependency": 9,
    "too restrictive range": 10,
    "i_o": 11,
    "floating point operations": 12,
}

# 初始化 CodeBERT 模型和分词器（使用 microsoft/codebert-base）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
model.eval()

def get_node_embedding(text):
    """
    使用 CodeBERT 将文本转换为向量表示（取 [CLS] token 对应的隐藏状态）
    """
    with torch.no_grad():
        # 控制输入长度，必要时可根据需要调整 max_length
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        # 取 CLS token 的输出（shape: [1, hidden_dim]）
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.cpu().squeeze(0)  # shape: [hidden_dim]

def process_dot_file(file_path):
    """
    解析一个 .dot 文件：
      - 通过文件名解析不稳定原因标签；
      - 利用 pydot 和 networkx 解析图结构；
      - 对每个节点根据其属性（优先使用 'label' 属性）生成嵌入向量；
      - 构造 PyG Data 对象，包含节点特征 x、边索引 edge_index、图级标签 y。
    """
    base = os.path.basename(file_path)
    # 假设文件名格式为 "xxx@<原因类型>.dot"
    if "@" in base:
        label_str = base.split("@")[1].split(".")[0]
    else:
        label_str = "unknown"
    # 映射为数字标签，如果未找到则设为 -1（或根据需要处理）
    label = label_map.get(label_str, -1)

    # 解析 dot 文件
    graphs = pydot.graph_from_dot_file(file_path)
    if not graphs:
        print(f"无法解析 {file_path}")
        return None
    pydot_graph = graphs[0]
    # 转换为 networkx 图（默认得到有向图，可根据需要转换为无向图）
    graph = nx.nx_pydot.from_pydot(pydot_graph)

    # 建立节点编号映射，确保后续构造 edge_index 时节点顺序一致
    node_list = list(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # 生成节点特征：对于每个节点，若存在 "label" 属性，则使用该属性文本，否则使用节点名称
    node_features = []
    for node in node_list:
        node_attr = graph.nodes[node]
        text = node_attr.get("label", str(node))
        embedding = get_node_embedding(text)
        node_features.append(embedding.unsqueeze(0))
    if len(node_features) == 0:
        print(f"{file_path} 中未找到节点信息")
        return None
    x = torch.cat(node_features, dim=0)  # [num_nodes, embedding_dim]

    # 构造 edge_index（shape: [2, num_edges]）
    edges = list(graph.edges())
    if edges:
        edge_index = torch.tensor(
            [[node_to_idx[src], node_to_idx[dst]] for src, dst in edges],
            dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 构造图数据对象，y 为图级标签（需为 long 类型，注意这里是单一数字标签）
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    return data

def process_all_dot_files(raw_dir, processed_file="processed_data.pt"):
    """
    遍历 raw_dir 目录下所有 .dot 文件，
    对每个文件调用 process_dot_file 得到 PyG Data 对象，
    最后保存所有图数据到 processed_file 中。
    """
    data_list = []
    for file in os.listdir(raw_dir):
        if file.endswith(".dot"):
            file_path = os.path.join(raw_dir, file)
            data = process_dot_file(file_path)
            if data is not None:
                data_list.append(data)
                print(f"处理 {file} 成功")
            else:
                print(f"处理 {file} 失败")
    torch.save(data_list, processed_file)
    print(f"共处理 {len(data_list)} 个图，保存至 {processed_file}")

if __name__ == "__main__":
    # 指定 .dot 文件目录（根据实际情况修改路径）
    raw_directory = r"C:\Users\86130\Desktop\Code\Pycharm\flaky_test_prediction\data\raw"
    # 指定保存处理后的数据文件（可放置于项目 processed 目录下）
    output_file = "../../data/processed/processed_data.pt"
    process_all_dot_files(raw_directory, processed_file=output_file)
