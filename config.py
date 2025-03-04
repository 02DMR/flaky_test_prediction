# 董孟儒
# 开发时间：2025/3/3 14:57
"""
配置文件，包含项目中使用的所有参数
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

# 确保所有必要的目录存在
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# CodeBERT配置
CODEBERT_MODEL = "microsoft/codebert-base"
MAX_TOKEN_LENGTH = 512
EMBEDDING_DIM = 768  # CodeBERT默认嵌入维度

# 图神经网络配置
GNN_HIDDEN_CHANNELS = 128
GNN_NUM_LAYERS = 3
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10

# 数据集配置
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# 节点类型映射
NODE_TYPES = {
    'method': 0,
    'statement': 1,
    'expression': 2,
    'variable': 3,
    'control': 4,
    'other': 5
}

# 边类型映射
EDGE_TYPES = {
    'control_flow': 0,
    'data_dependency': 1,
    'method_call': 2,
    'contains': 3,
    'other': 4
}

# 标签映射 (根因类型)
ROOT_CAUSE_TYPES = {
    'async': 0,
    'concurrency': 1,
    'resource_leak': 2,
    'network': 3,
    'time': 4,
    'io': 5,
    'other': 6
}