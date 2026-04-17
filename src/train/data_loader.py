import os
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split

# 多线程
from concurrent.futures import ThreadPoolExecutor

from src.utils.labels import LabelRepository

# ================================================================================
# 【数据可视化模块导入】
# ================================================================================
from src.tools.visualizer import plot_dataset_distribution, plot_data_validity, plot_train_val_test_split
# ================================================================================
'''
输入: .npy 文件，
输出:模型能直接吃的 6 份数据：
X_train
X_val
X_test
y_train
y_val
y_test
外加标签顺序 labels_order（供保存标签映射）
'''

# 计算并行加载的线程数（根据 CPU 核心数和任务数量动态调整，避免过多线程导致性能下降）
def _get_loader_workers(task_count: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(task_count, cpu_count * 2, 16))
# 过滤掉无效样本（形状不对的 .npy 文件），并返回有效数据和标签
def load_single_file(args):
    npy_file, label = args
    data = np.load(npy_file)

    if data.shape == (42,):
        return data, label
    return None


def load_parallel(tasks):
    """并发加载任务并聚合有效样本。"""
    X = []
    y = []
    with ThreadPoolExecutor(max_workers=_get_loader_workers(len(tasks))) as executor:
        results = executor.map(load_single_file, tasks)

    for r in results:
        if r is None:
            continue
        data, label = r
        X.append(data)
        y.append(label)

    return np.array(X), np.array(y)


def load_dataset():
    # 定义数据路径
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_path = PROJECT_ROOT / "data" / "keypoints"

    # 从统一标签仓储读取标签顺序（models/labels.json）
    label_repo = LabelRepository(PROJECT_ROOT)
    # 获取标签顺序
    labels_order = label_repo.get_labels_order()
    # 验证数据目录结构
    label_repo.validate_data_directories(data_path)

    # 收集所有文件任务
    tasks = []
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            for npy_file in class_dir.glob("*.npy"):
                tasks.append((npy_file, label))

    print("全部文件:", len(tasks))


    # 多线程读取
    #X 是二维数组，里面每条样本是 42 维[x1, y1, x2, y2, ..., x21, y21]；
    #y 还是文本标签，比如 good、left 这种字符串
    X, y = load_parallel(tasks)
    # 空样本保护
    if len(X) == 0:
        raise ValueError("未找到有效的样本。预期的 .npy 文件形状为 (42,)")

    print("有效样本:", len(X))
    print("过滤样本:", len(tasks) - len(X))

    print("原始数据形状:", X.shape)
    print("原始标签形状:", y.shape)
# ===============================================================================
    # 构建 label -> index 映射，把 y 中每个字符串标签，映射成数字。
    label_to_index = label_repo.build_index_map()
    label_repo.validate_labels(y)

    # 将字符串标签转换为索引

    m_list = []
    for label in y:
        # 每次查字典，把结果放进列表
        m = label_to_index[label]
        m_list.append(m)
    # 最后统一转成数组
    y_indices = np.array(m_list, dtype=np.int32)

    # y_indices = np.array([label_to_index[label] for label in y], dtype=np.int32)

    print("标签映射（使用 models/labels.json）:")
    for i, name in enumerate(labels_order):
        print(i, name)

    # one-hot 编码（确保 num_classes = len(labels_order)）
    num_classes = len(labels_order)
    y_onehot = to_categorical(y_indices, num_classes=num_classes)
# ===============================================================================
    # 第一次划分（使用索引作为 stratify分层抽样）
    X_train, X_temp, y_train, y_temp, _, y_temp_idx = train_test_split(
        # x 提供证据（手型特征）
        X,
        #0 -> [1,0,0,0,0,0,0,0]
        # 3 -> [0,0,0,1,0,0,0,0]
        # 7 -> [0,0,0,0,0,0,0,1]
        y_onehot,
        # [0, 0, 0, ..., 1, 1, ..., 7, 7]
        y_indices,
        test_size=0.3,
        random_state=42,
        stratify=y_indices # 按照y_indices的比例划分
    )

    # 第二次划分
    X_val, X_test, y_val, y_test = train_test_split(
        # X_temp = [
        #   [0.09, -0.01, 0.15, -0.04, ..., 0.39, -0.20],
        #   [0.14, -0.06, 0.21,  0.00, ..., 0.47, -0.25],
        #   ...
        # ]
        
        X_temp,
        # y_temp = [
        #   [1,0,0,0,0,0,0,0],   # 类别0
        #   [0,0,0,1,0,0,0,0],   # 类别3
        #   [0,0,0,0,0,0,0,1],   # 类别7
        #   ...
        # ]
        y_temp,
        test_size=0.5,
        random_state=42,
        # y_temp_idx = [0, 3, 7, 1, 1, 2, 0, ...]
        stratify=y_temp_idx
    )

    print("训练:", X_train.shape)
    print("验证:", X_val.shape)
    print("测试:", X_test.shape)

# ================================================================================
# 【生成数据处理可视化图表】
# ================================================================================
    try:
        # 创建结果目录
        results_dir = PROJECT_ROOT / "results" / "data_visualization"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据有效性对比图（英文 + 中文）
        plot_data_validity(len(tasks), len(X), save_path=results_dir / "01_data_validity.png", lang="en")
        plot_data_validity(len(tasks), len(X), save_path=results_dir / "01_数据有效性.png", lang="zh")
        
        # 原始数据集分布（在做train_test_split前的完整数据集，英文 + 中文）
        total_valid_samples = len(X)
        plot_dataset_distribution(
            list(y),
            labels_order,
            save_path=results_dir / "02_dataset_distribution.png",
            lang="en",
            total_samples=total_valid_samples,
        )
        plot_dataset_distribution(
            list(y),
            labels_order,
            save_path=results_dir / "02_数据集分布.png",
            lang="zh",
            total_samples=total_valid_samples,
        )
        
        # Train/Val/Test 分布对比
        # 需要从one-hot编码恢复为标签字符串
        y_train_labels = [labels_order[np.argmax(y_i)] for y_i in y_train]
        y_val_labels = [labels_order[np.argmax(y_i)] for y_i in y_val]
        y_test_labels = [labels_order[np.argmax(y_i)] for y_i in y_test]
        
        split_data = {
            '训练': (len(X_train), y_train_labels),
            '验证': (len(X_val), y_val_labels),
            '测试': (len(X_test), y_test_labels),
        }

        split_data_en = {
            'Train': (len(X_train), y_train_labels),
            'Validation': (len(X_val), y_val_labels),
            'Test': (len(X_test), y_test_labels),
        }
        plot_train_val_test_split(split_data_en, labels_order, save_path=results_dir / "03_train_val_test_split.png", lang="en")
        plot_train_val_test_split(split_data, labels_order, save_path=results_dir / "03_训练_验证_测试_拆分.png", lang="zh")
        
        print(f"\n[OK] 数据处理可视化已保存到: {results_dir}")
    except Exception as e:
        print(f"[WARN] 数据可视化生成失败: {e}")
# ================================================================================

    return X_train, X_val, X_test, y_train, y_val, y_test, labels_order
