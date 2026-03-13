import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# 多线程版本
from concurrent.futures import ThreadPoolExecutor
from src.utils.labels import get_labels_order

def load_single_file(args):
    npy_file, label = args
    data = np.load(npy_file)

    if data.shape == (42,):
        return data, label
    return None


def load_dataset():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_path = PROJECT_ROOT / "data" / "keypoints"

    # 从统一配置读取标签顺序（models/labels.json）
    labels_order = get_labels_order(PROJECT_ROOT)

    tasks = []

    # 收集所有文件任务
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            for npy_file in class_dir.glob("*.npy"):
                tasks.append((npy_file, label))

    print("全部文件:", len(tasks))

    X = []
    y = []

    # 多线程读取
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_single_file, tasks)

    for r in results:
        if r is not None:
            data, label = r
            X.append(data)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("原始数据形状:", X.shape)
    print("原始标签形状:", y.shape)

    # 构建 label -> index 映射（按照 labels_order 的顺序）
    label_to_index = {label: idx for idx, label in enumerate(labels_order)}

    # 检查是否存在未知标签（防止目录名拼写错误）
    unknowns = set(y) - set(label_to_index.keys())
    if unknowns:
        raise ValueError(f"Found unknown labels not in labels_order: {unknowns}")

    # 将字符串标签转换为索引
    y_indices = np.array([label_to_index[label] for label in y], dtype=np.int32)

    print("标签映射（使用 models/labels.json）:")
    for i, name in enumerate(labels_order):
        print(i, name)

    # one-hot 编码（确保 num_classes = len(labels_order)）
    num_classes = len(labels_order)
    y_onehot = to_categorical(y_indices, num_classes=num_classes)

    # 第一次划分（使用索引作为 stratify）
    X_train, X_temp, y_train, y_temp, y_train_idx, y_temp_idx = train_test_split(
        X,
        y_onehot,
        y_indices,
        test_size=0.3,
        random_state=42,
        stratify=y_indices
    )

    # 第二次划分
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp_idx
    )

    print("训练:", X_train.shape)
    print("验证:", X_val.shape)
    print("测试:", X_test.shape)

    # 为兼容原训练脚本中使用的 LabelEncoder.le.classes_，返回一个简单对象
    class DummyLE:
        def __init__(self, classes):
            import numpy as _np
            self.classes_ = _np.array(classes)

    le = DummyLE(labels_order)

    return X_train, X_val, X_test, y_train, y_val, y_test, le
