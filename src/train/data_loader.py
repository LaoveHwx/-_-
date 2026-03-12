import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
# 多线程版本
from concurrent.futures import ThreadPoolExecutor

def load_single_file(args):
    npy_file, label = args
    data = np.load(npy_file)

    if data.shape == (42,):
        return data, label
    return None


def load_dataset():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_path = PROJECT_ROOT / "data" / "keypoints"

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

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("标签映射:")
    for i, name in enumerate(le.classes_):
        print(i, name)

    y_onehot = to_categorical(y_encoded)

    # 第一次划分
    X_train, X_temp, y_train, y_temp, y_train_label, y_temp_label = train_test_split(
        X,
        y_onehot,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    # 第二次划分
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp_label
    )

    print("训练:", X_train.shape)
    print("验证:", X_val.shape)
    print("测试:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, le
