import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def load_dataset():

    # 定位项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_path = PROJECT_ROOT / "data" / "keypoints"

    X = []
    y = []

    # 遍历每个类别文件夹
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():

            label = class_dir.name

            for npy_file in class_dir.glob("*.npy"):
                data = np.load(npy_file)

                if data.shape == (42,):
                    X.append(data)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("原始数据形状:", X.shape)
    print("原始标签形状:", y.shape)

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ===测试代码开始===
    print("标签映射:")
    for i, name in enumerate(le.classes_):
        print(i, name)
    # ===测试代码结束===

    # one-hot
    # 将整数编码的标签转换为 one-hot 编码，用于多分类任务
    y_onehot = to_categorical(y_encoded)

    # ===测试代码开始===
    print("onehot example:")
    print(y_onehot[:5])
    # ===测试代码结束===

    # 划分训练/测试
    X_train, X_temp, y_train, y_temp, y_train_label, y_temp_label = train_test_split(
        X,y_onehot,y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp_label
    )

    print("训练集:", X_train.shape)
    print("验证集:", X_val.shape)
    print("测试集:", X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test, le