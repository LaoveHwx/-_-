# 标签编码，划分训练集
from data_loader import load_dataset

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, le = load_dataset()


