# 标签编码，划分训练集
# import sys
# from pathlib import Path

# # 将项目根目录添加到 Python 路径
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(PROJECT_ROOT))

from src.train.data_loader import load_dataset
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, labels = load_dataset()


