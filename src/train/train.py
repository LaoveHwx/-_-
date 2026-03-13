# import sys
# from pathlib import Path

# # 将项目根目录添加到 Python 路径
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(PROJECT_ROOT))

# 训练脚本
from pathlib import Path
from src.core.data_manager import DataManager
from src.core.model_manager import ModelManager


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    dm = DataManager()
    X_train, X_val, X_test, y_train, y_val, y_test, le = dm.load()

    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]

    mm = ModelManager(PROJECT_ROOT)
    model = mm.build(input_dim, num_classes)
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test Accuracy测试准确率:", test_acc)

    model_path = mm.save(model, le)
    print("Labels saved:", model_path / "labels.json")
    print("模型已保存为gesture_model.keras")

if __name__ == "__main__":
    main()


