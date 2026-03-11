# 训练脚本
from data_loader import load_dataset
from model import build_model
import tensorflow as tf
from pathlib import Path


def main():

    # 1 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test, le = load_dataset()

    # 2 构建模型
    model = build_model()

    model.summary()

    # 3 训练模型
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    # 4 测试集评估
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print("Test Accuracy:", test_acc)

    # 5 保存模型
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    model_path = PROJECT_ROOT / "models"

    model_path.mkdir(exist_ok=True)

    model.save(model_path / "gesture_model.h5")

    print("模型已保存")


if __name__ == "__main__":
    main()
