# 训练脚本
from data_loader import load_dataset
from model import build_model
from pathlib import Path
import json

def main():

    # 1 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test, le = load_dataset()

# 自动确定维度
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]

    # 2 构建模型 自动确定传入build_model
    model = build_model(input_dim, num_classes)
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

    print("Test Accuracy测试准确率:", test_acc)

    # 5 保存模型
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    model_path = PROJECT_ROOT / "models"

    model_path.mkdir(exist_ok=True)

    model.save(model_path / "gesture_model.keras")

    # 保存标签顺序
    labels_path = PROJECT_ROOT / "models" / "labels.json"

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(list(le.classes_), f)

    print("Labels saved:", labels_path)

    print("模型已保存为gesture_model.keras")


if __name__ == "__main__":
    main()

# # 先测试小数据
# from data_loader import load_dataset
# from model import build_model
#
#
# def main():
#
#     X_train, X_val, X_test, y_train, y_val, y_test, le = load_dataset()
#
#     num_classes = y_train.shape[1]
#
#     model = build_model(num_classes)
#
#     model.compile(
#         optimizer="adam",
#         loss="categorical_crossentropy",
#         metrics=["accuracy"]
#     )
#
#     model.summary()
#
#     history = model.fit(
#         X_train,
#         y_train,
#         validation_data=(X_val, y_val),
#         epochs=50,
#         batch_size=32
#     )
#
#     test_loss, test_acc = model.evaluate(X_test, y_test)
#
#     print("Test Accuracy:", test_acc)
#
#
# if __name__ == "__main__":
#     main()


