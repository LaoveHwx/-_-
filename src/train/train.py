import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig
from src.usecases.train_usecase import TrainUseCase


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = AppConfig.default(project_root)

    trainer = TrainUseCase(config)
    result = trainer.run()

    print(f"Test Accuracy测试准确率: {result['test_acc']}")
    if result["model_path"] is not None:
        print("Labels saved:", result["model_path"] / "labels.json")
        print("模型已保存为gesture_model.keras")


if __name__ == "__main__":
    main()


