from typing import Optional
from src.core.config import AppConfig
from src.core.model_manager import ModelManager
from src.core.data_manager import DataManager
from pathlib import Path


class TrainUseCase:
    """训练用例：封装训练流程，便于单元测试与依赖注入。"""

    def __init__(self, config: AppConfig, data_manager: Optional[DataManager] = None, model_manager: Optional[ModelManager] = None):
        self.config = config
        self.data_manager = data_manager or DataManager()
        self.model_manager = model_manager or ModelManager(self.config.project_root)

    def run(self, epochs: Optional[int] = None, batch_size: Optional[int] = None):
        epochs = epochs if epochs is not None else self.config.epochs
        batch_size = batch_size if batch_size is not None else self.config.batch_size

        X_train, X_val, X_test, y_train, y_val, y_test, le = self.data_manager.load()

        input_dim = X_train.shape[1]
        num_classes = y_train.shape[1]

        model = self.model_manager.build(input_dim, num_classes)
        model.summary()

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )

        test_loss, test_acc = model.evaluate(X_test, y_test)

        model_path = None
        if self.config.save_model:
            model_path = self.model_manager.save(model, le)

        return {
            "history": history,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "model_path": model_path,
        }
