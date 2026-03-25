from typing import Optional
from src.core.config import AppConfig
from src.core.model_manager import ModelManager
from src.core.data_manager import DataManager


class TrainUseCase:
    """训练用例：封装训练流程，便于单元测试与依赖注入。"""

    def __init__(self, config: AppConfig, 
                 data_manager: Optional[DataManager] = None, 
                 model_manager: Optional[ModelManager] = None):
        
        self.config = config
        self.data_manager = data_manager or DataManager()
        self.model_manager = model_manager or ModelManager(self.config.project_root)

    def run(self, epochs: Optional[int] = None, batch_size: Optional[int] = None):
        """训练模型。

        Args:“可覆盖默认参数”
            epochs: 训练轮数。默认为 None，使用配置文件中的默认值。
            batch_size: 批次大小。默认为 None，使用配置文件中的默认值。

        Returns:
            dict: 训练结果，包含训练历史、测试损失、测试准确率和模型保存路径的字典。
        """
        epochs = epochs if epochs is not None else self.config.epochs
        batch_size = batch_size if batch_size is not None else self.config.batch_size

        # 1.加载数据，同时拿到固定标签顺序用于保存映射
        X_train, X_val, X_test, y_train, y_val, y_test, labels = self.data_manager.load()

        #从训练数据实际形状里推出来输入维度和类别数量，避免硬编码
        input_dim = X_train.shape[1]
        num_classes = y_train.shape[1]

        # 2.构建模型
        model = self.model_manager.build(input_dim, num_classes)
        model.summary()
        # 3.训练模型
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )

        # 4.测试集做评估
        test_loss, test_acc = model.evaluate(X_test, y_test)
        # 5.保存模型和标签顺序
        model_path = None
        if self.config.save_model:
            model_path = self.model_manager.save(model, labels)
        # 6.返回训练历史、测试损失、测试准确率和模型保存路径的字典
        return {
            "history": history,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "model_path": model_path,
        }
