from dataclasses import dataclass
from pathlib import Path
# 只存数据类
@dataclass
class AppConfig:
    #项目根路径
    project_root: Path
    # 训练轮数
    epochs: int = 50
    # 批大小
    batch_size: int = 32
    # 验证集比例
    validation_split: float = 0.0
    # 是否保存模型
    save_model: bool = True
    # 模型文件名
    model_name: str = "gesture_model.keras"
    # 把方法变成类方法，不需要实例化对象，直接用 类名.方法() 调用
    @classmethod
    def default(cls, project_root: Path) -> "AppConfig":
        return cls(project_root=project_root)
