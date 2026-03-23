# 封装模型构建与保存
from src.train.model import build_model
from src.utils.labels import LabelRepository
from pathlib import Path
import json

class ModelManager:
    def __init__(self, project_root):
        self.project_root = Path(project_root)

    def build(self, input_dim, num_classes):
        return build_model(input_dim, num_classes)

    def save(self, model, le):
        model_path = self.project_root / "models"
        model_path.mkdir(exist_ok=True)
        model.save(model_path / "gesture_model.keras")
        label_repo = LabelRepository(self.project_root)
        label_repo.save(list(le.classes_))
        return model_path