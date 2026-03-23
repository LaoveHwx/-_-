from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    project_root: Path
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.0
    save_model: bool = True
    model_name: str = "gesture_model.keras"

    @classmethod
    def default(cls, project_root: Path) -> "AppConfig":
        return cls(project_root=project_root)
