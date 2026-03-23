import json
from pathlib import Path
from typing import Iterable, List, Optional

DEFAULT_LABELS = ['good', 'left', 'number1', 'number2', 'number3', 'heart', 'right', 'stop']


class LabelRepository:
    """统一的标签读写与管理仓储。

    功能：
    - 解析 models/labels.json 路径
    - 加载 / 保存 标签列表
    - 在缺失时按默认值创建 labels.json
    """

    def __init__(self, project_root: Path, default: Optional[List[str]] = None) -> None:
        self.project_root = Path(project_root)
        self.default = list(default) if default is not None else list(DEFAULT_LABELS)

    @property
    def labels_path(self) -> Path:
        return self.project_root / "models" / "labels.json"

    def load(self, save_if_missing: bool = True) -> List[str]:
        """加载标签顺序；如果文件不存在，可根据需要自动创建。"""
        path = self.labels_path
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            return labels

        labels = self.default
        if save_if_missing:
            self.save(labels)
        return labels

    def save(self, labels: List[str]) -> None:
        """保存标签顺序到 models/labels.json。"""
        path = self.labels_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(labels), f, ensure_ascii=False, indent=2)

    def get_labels_order(self, save_if_missing: bool = True) -> List[str]:
        """语义化包装，供上层直接获取标签顺序使用。"""
        return self.load(save_if_missing=save_if_missing)

    def build_index_map(self) -> dict[str, int]:
        """基于 labels.json 中的固定顺序构建 label -> index 映射。"""
        labels = self.get_labels_order()
        return {label: idx for idx, label in enumerate(labels)}

    def validate_labels(self, labels: Iterable[str]) -> None:
        """校验给定标签集合是否都在 labels.json 中定义。"""
        known_labels = set(self.get_labels_order())
        unknown_labels = set(labels) - known_labels
        if unknown_labels:
            raise ValueError(f"Found unknown labels not in models/labels.json: {sorted(unknown_labels)}")

    def validate_data_directories(self, data_path: Path) -> None:
        """校验数据目录名与 labels.json 是否一致，但不要求目录顺序一致。"""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Keypoint data path not found: {data_path}")

        defined_labels = set(self.get_labels_order())
        directory_labels = {path.name for path in data_path.iterdir() if path.is_dir()}

        missing_labels = defined_labels - directory_labels
        unknown_labels = directory_labels - defined_labels

        if missing_labels or unknown_labels:
            details = []
            if missing_labels:
                details.append(f"missing directories for labels: {sorted(missing_labels)}")
            if unknown_labels:
                details.append(f"unexpected directories not in labels.json: {sorted(unknown_labels)}")
            raise ValueError("Data directory labels mismatch: " + "; ".join(details))


def get_labels_order(project_root: Path, default: Optional[List[str]] = None, save_if_missing: bool = True) -> List[str]:
    """保持与旧接口兼容的快捷函数，内部委托给 LabelRepository。

    这样现有代码可以逐步迁移到面向对象用法，而不用一次性全改。
    """
    repo = LabelRepository(project_root, default)
    return repo.get_labels_order(save_if_missing=save_if_missing)
