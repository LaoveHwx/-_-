import json
from pathlib import Path
from typing import List, Optional

DEFAULT_LABELS = ['good', 'left', 'number1', 'number2', 'number3', 'heart', 'right', 'stop']

def get_labels_order(project_root: Path, default: Optional[List[str]] = None, save_if_missing: bool = True) -> List[str]:
    """
    返回 labels_order（确定的类别顺序）。
    优先读取 project_root / models / labels.json；
    如果不存在，则使用 default（若未提供则用 DEFAULT_LABELS），并可选择保存为 labels.json 以供后续使用。
    """
    if default is None:
        default = DEFAULT_LABELS

    labels_path = Path(project_root) / "models" / "labels.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return labels
    else:
        labels = default
        if save_if_missing:
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)
        return labels
