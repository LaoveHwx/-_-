from collections import deque
import numpy as np


def hwv(history: deque, conf_history: deque, class_id: int, confidence: float) -> tuple[int, float]:
    """历史窗口投票。
    将当前预测附加到历史窗口，
    然后返回投票的类别ID以及
    当前窗口中该类别的平均置信度。
    """
    history.append(class_id)
    conf_history.append(confidence)

    final_id = max(set(history), key=history.count)

    ids = list(history)
    confs = list(conf_history)
    if len(ids) != len(confs):
        return final_id, confidence

    matched = [c for i, c in zip(ids, confs) if i == final_id]
    if not matched:
        return final_id, confidence

    return final_id, float(np.mean(matched))
