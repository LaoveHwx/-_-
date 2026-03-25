import numpy as np


def extract_xy_keypoints(hand_landmarks):
    """将21个(x, y)关键点提取到形状为(42,)的扁平 float32 向量中。"""
    points = []
    for lm in hand_landmarks.landmark:
        points.append(lm.x)
        points.append(lm.y)
    return np.array(points, dtype="float32")

def normalize_keypoints(keypoints):
    """将关键点归一化：以手腕为中心平移，并按中指第2关节长度缩放。"""
    # 支持输入：(21,3)、(21,2)、(42,)
    kp = np.array(keypoints)
    if kp.ndim == 2 and kp.shape[1] == 3: # ndim判断维度
        pts = kp[:, :2]# 所有行，前两列，丢掉Z坐标
    elif kp.ndim == 2 and kp.shape[1] == 2:
        pts = kp
    elif kp.ndim == 1 and kp.size == 42:
        pts = kp.reshape(21, 2)
    else:
        raise ValueError(f"Unsupported keypoints shape: {kp.shape}")

    wrist = pts[0]# 取手腕坐标
    pts = pts - wrist # 以手腕为中心平移
    scale = np.linalg.norm(pts[9]) #取中指第 2 关节长度作为缩放基准  norm是算模长 / 距离
    if scale > 1e-6:
        pts = pts / scale # 缩放归一化：无论手大小，输出长度一致
    return pts.flatten().astype("float32")
# 展平成 1 维 42 个数字；并转换为 float32 类型，适合后续模型输入
