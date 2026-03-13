import numpy as np

def normalize_keypoints(keypoints):
    # 支持输入：(21,3)、(21,2)、(42,)
    kp = np.array(keypoints)
    if kp.ndim == 2 and kp.shape[1] == 3:
        pts = kp[:, :2]
    elif kp.ndim == 2 and kp.shape[1] == 2:
        pts = kp
    elif kp.ndim == 1 and kp.size == 42:
        pts = kp.reshape(21, 2)
    else:
        raise ValueError(f"Unsupported keypoints shape: {kp.shape}")

    wrist = pts[0]
    pts = pts - wrist
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts = pts / scale
    return pts.flatten().astype("float32")
