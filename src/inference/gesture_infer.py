import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import deque
from src.utils.labels import LabelRepository

class GestureInference:

    def __init__(self, conf_threshold: float = 0.7, history_len: int = 5):
        # 计算项目根路径（文件位于 src/inference -> parents[2] 通常到项目根）
        self.project_root = Path(__file__).resolve().parents[2]
        model_path = self.project_root / "models" / "gesture_model.keras"

        # 加载模型
        print("Loading model:", model_path)
        self.model = tf.keras.models.load_model(model_path)

        # 统一从 models/labels.json 读取标签顺序（通过 LabelRepository 统一管理）
        label_repo = LabelRepository(self.project_root)
        self.gesture_labels = label_repo.get_labels_order()
        print("Loaded gesture labels:", self.gesture_labels)

        # 参数
        self.conf_threshold = conf_threshold

        # 滑动窗口（用于投票）
        self.history = deque(maxlen=history_len)
        self._conf_history = deque(maxlen=history_len)

        # MediaPipe 初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

    # ------------------------------------------------
    # 关键点提取（必须 42 维）再归一化
    # ------------------------------------------------
    def extract_keypoints(self, hand_landmarks):
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.append(lm.x)
            keypoints.append(lm.y)
        return np.array(keypoints, dtype="float32")

    def normalize_keypoints(self, keypoints):
        pts = keypoints.reshape(21, 2)
        wrist = pts[0]
        pts = pts - wrist
        scale = np.linalg.norm(pts[9])
        if scale > 1e-6:
            pts = pts / scale
        return pts.flatten().astype("float32")

    # ------------------------------------------------
    # 手势预测（返回与投票一致的置信度）
    # ------------------------------------------------
    def predict(self, keypoints):
        keypoints = np.asarray(keypoints, dtype="float32")
        preds = self.model.predict(keypoints, verbose=0)  # shape (1, num_classes)

        # print("preds:", preds)  # 调试用

        class_id = int(np.argmax(preds))
        confidence = float(preds[0][class_id])

        # 置信度阈值判断
        if confidence < self.conf_threshold:
            return "unknown", confidence

        # 维护 id 与 confidence 历史
        self.history.append(class_id)
        self._conf_history.append(confidence)

        # 投票决定最终类别
        final_id = max(set(self.history), key=self.history.count)

        # 计算 final_id 在历史中对应的平均置信度
        try:
            ids = list(self.history)
            confs = list(self._conf_history)
            if len(ids) != len(confs):
                avg_conf = confidence
            else:
                matched = [c for i, c in zip(ids, confs) if i == final_id]
                avg_conf = float(np.mean(matched)) if matched else confidence
        except Exception:
            avg_conf = confidence

        return self.gesture_labels[final_id], avg_conf

    # ------------------------------------------------
    # 主循环（注意：为了与训练数据保持一致，这里不对帧做左右镜像 flip）
    # 若你想镜像显示给用户，可在显示前单独 flip，但不要在送入检测/归一化前 flip。
    # ------------------------------------------------
    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Camera open failed")
            return

        print("Press ESC to exit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 为保证与训练数据一致，送入 MediaPipe 前不做水平翻转
            # （如果你采集时也用的是镜像显示，需确保采集与训练一致）
            # frame_for_processing = cv2.flip(frame, 1)  # 不要这么做，除非采集也用了 flip
            frame_for_processing = frame

            h, w, _ = frame_for_processing.shape
            rgb = cv2.cvtColor(frame_for_processing, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 在 frame_for_processing 上绘制（保持坐标一致）
                    self.mp_drawing.draw_landmarks(
                        frame_for_processing,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    # 提取关键点并归一化
                    keypoints = self.extract_keypoints(hand_landmarks)
                    keypoints = self.normalize_keypoints(keypoints)
                    keypoints = keypoints.reshape(1, -1)

                    # 推理
                    gesture, conf = self.predict(keypoints)

                    # 计算包围盒（使用 normalized landmark 的原始 x,y）
                    x_list = [lm.x for lm in hand_landmarks.landmark]
                    y_list = [lm.y for lm in hand_landmarks.landmark]
                    x1 = int(min(x_list) * w)
                    y1 = int(min(y_list) * h)
                    x2 = int(max(x_list) * w)
                    y2 = int(max(y_list) * h)
                    cv2.rectangle(frame_for_processing, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    text = f"{gesture} {conf:.2f}"
                    cv2.putText(
                        frame_for_processing,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            # 显示同一帧（未镜像,使用和采集方向一致即可），这样绘制与展示坐标一致
            cv2.imshow("Gesture Recognition", frame_for_processing)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    infer = GestureInference()
    infer.run()