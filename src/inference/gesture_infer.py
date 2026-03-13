import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import deque

class GestureInference:

    def __init__(self):

        # -----------------------------
        # 1 计算项目路径（工程安全）
        # -----------------------------
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "models" / "gesture_model.keras"

        print("Loading model:", model_path)

        self.model = tf.keras.models.load_model(model_path)

        # 2 手势标签
        self.gesture_labels = [
            "good",
            "left",
            "number1","number2","number3",
            "heart",
            "right",
            "stop"
        ]
        # 滑动窗口
        self.history = deque(maxlen=5)
        # -----------------------------
        # 3 MediaPipe 初始化
        # -----------------------------
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
    # 手势预测
    # ------------------------------------------------
    # def predict(self, keypoints):
    #
    #     preds = self.model.predict(keypoints, verbose=0)
    #
    #     class_id = np.argmax(preds)
    #
    #     confidence = preds[0][class_id]
    #
    #     label = self.gesture_labels[class_id]
    #
    #     return label, confidence

    def predict(self, keypoints):

        preds = self.model.predict(keypoints, verbose=0)

        print("preds:", preds)  # 调试用

        class_id = np.argmax(preds)

        confidence = preds[0][class_id]
# 置信度过滤
        if confidence < 0.9:
            return "unknown", confidence
        self.history.append(class_id)

        # 投票
        final_id = max(set(self.history), key=self.history.count)

        return self.gesture_labels[final_id], confidence
    # ------------------------------------------------
    # 主循环
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

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:

                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    # 提取关键点
                    keypoints = self.extract_keypoints(hand_landmarks)
                    keypoints = self.normalize_keypoints(keypoints)
                    keypoints = keypoints.reshape(1, -1)
                    # 推理
                    gesture, conf = self.predict(keypoints)

                    # 计算包围盒
                    x_list = [lm.x for lm in hand_landmarks.landmark]
                    y_list = [lm.y for lm in hand_landmarks.landmark]

                    x1 = int(min(x_list) * w)
                    y1 = int(min(y_list) * h)

                    x2 = int(max(x_list) * w)
                    y2 = int(max(y_list) * h)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    text = f"{gesture} {conf:.2f}"

                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    infer = GestureInference()

    infer.run()

