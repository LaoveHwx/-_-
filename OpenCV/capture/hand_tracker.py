import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
# 借用指令from src.capture.hand_tracker import HandTracker

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
    # 处理帧
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        keypoints = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                points = []
                for lm in hand_landmarks.landmark:
                    points.append([lm.x, lm.y, lm.z])
                raw_keypoints = np.array(points)
                keypoints = self.normalize_keypoints(raw_keypoints)

        return frame, keypoints
    # 手部关键点归一化
    def normalize_keypoints(self, keypoints):
        keypoints = keypoints.reshape(21, 3)

        # 以手腕为原点
        wrist = keypoints[0]
        keypoints = keypoints - wrist

        # 只取 x,y
        keypoints = keypoints[:, :2]

        # 计算最大距离
        max_value = np.max(np.abs(keypoints))
        if max_value != 0:
            keypoints = keypoints / max_value

        return keypoints.flatten()  # (42,)

GESTURES = {
    ord('1'): 'good',
    ord('2'): 'left',
    ord('3'): 'number1',
    ord('4'): 'number2',
    ord('5'): 'number3',
    ord('6'): 'ok',
    ord('7'): 'right',
    ord('8'): 'stop',
}

SAMPLES_PER_CLASS = 500

def run_camera_test():
    cap = cv2.VideoCapture(0) # 0 默认摄像头
    tracker = HandTracker() # 创建手部检测器

    # 项目根目录，当前目录上2级parents[2]
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # 数据目录
    base_path = PROJECT_ROOT / "data" / "keypoints"
    # 创建数据目录
    os.makedirs(base_path, exist_ok=True)
    # 当前类别
    current_label = None
    # 样本计数
    sample_count = 0
    # 采集状态
    collecting = False

    print("1~8 选择类别")
    print("c 开始采集")
    print("v 暂停采集")
    print("ESC 退出")

    while True:
        # 从摄像头捕获一帧图像
        # cap.read() 返回一个元组 (ret, frame)
        ret, frame = cap.read()
        # 检查是否成功读取到帧
        # 如果 ret 为 False，表示读取失败（例如摄像头断开或视频结束）
        if not ret:
            # 跳出循环，结束程序
            break
        # 处理帧：执行手部检测与关键点提取，并等待键盘输入（1ms）
        frame, keypoints = tracker.process_frame(frame)
        key = cv2.waitKey(1) & 0xFF

        # 选择类别
        if key in GESTURES:
            current_label = GESTURES[key]
            class_dir = os.path.join(base_path, current_label)
            os.makedirs(class_dir, exist_ok=True)

            count_file = os.path.join(class_dir, "count.txt")

            if os.path.exists(count_file):
                with open(count_file, "r") as f:
                    sample_count = int(f.read().strip())
            else:
                sample_count = 0

            print(f"当前类别: {current_label}")
            print(f"已存在样本: {sample_count}")
            collecting = False

        # 开始采集
        if key == ord('c') and current_label:
            if sample_count >= SAMPLES_PER_CLASS:
                print("该类别已达 300，无需继续")
            else:
                collecting = True
                print("开始采集")

        # 暂停采集
        if key == ord('v') and current_label:
            collecting = False
            class_dir = os.path.join(base_path, current_label)
            count_file = os.path.join(class_dir, "count.txt")

            with open(count_file, "w") as f:
                f.write(str(sample_count))

            print("已暂停并保存进度")

        # 采集逻辑
        if collecting and keypoints is not None:
            if sample_count < SAMPLES_PER_CLASS:
                class_dir = os.path.join(base_path, current_label)

                file_path = os.path.join(
                    class_dir,
                    f"{sample_count}.npy"
                )

                np.save(file_path, keypoints)
                sample_count += 1

                cv2.putText(
                    frame,
                    f"{current_label}: {sample_count}/{SAMPLES_PER_CLASS}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:
                collecting = False
                print("该手势类别采集完成")

        cv2.imshow("数据收集", frame)
        # ESC 退出
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_test()
