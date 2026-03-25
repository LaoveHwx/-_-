import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
from src.utils.normalizer import normalize_keypoints
from src.utils.labels import LabelRepository
SAMPLES_PER_CLASS = 500
# import sys
# 运行的时候把这个部分解除注释就行
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# sys.path.append(str(PROJECT_ROOT))


# 算法层
class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands  # type: ignore
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,# 只检测一只手
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore

    # 处理帧
    # 输入：一帧图像；
    # 输出：绘制关键点后的帧和归一化关键点
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# 转换为 RGB
        results = self.hands.process(rgb_frame)

        keypoints = None
            # 判断检测是否有手部关键点，如果有则绘制并提取关键点坐标
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks( # MP自带画图方法
                    frame, # 原图
                    hand_landmarks,# 单只手的关键点
                    self.mp_hands.HAND_CONNECTIONS,# MP自带手指连线规则
                )

                points = []

                for lm in hand_landmarks.landmark: # MP自带结构，存 21 个关键点
                    points.append([lm.x, lm.y, lm.z])
                raw_keypoints = np.array(points)
                # 覆盖结果，返回将（21,3）变为（21，2）的一堆坐标数据
                keypoints = normalize_keypoints(raw_keypoints)

        return frame, keypoints


# 流程层
class HandDataCollector:
    """负责摄像头采集、按键控制和关键点保存的类。"""

    def __init__(self, project_root: Path | None = None, camera_index: int = 0):# 打开摄像头的编号设置0

        self.cap = cv2.VideoCapture(camera_index)
        print(f"打开电脑摄像头")
        # 手部关键点跟踪
        self.tracker = HandTracker()
        print("手部关键点跟踪初始化完成")


        # 项目根目录，当前目录上 2 级 parents[2]
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.base_path = self.project_root / "data" / "keypoints"
        # 文件夹不存在就创建，存在就不报错
        os.makedirs(self.base_path, exist_ok=True)

        self.label_repo = LabelRepository(self.project_root)
        labels_order = self.label_repo.get_labels_order() # 获取标签顺序列表
        self.gestures = {ord(str(i + 1)): label  #ord() = 转键盘按键码
                         for i, label 
                         in enumerate(labels_order)}# 遍历列表时，同时拿到序号 + 元素

        # 运行时状态
        # 当前采集类别标签，标签只能是 字符串 或 空值
        self.current_label: str | None = None 
        # 该类别已采集的样本数量，初始为 0
        self.sample_count: int = 0
        # 默认停止采集，不工作
        self.collecting: bool = False 

        print("Key -> gesture mapping:")
        for i, label in enumerate(labels_order):
            print(f"  {i + 1} -> {label}")

        print("1~8 选择类别")
        print("c 开始采集")
        print("v 暂停采集")
        print("ESC 退出")

    def _class_dir(self, label: str) -> str:
        class_dir = os.path.join(self.base_path, label)
        os.makedirs(class_dir, exist_ok=True)
        return class_dir

    def _load_count(self, label: str) -> None:
        class_dir = self._class_dir(label)
        count_file = os.path.join(class_dir, "count.txt")
        if os.path.exists(count_file):
            with open(count_file, "r") as f:
                self.sample_count = int(f.read().strip())
        else:
            self.sample_count = 0

    def _save_count(self) -> None:
        if not self.current_label:
            return
        class_dir = self._class_dir(self.current_label)
        count_file = os.path.join(class_dir, "count.txt")
        with open(count_file, "w") as f:
            f.write(str(self.sample_count))

    def _save_sample(self, keypoints: np.ndarray) -> None:
        if not self.current_label:
            return
        class_dir = self._class_dir(self.current_label)
        file_path = os.path.join(class_dir, f"{self.sample_count}.npy")
        np.save(file_path, keypoints)
        self.sample_count += 1

    def _handle_key(self, key: int) -> None:
        # 选择类别
        if key in self.gestures:
            self.current_label = self.gestures[key]
            self._load_count(self.current_label)
            print(f"当前类别: {self.current_label}")
            print(f"已存在样本: {self.sample_count}")
            self.collecting = False

        # 开始采集
        if key == ord("c") and self.current_label:
            if self.sample_count >= SAMPLES_PER_CLASS:
                print("该类别已达 500，无需继续")
            else:
                self.collecting = True
                print("开始采集")

        # 暂停采集
        if key == ord("v") and self.current_label:
            self.collecting = False
            self._save_count()
            print("已暂停并保存进度")

    def run(self) -> None:
        try:
            while True:
                # 从摄像头捕获一帧图像
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 处理帧：执行手部检测与关键点提取
                frame, keypoints = self.tracker.process_frame(frame)
                key = cv2.waitKey(1) & 0xFF

                # 按键处理
                self._handle_key(key)

                # 采集逻辑
                if self.collecting and keypoints is not None:
                    if self.sample_count < SAMPLES_PER_CLASS:
                        self._save_sample(keypoints)
                        cv2.putText(
                            frame,
                            f"{self.current_label}: {self.sample_count}/{SAMPLES_PER_CLASS}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        self.collecting = False
                        print("该手势类别采集完成")

                cv2.imshow("数据收集", frame)

                # ESC 退出
                if key == 27:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def run_camera_test():
    collector = HandDataCollector()
    collector.run()

if __name__ == "__main__":
    run_camera_test()
