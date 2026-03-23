import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.utils.normalizer import normalize_keypoints


_THREAD_LOCAL = threading.local()


def _get_hands():
    hands = getattr(_THREAD_LOCAL, "hands", None)
    if hands is None:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
        )
        _THREAD_LOCAL.hands = hands
    return hands


def _get_max_workers(task_count: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(task_count, cpu_count, 8))


def _convert_single_image(task):
    label, img_file, save_dir, index = task

    img = cv2.imread(str(img_file))
    if img is None:
        return label, "unreadable", img_file

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = _get_hands().process(rgb)

    if not results.multi_hand_landmarks:
        return label, "no_hand", img_file

    hand = results.multi_hand_landmarks[0]
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
    keypoints = normalize_keypoints(points)

    np.save(save_dir / f"{index}.npy", keypoints)
    return label, "saved", img_file


def convert_images():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    image_path = PROJECT_ROOT / "data" / "images_raw"
    save_path = PROJECT_ROOT / "data" / "keypoints"

    tasks = []
    stats = defaultdict(lambda: {"saved": 0, "unreadable": 0, "no_hand": 0})

    for class_dir in sorted(image_path.iterdir()):

        if not class_dir.is_dir():
            continue

        label = class_dir.name

        save_dir = save_path / label
        save_dir.mkdir(parents=True, exist_ok=True)

        count = len(list(save_dir.glob("*.npy")))

        image_files = sorted(img_file for img_file in class_dir.iterdir() if img_file.is_file())
        for offset, img_file in enumerate(image_files):
            tasks.append((label, img_file, save_dir, count + offset))

    if not tasks:
        print("未找到可转换的图片")
        return

    max_workers = _get_max_workers(len(tasks))
    print(f"开始并发转换，共 {len(tasks)} 张图片，线程数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_convert_single_image, task) for task in tasks]
        for future in as_completed(futures):
            label, status, img_file = future.result()
            stats[label][status] += 1
            if status == "unreadable":
                print("跳过无法读取的文件:", img_file)

    for label in sorted(stats):
        saved = stats[label]["saved"]
        unreadable = stats[label]["unreadable"]
        no_hand = stats[label]["no_hand"]
        print(f"{label} 转换完成: 保存 {saved}, 无法读取 {unreadable}, 未检测到手 {no_hand}")


if __name__ == "__main__":
    convert_images()

# ========检查版本环境========
# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# import jax
#
# print("numpy:", np.__version__)
# print("opencv:", cv2.__version__)
# print("mediapipe:", mp.__version__)
# print("tensorflow:", tf.__version__)
# print("jax:", jax.__version__)