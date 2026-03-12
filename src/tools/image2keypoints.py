import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def normalize_keypoints(keypoints):

    keypoints = keypoints.reshape(21,3)

    wrist = keypoints[0]
    keypoints = keypoints - wrist

    keypoints = keypoints[:,:2]

    max_value = np.max(np.abs(keypoints))

    if max_value != 0:
        keypoints = keypoints / max_value

    return keypoints.flatten()


def convert_images():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    image_path = PROJECT_ROOT / "data" / "images_raw"
    save_path = PROJECT_ROOT / "data" / "keypoints"

    for class_dir in image_path.iterdir():

        if not class_dir.is_dir():
            continue

        label = class_dir.name

        save_dir = save_path / label
        save_dir.mkdir(parents=True, exist_ok=True)

        count = len(list(save_dir.glob("*.npy")))

        for img_file in class_dir.glob("*"):

            img = cv2.imread(str(img_file))
            # 如果图片读取失败就跳过
            if img is None:
                print("跳过无法读取的文件:", img_file)
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]

                points = []

                for lm in hand.landmark:
                    points.append([lm.x, lm.y, lm.z])

                points = np.array(points)

                keypoints = normalize_keypoints(points)

                np.save(save_dir / f"{count}.npy", keypoints)

                count += 1

        print(label, "转换完成")


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