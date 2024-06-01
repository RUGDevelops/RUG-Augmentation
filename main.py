import os
import numpy as np
import cv2

from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from augmentation import ImageDataAugmentation  # Custom augmentation class
from keras._tf_keras.keras.preprocessing.image import save_img


def get_dataset_from_video(video):
    """Get dataset from video file"""
    video = cv2.VideoCapture(video)
    dataset = []
    saved_counter = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dataset.append(frame)

        filename = f"frame{saved_counter}.jpg"
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        if os.path.exists(f'dataset/{filename}'):
            os.remove(f'dataset/{filename}')

        save_img(f'dataset/{filename}', frame)

        saved_counter += 1

    video.release()

    dataset = np.array(dataset)
    return dataset


if __name__ == "__main__":
    # Load data from dataset folder and save to array
    dataset = get_dataset_from_video("face_recognition.mp4")

    augmentor = ImageDataAugmentation(dataset)

    augmentor.augment()

    pass