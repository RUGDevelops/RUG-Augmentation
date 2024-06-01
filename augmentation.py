import os
import cv2
import numpy as np
import numba

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras  # Keras is the high-level API of TensorFlow 2
from keras._tf_keras.keras.preprocessing.image import save_img

# Create global variable saved_counter intialized to 0
saved_counter = 0


def save_image(image):
    """Save image to dataset/augmented folder"""
    global saved_counter

    filename = f"augmented{saved_counter}.jpg"
    if not os.path.exists('augmented_dataset'):
        os.makedirs('augmented_dataset')
    if os.path.exists(f'augmented_dataset/{filename}'):
        os.remove(f'augmented_dataset/{filename}')

    save_img(f'augmented_dataset/{filename}', image)

    saved_counter += 1
    pass


class ImageDataAugmentation:
    """
    Rotation - Applies both left and right.
    Horizontal shift - Put negative number for left and positive for right shift.
    Vertical shift - Put negative number for up and positive for down shift.
    Zoom - Zoom in the image.
    Brightness - Negative number for decrease and positive for increase.
    """
    dataset = None
    rotation = 0
    horizontal_shift = 0.1
    vertical_shift = -0.0
    zoom = 0
    brightness = 0

    def __init__(self, images, rotation=15, horizontal_shift=0.15, vertical_shift=-0.15, zoom=30, brightness=75):
        self.dataset = images
        self.rotation = rotation
        self.horizontal_shift = horizontal_shift
        self.vertical_shift = vertical_shift
        self.zoom = zoom
        self.brightness = brightness
        pass

    def augment(self):
        """Augment the images in the dataset"""
        for image in self.dataset:
            # Rotate right and left
            self.rotate_left_and_right(image)
            # Horizontal shift and vertical shift
            self.shift_horizontaly_and_verticaly(image)
            # Zoom
            self.zoom_image(image)
            # Flip
            self.flip(image)
            # Decrease brightness
            self.adjust_brightness(image)
        pass

    def rotate_left_and_right(self, image):
        """Rotate image to right for right_rotation degrees and save to dataset/augmented folder"""
        height, width = image.shape[:2]
        # center coordinates of the image
        centerX, centerY = (width // 2, height // 2)
        # 1.0 scales the image to the same dimensions as original
        right_rotation_matrix = cv2.getRotationMatrix2D((centerX, centerY), self.rotation, 1.0)
        left_rotation_matrix = cv2.getRotationMatrix2D((centerX, centerY), -self.rotation, 1.0)

        right_rotated_image = cv2.warpAffine(image, right_rotation_matrix, (width, height))
        left_rotated_image = cv2.warpAffine(image, left_rotation_matrix, (width, height))

        save_image(right_rotated_image)
        save_image(left_rotated_image)
        pass

    def shift_horizontaly_and_verticaly(self, image):
        """Shift image horizontally and vertically for shift percentage"""
        height, width = image.shape[:2]

        translation_matrix = np.float32([[1, 0, self.horizontal_shift * width], [0, 1, self.vertical_shift * height]])
        translated_image = cv2.warpAffine(image, translation_matrix, (width, height))

        save_image(translated_image)
        pass

    def zoom_image(self, image):
        """Zoom image for zoom percentage"""
        height, width = image.shape[:2]
        x1 = self.zoom
        x2 = width - x1 * 2
        y1 = self.zoom
        y2 = height - y1 * 2
        cropped = image[y1:y2, x1:x2]
        zoomed_image = cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        save_image(zoomed_image)
        pass

    def flip(self, image):
        """Flip image horizontally"""
        # Flip code 0 = horizontal flip
        # Flip code 1 = vertical flip
        image = cv2.flip(image, flipCode=1)
        save_image(image)
        pass

    def adjust_brightness(self, image):
        """Adjust brightness of the image by brightness percentage"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - self.brightness
        v[v > lim] = 255
        v[v <= lim] += self.brightness

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        save_image(image)
        pass