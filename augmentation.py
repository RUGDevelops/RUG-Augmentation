import os
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras._tf_keras.keras.preprocessing.image import save_img

# Create global variable saved_counter intialized to 0
saved_counter = 0
# user_name = "Rene Jausovec"


def save_image(image, user_name):
    """Save image to dataset/augmented folder"""
    global saved_counter

    filename = f"{user_name.replace(' ', '_')}{saved_counter}.jpg"
    if not os.path.exists(f'dataset/{user_name}'):
        os.makedirs(f'dataset/{user_name}')
    if os.path.exists(f'dataset/{user_name}/{filename}'):
        os.remove(f'dataset/{user_name}/{filename}')

    save_img(f'dataset/{user_name}/{filename}', image, )

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

    def __init__(self, images, user_name, rotation=15, horizontal_shift=0.15, vertical_shift=-0.15, zoom=30, brightness=50):
        self.dataset = images
        self.rotation = rotation
        self.horizontal_shift = horizontal_shift
        self.vertical_shift = vertical_shift
        self.zoom = zoom
        self.brightness = brightness
        self.user_name = user_name
        pass

    def augment(self):
        """Augment the images in the dataset"""
        for image in self.dataset:
            # Resize all images to 160x160 for best training results
            # image = cv2.resize(image, (160, 160))
            image = np.array(image, dtype=np.uint8)
            # Rotate right and left
            self.rotate_left_and_right(image)
            # Horizontal shift and vertical shift
            # self.shift_horizontaly_and_verticaly(image)
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

        save_image(right_rotated_image, self.user_name)
        save_image(left_rotated_image, self.user_name)
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
        save_image(zoomed_image, self.user_name)
        pass

    def flip(self, image):
        """Flip image horizontally"""
        # Flip code 0 = horizontal flip
        # Flip code 1 = vertical flip
        image = cv2.flip(image, flipCode=1)
        save_image(image, self.user_name)
        pass

    def adjust_brightness(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - self.brightness
        v[v > lim] = 255
        v[v <= lim] += self.brightness

        final_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        save_image(image, self.user_name)
        pass
