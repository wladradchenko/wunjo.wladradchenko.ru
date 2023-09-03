import cv2


def save_image_cv2(save_file, target_frame):
    cv2.imwrite(save_file, target_frame)
    return save_file


def read_image_cv2(image_path):
    return cv2.imread(image_path)
