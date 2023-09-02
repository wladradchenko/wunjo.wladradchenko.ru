import cv2


def save_image(save_file, target_frame):
    cv2.imwrite(save_file, target_frame)
    return save_file
