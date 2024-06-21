import os
import cv2
import math
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

from deepfake.src.east.model import EAST


def get_rotate_mat(theta):
    """
    Positive theta value means rotate clockwise
    :param theta: theta
    :return:
    """
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def resize_img(img):
    """
    Resize image to be divisible by 32
    :param img: image
    :return: resized image
    """
    w, h = img.size
    resize_w = w
    resize_h = h

    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w

    return img, ratio_h, ratio_w


def load_pil(img):
    """
    Convert PIL Image to torch.Tensor
    :param img: image
    :return: converted image
    """
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return t(img).unsqueeze(0)



def is_valid_poly(res, score_shape, scale):
    """
    Check if the poly in image scope
    :param res: restored poly in original image
    :param score_shape: score map shape
    :param scale: feature map -> image
    :return: True if valid
    """
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    """
    Restore polys from feature maps in given positions
    :param valid_pos: potential text positions <numpy.ndarray, (n,2)>
    :param valid_geo: geometry in valid_pos <numpy.ndarray, (5,n)>
    :param score_shape: shape of score map
    :param scale: image / feature map
    :return: restored polys <numpy.ndarray, (n,8)>, index
    """
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    """
    Get boxes from feature map
    :param score: score map from model <numpy.ndarray, (1,row,col)>
    :param geo: geo map from model <numpy.ndarray, (5,row,col)>
    :param score_thresh: threshold to segment score map
    :param nms_thresh: threshold in nms
    :return: inal polys <numpy.ndarray, (n,9)>
    """
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    """
    Refine boxes
    :param boxes: detected polys <numpy.ndarray, (n,9)>
    :param ratio_w: ratio of width
    :param ratio_h: ratio of height
    :return: refined boxes
    """
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def detect(img, model, device):
    """
    Detect text regions of img using model
    :param img: PIL Image
    :param model: detection model
    :param device: gpu if gpu is available
    :return: detected polys
    """
    img, ratio_h, ratio_w = resize_img(img)
    with torch.no_grad():
        score, geo = model(load_pil(img).to(device))
    boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
    """
    Plot boxes on image
    :param img: image
    :param boxes: boxes
    :return: new image
    """
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0, 255, 0))
    return img


def detect_dataset(model, device, test_img_path, submit_path):
    """
    Detection on whole dataset, save .txt results in submit_path
    :param model: detection model
    :param device: gpu if gpu is available
    :param test_img_path: dataset path
    :param submit_path:  submit result for evaluation
    :return:
    """
    img_files = os.listdir(test_img_path)
    img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])

    for i, img_file in enumerate(img_files):
        print('evaluating {} image'.format(i), end='\r')
        boxes = detect(Image.open(img_file), model, device)
        seq = []
        if boxes is not None:
            seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
        with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg', '.txt')), 'w') as f:
            f.writelines(seq)


def plot_filled_mask(img, boxes):
    """
    Plot filled boxes on a black image
    :param img: images
    :param boxes: boxes
    :return: mask
    """
    # Create a black image of the same size as the original
    mask = Image.new('RGB', img.size, (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    if boxes is not None:
        for box in boxes:
            # Draw filled rectangle (white filled box) for each detected text region
            draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], fill=(255, 255, 255))
    return mask


class SegmentText:
    def __init__(self, device, vgg16_path, east_path):
        self.device = device
        self.model = EAST(pretrained=True, model_path=vgg16_path).to(device)
        self.model.load_state_dict(torch.load(east_path, map_location=device))  # Add map_location for device
        self.model.eval()

    @staticmethod
    def crop(filled_mask, text_area: list = None):
        # If text_area is provided, extract the specified region and place it onto a black background
        if text_area is not None:
            x1, y1, x2, y2 = text_area
            # Create a new black image with the same size as the original filled_mask
            img = Image.new('L', filled_mask.size, color=0)
            # Paste the specified region onto the black image
            region = filled_mask.crop((x1, y1, x2, y2))
            img.paste(region, (x1, y1))
            return img
        else:
            return filled_mask  # Return the original image if no text_area is provided

    def detect_text(self, img_path, max_resolution=1280):
        img = Image.open(img_path)
        # Resize to improve quality
        w, h = img.size
        if max(w, h) > max_resolution:
            scale = max_resolution / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h))
        img, _, _ = resize_img(img)  # Get resized image
        # Detect text boxes
        boxes = detect(img, self.model, self.device)
        # Use the new function to draw filled boxes
        filled_mask = plot_filled_mask(img, boxes)
        # Restore image
        return filled_mask.resize((w, h))

    @staticmethod
    def apply_mask_on_frame(mask, frame, color, width=None, height=None):
        # Assuming mask is a single-channel image where non-zero values indicate the mask
        if width is not None and height is not None:
            mask = Image.fromarray(mask.astype('uint8'))
            mask = mask.resize((width, height))

        # Convert mask's values to 255 where mask is non-zero (True)
        mask_to_save = mask.point(lambda p: 0 if p > 0 else 255)
        mask_to_save = mask_to_save.convert('L')  # Ensure it's L mode for the mask

        # Create an image for the colored mask with the same dimensions as the mask
        # If the color is transparent (we'll assume an RGBA tuple where A=0 means fully transparent)
        if color[3] == 0:
            colored_mask = Image.new('RGBA', mask.size, (0, 0, 0, 0))
        else:
            colored_mask = Image.new('RGBA', mask.size, color)

        # Composite the colored mask with the frame using the mask
        frame_rgba = frame.convert('RGBA')
        result_img = Image.composite(colored_mask, frame_rgba, mask_to_save)

        return result_img

    @staticmethod
    def hex_to_rgba(hex_color):
        hex_color = hex_color.lstrip('#')
        # If the color is transparent, return (0, 0, 0, 0) for RGBA
        if hex_color.lower() == 'transparent':
            return (0, 0, 0, 0)

        # Convert hex to RGB
        lv = len(hex_color)
        return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)) + (255,)