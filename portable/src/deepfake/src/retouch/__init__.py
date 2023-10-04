import cv2
import torch
import numpy as np
from PIL import Image

from .inpaint_model import InpaintModel


def process_retouch(img, mask, model):
    img = img.convert("RGB")
    img_raw = np.array(img)
    w_raw, h_raw = img.size
    h_t, w_t = h_raw // 8 * 8, w_raw // 8 * 8

    img = img.resize((w_t, h_t))
    img = np.array(img).transpose((2, 0, 1))

    mask_raw = np.array(mask)[..., None] > 0
    mask = mask.resize((w_t, h_t))
    mask = np.array(mask)
    mask = (torch.Tensor(mask) > 0).float()

    img = (torch.Tensor(img)).float()
    img = (img / 255 - 0.5) / 0.5
    img = img[None]
    mask = mask[None, None]

    with torch.no_grad():
        generated, _ = model({'image': img, 'mask': mask})
    generated = torch.clamp(generated, -1, 1)
    generated = (generated + 1) / 2 * 255
    generated = generated.cpu().numpy().astype(np.uint8)
    generated = generated[0].transpose((1, 2, 0))

    # Resize mask_raw to match the shape of generated
    mask_raw_resized = cv2.resize(mask_raw.astype(np.uint8), (generated.shape[1], generated.shape[0]))

    # If the shapes don't match, resize the smaller to the shape of the larger
    if generated.shape != img_raw.shape:
        if generated.shape[1] < img_raw.shape[1]:
            img_raw = cv2.resize(img_raw, (generated.shape[1], generated.shape[0]))
        else:
            generated = cv2.resize(generated, (img_raw.shape[1], img_raw.shape[0]))

    # Ensure mask_raw_resized has the same shape as img_raw and generated
    if mask_raw_resized.shape[:2] != img_raw.shape[:2]:
        mask_raw_resized = cv2.resize(mask_raw_resized, (img_raw.shape[1], img_raw.shape[0]))

    # Replicate single-channel mask to have 3 channels
    mask_raw_3ch = np.repeat(mask_raw_resized[..., None], 3, axis=2)

    merge = generated * mask_raw_3ch + img_raw * (1 - mask_raw_3ch)
    merge = merge.astype(np.uint8)

    return Image.fromarray(merge).resize((w_raw, h_raw))


def convert_cv2_to_pil(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def pil_to_cv2(pil_image):
    # Convert the PIL image to a numpy array
    cv2_image = np.array(pil_image)
    # Convert RGB to BGR
    cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image