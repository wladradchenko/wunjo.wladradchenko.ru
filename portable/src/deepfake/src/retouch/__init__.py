import cv2
import torch
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from .inpaint_model import InpaintModel

# Retouch new approach
from .model.modules.flow_comp_raft import RAFT_bi
from .model.recurrent_flow_completion import RecurrentFlowCompleteNet
from .model.propainter import InpaintGenerator


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


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


def convert_colored_mask_thickness_cv2(mask):
    # Reduce the dimensions if necessary
    if mask.ndim == 4:
        mask = mask[0, 0]
    # Convert boolean values to 0 or 255
    mask_to_save = (mask * 255).astype(np.uint8)

    # Dilation to increase line thickness
    kernel_size = 10  # Thickness line weight
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_to_save = cv2.dilate(mask_to_save, kernel, iterations=1)

    # Convert grayscale to BGR
    colored_mask = cv2.cvtColor(mask_to_save, cv2.COLOR_GRAY2BGR)
    # Replace white with random color
    random_color = (0, 0, 255)
    colored_mask[mask_to_save == 255] = random_color
    # Create alpha channel: 0 for black, 255 for colored regions
    alpha_channel = np.ones(mask_to_save.shape, dtype=mask_to_save.dtype) * 255
    alpha_channel[mask_to_save == 0] = 0
    # Merge RGB and alpha into a 4-channel image (BGRA)
    bgra_mask = cv2.merge((colored_mask[:, :, 0], colored_mask[:, :, 1], colored_mask[:, :, 2], alpha_channel))
    # Convert to PIL
    # Convert the BGRA mask to RGBA
    rgba_mask = bgra_mask[..., [2, 1, 0, 3]]
    # Convert the RGBA mask to a Pillow image
    pil_image = Image.fromarray(rgba_mask, 'RGBA')
    # Or grayscale
    grayscale_image = pil_image.convert("L")
    return grayscale_image


def upscale_retouch_frame(mask, frame, original_frame, width, height):
    color = (0, 0, 0, 0)
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Reduce the dimensions if necessary
    if mask.ndim == 4:
        mask = mask[0, 0]

    # Dilation to increase line thickness
    # Convert boolean mask to uint8
    mask_img_uint8 = (mask * 255).astype(np.uint8)
    kernel_size = 10  # Thickness line weight
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask_uint8 = cv2.dilate(mask_img_uint8, kernel, iterations=1)
    # Convert dilated uint8 mask back to boolean
    mask = dilated_mask_uint8.astype(bool)

    mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert mask's False values to 0 and other values to 255
    mask_to_save = (mask * 255).astype(np.uint8)

    # Dilation to increase line thickness
    kernel_size = 10  # Thickness line weight
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_to_save = cv2.dilate(mask_to_save, kernel, iterations=1)

    # Create an empty colored image with the same dimensions as the mask
    colored_img = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

    # Set the RGB channels to the specified color and alpha channel to 255 for non-transparent regions
    colored_img[mask_to_save == 0, :3] = color[:3]
    colored_img[mask_to_save == 0, 3] = color[3]

    # Convert frame to RGBA format
    frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)

    # Where the mask is True, replace with the frame values
    colored_img[mask_to_save == 255] = frame_rgba[mask_to_save == 255]

    # Convert to PIL Image and then back to OpenCV format
    pil_image = Image.fromarray(colored_img, 'RGBA')
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)

    # For non-transparent regions in cv2_image, replace the corresponding regions in original_frame with those from frame
    alpha_channel = cv2_image[..., 3]
    original_frame[alpha_channel != 0] = cv2_image[alpha_channel != 0, :3]

    return original_frame