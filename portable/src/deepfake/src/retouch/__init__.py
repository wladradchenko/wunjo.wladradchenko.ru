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


class VideoRemoveObjectProcessor:
    def __init__(self, device, model_raft_things_path, model_recurrent_flow_path, model_pro_painter_path):
        self.device = device
        self.fix_raft = self.load_fix_raft(model_raft_things_path, device)
        self.fix_flow_complete = self.load_fix_flow_complete(model_recurrent_flow_path, device)
        self.fix_flow_complete.eval()
        self.model_inpaint = self.load_inpaint(model_pro_painter_path, device)
        self.model_inpaint.eval()


    @staticmethod
    def load_fix_raft(model_raft_things_path, device):
        return RAFT_bi(model_raft_things_path, device)

    @staticmethod
    def load_fix_flow_complete(model_recurrent_flow_path, device):
        fix_flow_complete = RecurrentFlowCompleteNet(model_recurrent_flow_path)
        for p in fix_flow_complete.parameters():
            p.requires_grad = False
        return fix_flow_complete.to(device)

    @staticmethod
    def load_inpaint(model_pro_painter_path, device):
        return InpaintGenerator(model_path=model_pro_painter_path).to(device)


    @staticmethod
    def to_tensors():
        return transforms.Compose([Stack(), ToTorchFormatTensor()])

    @staticmethod
    def binary_mask(mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        return mask

    @staticmethod
    def read_retouch_mask(masks, width, height, flow_mask_dilates=8, mask_dilates=5):
        masks_img = masks
        masks_dilated = []
        flow_masks = []
        for mask_img in masks_img:
            if mask_img.ndim == 4:
                mask_img = mask_img[0, 0]
            # Convert to numpy array
            mask_img = np.array(mask_img)

            # Dilation to increase line thickness
            # Convert boolean mask to uint8
            mask_img_uint8 = (mask_img * 255).astype(np.uint8)
            kernel_size = 10  # Thickness line weight
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_mask_uint8 = cv2.dilate(mask_img_uint8, kernel, iterations=1)
            # Convert dilated uint8 mask back to boolean
            mask_img = dilated_mask_uint8.astype(bool)

            # Resize the mask_img using resize_frames method
            resized_masks, _, _ = VideoRemoveObjectProcessor.resize_frames([Image.fromarray(mask_img.astype(np.uint8))], size=(width, height))
            mask_img = np.array(resized_masks[0])
            # Set values other than False to 1
            mask_img[mask_img != False] = 1
            # Dilate the mask for flow_mask
            if flow_mask_dilates > 0:
                flow_mask_img = binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
            else:
                flow_mask_img = VideoRemoveObjectProcessor.binary_mask(mask_img).astype(np.uint8)
            flow_masks.append(Image.fromarray(flow_mask_img * 255))
            # Dilate the mask for masks_dilated
            if mask_dilates > 0:
                mask_img = binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
            else:
                mask_img = VideoRemoveObjectProcessor.binary_mask(mask_img).astype(np.uint8)
            masks_dilated.append(Image.fromarray(mask_img * 255))
        if len(masks_img) == 1:
            flow_masks = flow_masks * len(masks)
            masks_dilated = masks_dilated * len(masks)
        return flow_masks, masks_dilated

    def process_frames_with_retouch_mask(self, frames, masks_dilated, flow_masks, height, width):
        # Ensure the input frames and masks are of the same length
        assert len(frames) == len(masks_dilated) == len(flow_masks), "Mismatch in number of frames and masks."
        masked_frame_for_save = []

        for i in range(len(frames)):
            mask_ = np.expand_dims(np.array(masks_dilated[i]), 2).repeat(3, axis=2) / 255.
            img = np.array(frames[i])
            green = np.zeros([height, width, 3])
            green[:, :, 1] = 255
            alpha = 0.6
            fuse_img = (1 - alpha) * img + alpha * green
            fuse_img = mask_ * fuse_img + (1 - mask_) * img
            masked_frame_for_save.append(fuse_img.astype(np.uint8))

        frames_inp = [np.array(f).astype(np.uint8) for f in frames]
        frames = VideoRemoveObjectProcessor.to_tensors()(frames).unsqueeze(0) * 2 - 1
        flow_masks = VideoRemoveObjectProcessor.to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = VideoRemoveObjectProcessor.to_tensors()(masks_dilated).unsqueeze(0)
        frames, flow_masks, masks_dilated = frames.to(self.device), flow_masks.to(self.device), masks_dilated.to(self.device)

        return frames, flow_masks, masks_dilated, masked_frame_for_save, frames_inp

    @staticmethod
    def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    @staticmethod
    def resize_frames(frames, size=None):
        if size is not None:
            out_size = size
            process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
            frames = [f.resize(process_size) for f in frames]
        else:
            out_size = frames[0].size
            process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
            if not out_size == process_size:
                frames = [f.resize(process_size) for f in frames]

        return frames, process_size, out_size

    def transfer_to_pil(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def process_video_with_mask(self, frames, masks_dilated, flow_masks, frames_inp, width, height, raft_iter=20, subvideo_length=80, neighbor_length=10, ref_stride=10, use_half=False):
        video_length = frames.size(1)

        with torch.no_grad():
            # ---- compute flow ----
            if frames.size(-1) <= 640:
                short_clip_len = 12
            elif frames.size(-1) <= 720:
                short_clip_len = 8
            elif frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            # use fp32 for RAFT
            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(frames[:, f:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = self.fix_raft(frames[:, f - 1:end_f], iters=raft_iter)

                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()

                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(frames, iters=raft_iter)
                torch.cuda.empty_cache()

            if use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                fix_flow_complete = self.fix_flow_complete.half()
                model = self.model_inpaint.half()
            else:
                fix_flow_complete = self.fix_flow_complete
                model = self.model_inpaint

            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + subvideo_length)
                    pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow((gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),flow_masks[:, s_f:e_f + 1])
                    pred_flows_bi_sub = fix_flow_complete.combine_flow((gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), pred_flows_bi_sub, flow_masks[:, s_f:e_f + 1])

                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()

                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                torch.cuda.empty_cache()

            # ---- image propagation ----
            masked_frames = frames * (1 - masks_dilated)
            subvideo_length_img_prop = min(100, subvideo_length)
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                    prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f], pred_flows_bi_sub, masks_dilated[:, s_f:e_f], 'nearest')
                    updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + prop_imgs_sub.view(b, t, 3, height, width) * masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, height, width)

                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()

                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, height, width) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, height, width)
                torch.cuda.empty_cache()

        ori_frames = frames_inp
        comp_frames = [None] * video_length

        neighbor_stride = neighbor_length // 2
        if video_length > subvideo_length:
            ref_num = subvideo_length // ref_stride
        else:
            ref_num = -1

        # ---- feature propagation + transformer ----
        for f in tqdm(range(0, video_length, neighbor_stride)):
            neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
            ref_ids = self.get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)

                # pred_img = selected_imgs # results of image propagation
                pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)

                pred_img = pred_img.view(-1, 3, height, width)

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)

            torch.cuda.empty_cache()

        return comp_frames


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