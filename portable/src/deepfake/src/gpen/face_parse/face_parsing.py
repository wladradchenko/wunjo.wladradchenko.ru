import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "deepfake"))
from src.gpen.face_parse.parse_model import ParseNet
sys.path.pop(0)


class FaceParse(object):
    def __init__(self, model_path, device='cuda', mask_map = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]):
        self.mfile = model_path
        self.size = 512
        self.device = device

        '''
        0: 'background' 1: 'skin'   2: 'nose'
        3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
        6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
        9: 'r_ear'  10: 'mouth' 11: 'u_lip'
        12: 'l_lip' 13: 'hair'  14: 'hat'
        15: 'ear_r' 16: 'neck_l'    17: 'neck'
        18: 'cloth'
        '''
        self.MASK_COLORMAP = mask_map

        self.load_model()

    def load_model(self):
        self.faceparse = ParseNet(self.size, self.size, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
        self.faceparse.load_state_dict(torch.load(self.mfile, map_location=torch.device('cpu')))
        self.faceparse.to(self.device)
        self.faceparse.eval()

    def process(self, im, masks=[0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]):
        im = cv2.resize(im, (self.size, self.size))
        imt = self.img2tensor(im)
        with torch.no_grad():
            pred_mask, sr_img_tensor = self.faceparse(imt)  # (1, 19, 512, 512)
        mask = self.tenor2mask(pred_mask, masks)

        return mask

    def process_tensor(self, imt):
        imt = F.interpolate(imt.flip(1)*2-1, (self.size, self.size))
        pred_mask, sr_img_tensor = self.faceparse(imt)

        mask = pred_mask.argmax(dim=1)
        for idx, color in enumerate(self.MASK_COLORMAP):
            mask = torch.where(mask==idx, color, mask)
        #mask = mask.repeat(3, 1, 1).unsqueeze(0) #.cpu().float().numpy()
        mask = mask.unsqueeze(0)

        return mask

    def img2tensor(self, img):
        img = img[..., ::-1] # BGR to RGB
        img = img / 255. * 2 - 1
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return img_tensor.float()

    def tenor2mask(self, tensor, masks):
        if len(tensor.shape) < 4:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] > 1:
            tensor = tensor.argmax(dim=1)

        tensor = tensor.squeeze(1).data.cpu().numpy()
        color_maps = []
        for t in tensor:
            tmp_img = np.zeros(tensor.shape[1:])
            for idx, color in enumerate(masks):
                tmp_img[t == idx] = color
            color_maps.append(tmp_img.astype(np.uint8))
        return color_maps