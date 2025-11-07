import os
import torch
from argparse import Namespace
from .inpaint_g import BaseConvGenerator


class InpaintModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        args = self.load_model_param()
        args.model_path = model_path

        # Use just CPU
        self.gpu = self.use_gpu()
        self.FloatTensor = torch.cuda.FloatTensor if self.gpu else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.gpu else torch.ByteTensor

        self.netG, self.netD = self.load(args)

    def forward(self, data):
        """
        Forward with inference model
        :param data:
        :return:
        """
        inputs, real_image, mask = self.preprocess_input(data)
        with torch.no_grad():
            coarse_image, fake_image = self.generate_fake(inputs, mask)
            composed_image = fake_image*mask + inputs*(1-mask)
        return composed_image, inputs

    def generate_fake(self, inputs, mask):
        coarse_image, fake_image = self.netG(inputs, mask)
        return coarse_image, fake_image

    def preprocess_input(self, data):
        if self.gpu:
            data['mask'] = data['mask'].cuda()
        mask = data['mask']
        # move to GPU and change data types
        if self.gpu:
            data['image'] = data['image'].cuda()
        inputs = data['image']*(1-mask)
        data['inputs'] = inputs
        return inputs, data['image'], mask

    def load(self, args):
        netG = self.create_retouch(args)
        device = "cuda" if self.gpu else "cpu"
        weights = torch.load(args.model_path, map_location=device, weights_only=True)
        new_dict = {}
        for k, v in weights.items():
            if k.startswith("module."):
                k = k.replace("module.", "")
            new_dict[k] = v
        netG.load_state_dict(new_dict, strict=False)
        netG.to(device)  # Move the network to the same device
        return netG, None

    @staticmethod
    def load_model_param():
        return Namespace(
            init_type='xavier', init_variance=0.02
        )

    @staticmethod
    def create_retouch(args):
        net = BaseConvGenerator()
        net.print_network()
        if args.init_type is not None:
            net.init_weights(args.init_type, args.init_variance)
        return net

    @staticmethod
    def use_gpu():
        return True if torch.cuda.is_available() and 'cpu' not in os.environ.get('WUNJO_TORCH_DEVICE', 'cpu') else False