import torch
import torch.nn as nn
import torch.nn.functional as F


class gen_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize, stride=1, rate=1, activation=nn.ELU()):
        """Define conv for generator

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            rate: Rate for or dilated conv.
            activation: Activation function after convolution.
        """
        p = int(rate*(ksize-1)/2)
        super(gen_conv, self).__init__(in_channels=cin, out_channels=cout, kernel_size=ksize, stride=stride, padding=p, dilation=rate, groups=1, bias=True)
        self.activation = activation

    def forward(self, x):
        x = super(gen_conv, self).forward(x)
        if self.out_channels == 3 or self.activation is None:
            return x
        x, y = torch.split(x, int(self.out_channels/2), dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x

class gen_deconv(gen_conv):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv, self).__init__(cin, cout, ksize=3)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv, self).forward(x)
        return x
