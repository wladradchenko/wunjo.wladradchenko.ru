import torch.optim
import torch.nn as nn
import torch.nn.init as init


def load_model(model_path):
    model = Model(16000, num_bit=32, n_fft=1000, hop_length=400, num_layers=8)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model_ckpt = checkpoint
    model.load_state_dict(model_ckpt, strict=True)
    model.eval()
    return model


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(in_channel + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(in_channel + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(in_channel + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(in_channel + 4 * 32, out_channel, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class INV_block(nn.Module):
    def __init__(self, channel=2, subnet_constructor=ResidualDenseBlock_out, clamp=2.0):
        super().__init__()
        self.clamp = clamp

        # ρ
        self.r = subnet_constructor(channel, channel)
        # η
        self.y = subnet_constructor(channel, channel)
        # φ
        self.f = subnet_constructor(channel, channel)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x1, x2, rev=False):
        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return y1, y2


class Hinet(torch.nn.Module):

    def __init__(self, in_channel=2, num_layers=16):
        super(Hinet, self).__init__()
        self.inv_blocks = torch.nn.ModuleList([INV_block(in_channel) for _ in range(num_layers)])

    def forward(self, x1, x2, rev=False):
        # x1:cover
        # x2:secret
        if not rev:
            for inv_block in self.inv_blocks:
                x1, x2 = inv_block(x1, x2)
        else:
            for inv_block in reversed(self.inv_blocks):
                x1, x2 = inv_block(x1, x2, rev=True)
        return x1, x2


class Model(nn.Module):
    def __init__(self, num_point, num_bit, n_fft, hop_length, num_layers):
        super(Model, self).__init__()
        self.hinet = Hinet(num_layers=num_layers)
        self.watermark_fc = torch.nn.Linear(num_bit, num_point)
        self.watermark_fc_back = torch.nn.Linear(num_point, num_bit)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        # torch: return_complex=False is deprecDeprecated since version 2.0: return_complex=False is deprecated,
        # instead use return_complex=True Note that calling torch.view_as_real() on the output will recover the deprecated output format.
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        tmp = torch.view_as_real(tmp)
        # [1, 501, 41, 2]
        return tmp

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        # torch: return_complex=False is deprecDeprecated since version 2.0: return_complex=False is deprecated,
        # instead use return_complex=True Note that calling torch.view_as_real() on the output will recover the deprecated output format.
        return torch.istft(torch.view_as_complex(signal_wmd_fft), n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)

    def encode(self, signal, message):
        signal_fft = self.stft(signal)
        # (batch,freq_bins,time_frames,2)

        message_expand = self.watermark_fc(message)
        message_fft = self.stft(message_expand)

        signal_wmd_fft, msg_remain = self.enc_dec(signal_fft, message_fft, rev=False)
        # (batch,freq_bins,time_frames,2)
        signal_wmd = self.istft(signal_wmd_fft)
        return signal_wmd

    def decode(self, signal):
        signal_fft = self.stft(signal)
        signature_fft = signal_fft
        _, message_restored_fft = self.enc_dec(signal_fft, signature_fft, rev=True)
        message_restored_expanded = self.istft(message_restored_fft)
        message_restored_float = self.watermark_fc_back(message_restored_expanded).clamp(-1, 1)
        return message_restored_float

    def enc_dec(self, signal, signature, rev):
        signal = signal.permute(0, 3, 2, 1)
        # [4, 2, 41, 501]
        signature = signature.permute(0, 3, 2, 1)
        signal2, signature2 = self.hinet(signal, signature, rev)
        return signal2.permute(0, 3, 2, 1), signature2.permute(0, 3, 2, 1)