import os
import uuid
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import soundfile
import librosa


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
        self.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5

    @staticmethod
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


class Hinet(nn.Module):

    def __init__(self, in_channel=2, num_layers=16):
        super(Hinet, self).__init__()
        self.inv_blocks = nn.ModuleList([INV_block(in_channel) for _ in range(num_layers)])

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
        self.digital_signature_fc = torch.nn.Linear(num_bit, num_point)
        self.digital_signature_fc_back = torch.nn.Linear(num_point, num_bit)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def stft(self, data):
        window = torch.hann_window(self.n_fft).to(data.device)
        tmp = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)  # Complex output
        real = tmp.real.unsqueeze(-1)  # Adds the 4th dimension
        imag = tmp.imag.unsqueeze(-1)
        return torch.cat([real, imag], dim=-1)  # Returns [B, F, T, 2]

    def istft(self, signal_wmd_fft):
        window = torch.hann_window(self.n_fft).to(signal_wmd_fft.device)
        complex_form = signal_wmd_fft[..., 0] + 1j * signal_wmd_fft[..., 1]  # Recreate the complex numbers from two last dimensions
        return torch.istft(complex_form, n_fft=self.n_fft, hop_length=self.hop_length, window=window)


    def encode(self, signal, message):
        signal_fft = self.stft(signal)

        message_expand = self.digital_signature_fc(message)
        message_fft = self.stft(message_expand)

        signal_wmd_fft, msg_remain = self.enc_dec(signal_fft, message_fft, rev=False)
        # (batch,freq_bins,time_frames,2)
        signal_wmd = self.istft(signal_wmd_fft)
        return signal_wmd

    def decode(self, signal):
        signal_fft = self.stft(signal)
        digital_signature_fft = signal_fft
        _, message_restored_fft = self.enc_dec(signal_fft, digital_signature_fft, rev=True)
        message_restored_expanded = self.istft(message_restored_fft)
        message_restored_float = self.digital_signature_fc_back(message_restored_expanded).clamp(-1, 1)
        return message_restored_float

    def enc_dec(self, signal, digital, rev):
        signal = signal.permute(0, 3, 2, 1)
        digital_signature = digital.permute(0, 3, 2, 1)
        signal2, digital_signature2 = self.hinet(signal, digital_signature, rev)
        return signal2.permute(0, 3, 2, 1), digital_signature2.permute(0, 3, 2, 1)


class DigitalSignature:
    def __init__(self, model_path, device="cpu"):
        """
        Initialization
        :param model_path: model path
        :param device: device
        """
        self.sample_rate = 16000
        self.payload = [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]  # ai
        self.model = self.load_model(model_path).to(device)

    @staticmethod
    def load_model(model_path):
        """
        Load model
        :param model_path: model path
        :return:  loaded model
        """
        model = Model(16000, num_bit=32, n_fft=1000, hop_length=400, num_layers=8)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_ckpt = checkpoint
        model.load_state_dict(model_ckpt, strict=True)
        model.eval()
        return model

    @staticmethod
    def to_equal_length(original, signal_digital_signature):
        """
        Ensure both signals are of equal length
        :param original: original length
        :param signal_digital_signature: signal digital length
        :return:
        """
        if original.shape != signal_digital_signature.shape:
            print("Warning: length not equal:", len(original), len(signal_digital_signature))
            min_length = min(len(original), len(signal_digital_signature))
            original = original[0:min_length]
            signal_digital_signature = signal_digital_signature[0:min_length]
        assert original.shape == signal_digital_signature.shape
        return original, signal_digital_signature

    def signal_noise_ratio(self, original, signal_digital_signature):
        """
        Compute the Signal to Noise Ratio (SNR)
        :param original: original length
        :param signal_digital_signature: signal digital length
        :return: SNR
        """
        original, signal_digital_signature = self.to_equal_length(original, signal_digital_signature)
        noise_strength = np.sum((original - signal_digital_signature) ** 2)
        if noise_strength == 0:
            return np.inf
        signal_strength = np.sum(original ** 2)
        ratio = signal_strength / noise_strength
        ratio = max(1e-10, ratio)
        return 10 * np.log10(ratio)

    def encode_trunck_with_snr_check(self, idx_trunck, signal, wm, device, min_snr, max_snr):
        """
        Encode the signal trunk and check the signal to noise ratio (SNR) to ensure quality.
        :param idx_trunck: Index of the trunk being processed.
        :param signal: The original audio signal trunk.
        :param wm: Digital signature.
        :param device: The computing device (cpu or gpu).
        :param min_snr: Minimum acceptable SNR value.
        :param max_snr: Maximum acceptable SNR value.
        :return: The digital signal trunk or the original trunk if encoding was skipped.
        """
        signal_for_encode = signal
        encode_times = 0
        while True:
            encode_times += 1
            signal_wmd = self.encode_trunck(signal_for_encode, wm, device)
            snr = self.signal_noise_ratio(signal, signal_wmd)
            if encode_times == 1 and snr < min_snr:
                print("skip section:%d, snr too low:%.1f" % (idx_trunck, min_snr))
                return signal, "skip"

            if snr < max_snr:
                return signal_wmd, encode_times
            # snr is too hugh
            signal_for_encode = signal_wmd

            if encode_times > 10:
                return signal_wmd, encode_times

    def encode_trunck(self, trunck, wm, device):
        """
        Encode a single trunk with digital information.
        :param trunck: The audio signal trunk to be encoded.
        :param wm: Digital signature.
        :param device: The computing device (cpu or gpu).
        :return: The digital signal trunk.
        """
        with torch.no_grad():
            signal = torch.FloatTensor(trunck).to(device)[None]
            message = torch.FloatTensor(np.array(wm)).to(device)[None]
            signal_wmd_tensor = self.model.encode(signal, message)
            signal_wmd = signal_wmd_tensor.detach().cpu().numpy().squeeze()
            return signal_wmd

    def set_digital_signature(self, bit_arr, data, num_point, shift_range, device, min_snr, max_snr, show_progress):
        """
        Embeds the digital signature into the audio data.
        :param bit_arr: The digital signature bits.
        :param data: The entire audio data.
        :param num_point: Number of data points in each chunk.
        :param shift_range: The fraction of additional data points for each chunk.
        :param device: The computing device (cpu or gpu).
        :param min_snr: Minimum acceptable SNR value.
        :param max_snr: Maximum acceptable SNR value.
        :param show_progress: Whether to show progress while processing.
        :return: The digital signature audio data.
        """
        chunk_size = num_point + int(num_point * shift_range)
        num_segments = int(len(data) / chunk_size)
        len_remain = len(data) - num_segments * chunk_size
        output_chunks = []

        the_iter = range(num_segments)
        if show_progress:
            the_iter = tqdm(the_iter, desc="Processing")

        for i in the_iter:
            start_point = i * chunk_size
            current_chunk = data[start_point:start_point + chunk_size].copy()
            current_chunk_cover_area = current_chunk[0:num_point]
            current_chunk_shift_area = current_chunk[num_point:]
            current_chunk_cover_area_wmd, state = self.encode_trunck_with_snr_check(i, current_chunk_cover_area, bit_arr, device, min_snr, max_snr)

            output = np.concatenate([current_chunk_cover_area_wmd, current_chunk_shift_area])
            assert output.shape == current_chunk.shape
            output_chunks.append(output)

        assert len(output_chunks) > 0
        if len_remain > 0:
            output_chunks.append(data[len(data) - len_remain:])

        reconstructed_array = np.concatenate(output_chunks)
        return reconstructed_array

    def encrypted_audio(self, signal, pattern_bit_length=16, min_snr=20, max_snr=38, show_progress=False):
        """
        Encrypts the audio with a pattern and payload to produce a digital audio signal.
        :param signal: The original audio signal.
        :param pattern_bit_length: Length of the pattern to be used for digital signal.
        :param min_snr: Minimum acceptable SNR value.
        :param max_snr: Maximum acceptable SNR value.
        :param show_progress: Whether to show progress while processing.
        :return: The digital signature audio signal.
        """
        # The pattern bits can be any random sequence.
        # But don't use all-zeros, all-ones, or any periodic sequence, which will seriously hurt decoding performance.
        fix_pattern = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1,
                       1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
                       1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0,
                       0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]

        device = next(self.model.parameters()).device
        pattern_bit = fix_pattern[0:pattern_bit_length]

        digital_signature = np.concatenate([pattern_bit, self.payload])
        assert len(digital_signature) == 32
        signal_wmd = self.set_digital_signature(digital_signature, signal, self.sample_rate, 0.1, device, min_snr, max_snr, show_progress=show_progress)
        return signal_wmd

    @staticmethod
    def read_as_single_channel(file, aim_sr):
        """
        Read wav as mono channel with set sample rate
        :param file: path to file
        :param aim_sr: necessary sample rate
        :return: change wav
        """
        if file.endswith(".mp3"):
            wave, sr = librosa.load(file, sr=aim_sr)
        else:
            wave, sr = soundfile.read(file)

        if len(wave.shape) == 2:  # multi-channel
            wave = wave[:, 0]  # only use the first channel

        # Calculate the duration of the audio in seconds
        duration = len(wave) / sr
        if duration <= 1.0:
            print(f"The audio length is less than or equal to 1 second.")
            return None

        if sr != aim_sr:
            wave = librosa.resample(y=wave, orig_sr=sr, target_sr=aim_sr)
        return wave

    def set_encrypted(self, audio_path, save_path):
        """Set encrypted"""
        file_name = str(uuid.uuid4()) + '.wav'
        save_file = os.path.join(save_path, file_name)
        signal = self.read_as_single_channel(audio_path, aim_sr=self.sample_rate)
        if signal is None:
            return None
        digital_signal = self.encrypted_audio(signal=signal, show_progress=True)
        soundfile.write(save_file, digital_signal, self.sample_rate)
        return save_file
