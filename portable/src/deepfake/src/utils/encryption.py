import os
import sys
import torch
import struct
import uuid
import pywt
import base64
import cv2
import subprocess
import numpy as np


class RivaEncryption(object):
    encoder = None
    decoder = None

    def __init__(self, encryptions=[], wmLen=32, threshold=0.52):
        self._encryptions = encryptions
        self._threshold = threshold
        if wmLen not in [32]:
            raise RuntimeError('Encryption only supports 32 bits encryptions now.')
        self._data = torch.from_numpy(np.array([self._encryptions], dtype=np.float32))

    @classmethod
    def loadModel(cls):
        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "The `RivaEncryption` class requires onnxruntime to be installed. "
                "You can install it with pip: `pip install onnxruntime`."
            )
        if RivaEncryption.encoder and RivaEncryption.decoder:
            return
        modelDir = os.path.dirname(os.path.abspath(__file__))
        encoder_path = os.path.join(modelDir, 'encoder.onnx')
        decoder_path = os.path.join(modelDir, 'decoder.onnx')
        # access read model
        if sys.platform == 'win32':
            username = os.environ.get('USERNAME') or os.environ.get('USER')
            encoder_cmd = f'icacls "{encoder_path}" /grant:r "{username}:(R,W)" /T'
            decoder_cmd = f'icacls "{decoder_path}" /grant:r "{username}:(R,W)" /T'
            if os.environ.get('DEBUG', 'False') == 'True':
                # not silence run
                os.system(encoder_cmd)
                os.system(decoder_cmd)
            else:
                # silence run
                subprocess.run(encoder_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(decoder_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == 'linux':
            # access read model
            encoder_cmd = f"chmod +x {encoder_path}"
            decoder_cmd = f"chmod +x {decoder_path}"
            if os.environ.get('DEBUG', 'False') == 'True':
                # not silence run
                os.system(encoder_cmd)
                os.system(decoder_cmd)
            else:
                # silence run
                subprocess.run(encoder_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(decoder_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        RivaEncryption.encoder = onnxruntime.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
        RivaEncryption.decoder = onnxruntime.InferenceSession(os.path.join(modelDir, 'decoder.onnx'), providers=['CPUExecutionProvider'])

    def encode(self, frame):
        if not RivaEncryption.encoder:
            RivaEncryption.loadModel()

        frame = torch.from_numpy(np.array([frame], dtype=np.float32)) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0)

        inputs = {
            'frame': frame.detach().cpu().numpy(),
            'data': self._data.detach().cpu().numpy()
        }

        outputs = RivaEncryption.encoder.run(None, inputs)
        wm_frame = outputs[0]
        wm_frame = torch.clamp(torch.from_numpy(wm_frame), min=-1.0, max=1.0)
        wm_frame = (
            (wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
        ).detach().cpu().numpy().astype('uint8')

        return wm_frame

    def decode(self, frame):
        if not RivaEncryption.decoder:
            RivaEncryption.loadModel()

        frame = torch.from_numpy(np.array([frame], dtype=np.float32)) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0)
        inputs = {
            'frame': frame.detach().cpu().numpy(),
        }
        outputs = RivaEncryption.decoder.run(None, inputs)
        data = outputs[0][0]
        return np.array(data > self._threshold, dtype=np.uint8)


class EmbedDwtDctSvd(object):
    def __init__(self, encryptions=[], wmLen=8, scales=[0,36,0], block=4):
        self._encryptions = encryptions
        self._wmLen = wmLen
        self._scales = scales
        self._block = block

    def encode(self, bgr):
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1,(h1,v1,d1) = pywt.dwt2(yuv[:row//4*4,:col//4*4,channel], 'haar')
            self.encode_frame(ca1, self._scales[channel])

            yuv[:row//4*4,:col//4*4,channel] = pywt.idwt2((ca1, (v1,h1,d1)), 'haar')

        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def decode(self, bgr):
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores = [[] for i in range(self._wmLen)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1,(h1,v1,d1) = pywt.dwt2(yuv[:row//4*4,:col//4*4,channel], 'haar')

            scores = self.decode_frame(ca1, self._scales[channel], scores)

        avgScores = list(map(lambda l: np.array(l).mean(), scores))

        bits = (np.array(avgScores) * 255 > 127)
        return bits

    def decode_frame(self, frame, scale, scores):
        (row, col) = frame.shape
        num = 0

        for i in range(row//self._block):
            for j in range(col//self._block):
                block = frame[i*self._block : i*self._block + self._block,
                              j*self._block : j*self._block + self._block]

                score = self.infer_dct_svd(block, scale)
                wmBit = num % self._wmLen
                scores[wmBit].append(score)
                num = num + 1

        return scores

    def diffuse_dct_svd(self, block, wmBit, scale):
        u,s,v = np.linalg.svd(cv2.dct(block))

        s[0] = (s[0] // scale + 0.25 + 0.5 * wmBit) * scale
        return cv2.idct(np.dot(u, np.dot(np.diag(s), v)))

    def infer_dct_svd(self, block, scale):
        u, s, v = np.linalg.svd(cv2.dct(block))
        score = 0
        score = int((s[0] % scale) > scale * 0.5)
        return score
        if score >= 0.5:
            return 1.0
        else:
            return 0.0

    def encode_frame(self, frame, scale):
        '''
        frame is a matrix (M, N)

        we get K (encryption bits size) blocks (self._block x self._block)

        For i-th block, we encode encryption[i] bit into it
        '''
        (row, col) = frame.shape
        num = 0
        for i in range(row//self._block):
            for j in range(col//self._block):
                block = frame[i*self._block : i*self._block + self._block, j*self._block : j*self._block + self._block]
                wmBit = self._encryptions[(num % self._wmLen)]


                diffusedBlock = self.diffuse_dct_svd(block, wmBit, scale)
                frame[i*self._block : i*self._block + self._block, j*self._block : j*self._block + self._block] = diffusedBlock

                num = num+1


class EmbedMaxDct(object):
    def __init__(self, encryptions=[], wmLen=8, scales=[0,36,36], block=4):
        self._encryptions = encryptions
        self._wmLen = wmLen
        self._scales = scales
        self._block = block

    def encode(self, bgr):
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1,(h1,v1,d1) = pywt.dwt2(yuv[:row//4*4,:col//4*4,channel], 'haar')
            self.encode_frame(ca1, self._scales[channel])

            yuv[:row//4*4,:col//4*4,channel] = pywt.idwt2((ca1, (v1,h1,d1)), 'haar')

        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def decode(self, bgr):
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores = [[] for i in range(self._wmLen)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1,(h1,v1,d1) = pywt.dwt2(yuv[:row//4*4,:col//4*4,channel], 'haar')

            scores = self.decode_frame(ca1, self._scales[channel], scores)

        avgScores = list(map(lambda l: np.array(l).mean(), scores))

        bits = (np.array(avgScores) * 255 > 127)
        return bits

    def decode_frame(self, frame, scale, scores):
        (row, col) = frame.shape
        num = 0

        for i in range(row//self._block):
            for j in range(col//self._block):
                block = frame[i*self._block : i*self._block + self._block,
                              j*self._block : j*self._block + self._block]

                score = self.infer_dct_matrix(block, scale)
                #score = self.infer_dct_svd(block, scale)
                wmBit = num % self._wmLen
                scores[wmBit].append(score)
                num = num + 1

        return scores

    def diffuse_dct_svd(self, block, wmBit, scale):
        u,s,v = np.linalg.svd(cv2.dct(block))

        s[0] = (s[0] // scale + 0.25 + 0.5 * wmBit) * scale
        return cv2.idct(np.dot(u, np.dot(np.diag(s), v)))

    def infer_dct_svd(self, block, scale):
        u,s,v = np.linalg.svd(cv2.dct(block))
        score = 0
        score = int ((s[0] % scale) > scale * 0.5)
        return score
        if score >= 0.5:
            return 1.0
        else:
            return 0.0

    def diffuse_dct_matrix(self, block, wmBit, scale):
        pos = np.argmax(abs(block.flatten()[1:])) + 1
        i, j = pos // self._block, pos % self._block
        val = block[i][j]
        if val >= 0.0:
            block[i][j] = (val//scale + 0.25 + 0.5 * wmBit) * scale
        else:
            val = abs(val)
            block[i][j] = -1.0 * (val//scale + 0.25 + 0.5 * wmBit) * scale
        return block

    def infer_dct_matrix(self, block, scale):
        pos = np.argmax(abs(block.flatten()[1:])) + 1
        i, j = pos // self._block, pos % self._block

        val = block[i][j]
        if val < 0:
            val = abs(val)

        if (val % scale) > 0.5 * scale:
            return 1
        else:
            return 0

    def encode_frame(self, frame, scale):
        '''
        frame is a matrix (M, N)

        we get K (encryption bits size) blocks (self._block x self._block)

        For i-th block, we encode encryption[i] bit into it
        '''
        (row, col) = frame.shape
        num = 0
        for i in range(row//self._block):
            for j in range(col//self._block):
                block = frame[i*self._block : i*self._block + self._block, j*self._block : j*self._block + self._block]
                wmBit = self._encryptions[(num % self._wmLen)]
                diffusedBlock = self.diffuse_dct_matrix(block, wmBit, scale)
                frame[i*self._block: i*self._block + self._block, j*self._block : j*self._block + self._block] = diffusedBlock
                num = num+1


class EncryptionEncoder(object):
    def __init__(self, content=b''):
        seq = np.array([n for n in content], dtype=np.uint8)
        self._encryptions = list(np.unpackbits(seq))
        self._wmLen = len(self._encryptions)
        self._wmType = 'bytes'

    def set_by_ipv4(self, addr):
        bits = []
        ips = addr.split('.')
        for ip in ips:
            bits += list(np.unpackbits(np.array([ip % 255], dtype=np.uint8)))
        self._encryptions = bits
        self._wmLen = len(self._encryptions)
        self._wmType = 'ipv4'
        assert self._wmLen == 32

    def set_by_uuid(self, uid):
        u = uuid.UUID(uid)
        self._wmType = 'uuid'
        seq = np.array([n for n in u.bytes], dtype=np.uint8)
        self._encryptions = list(np.unpackbits(seq))
        self._wmLen = len(self._encryptions)

    def set_by_bytes(self, content):
        self._wmType = 'bytes'
        seq = np.array([n for n in content], dtype=np.uint8)
        self._encryptions = list(np.unpackbits(seq))
        self._wmLen = len(self._encryptions)

    def set_by_b16(self, b16):
        content = base64.b16decode(b16)
        self.set_by_bytes(content)
        self._wmType = 'b16'

    def set_by_bits(self, bits=[]):
        self._encryptions = [int(bit) % 2 for bit in bits]
        self._wmLen = len(self._encryptions)
        self._wmType = 'bits'

    def set_encryption(self, wmType='bytes', content=''):
        if wmType == 'ipv4':
            self.set_by_ipv4(content)
        elif wmType == 'uuid':
            self.set_by_uuid(content)
        elif wmType == 'bits':
            self.set_by_bits(content)
        elif wmType == 'bytes':
            self.set_by_bytes(content)
        elif wmType == 'b16':
            self.set_by_b16(content)
        else:
            raise NameError('%s is not supported' % wmType)

    def get_length(self):
        return self._wmLen

    @classmethod
    def loadModel(cls):
        RivaEncryption.loadModel()

    def encode(self, cv2Image, method='dwtDct', **configs):
        (r, c, channels) = cv2Image.shape
        if r*c < 256*256:
            raise RuntimeError('image too small, should be larger than 256x256')

        if method == 'dwtDct':
            embed = EmbedMaxDct(self._encryptions, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == 'dwtDctSvd':
            embed = EmbedDwtDctSvd(self._encryptions, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == 'rivaGan':
            embed = RivaEncryption(self._encryptions, self._wmLen)
            return embed.encode(cv2Image)
        else:
            raise NameError('%s is not supported' % method)

class EncryptionDecoder(object):
    def __init__(self, wm_type='bytes', length=0):
        self._wmType = wm_type
        if wm_type == 'ipv4':
            self._wmLen = 32
        elif wm_type == 'uuid':
            self._wmLen = 128
        elif wm_type == 'bytes':
            self._wmLen = length
        elif wm_type == 'bits':
            self._wmLen = length
        elif wm_type == 'b16':
            self._wmLen = length
        else:
            raise NameError('%s is unsupported' % wm_type)

    def reconstruct_ipv4(self, bits):
        ips = [str(ip) for ip in list(np.packbits(bits))]
        return '.'.join(ips)

    def reconstruct_uuid(self, bits):
        nums = np.packbits(bits)
        bstr = b''
        for i in range(16):
            bstr += struct.pack('>B', nums[i])

        return str(uuid.UUID(bytes=bstr))

    def reconstruct_bits(self, bits):
        return bits

    def reconstruct_b16(self, bits):
        bstr = self.reconstruct_bytes(bits)
        return base64.b16encode(bstr)

    def reconstruct_bytes(self, bits):
        nums = np.packbits(bits)
        bstr = b''
        for i in range(self._wmLen//8):
            bstr += struct.pack('>B', nums[i])
        return bstr

    def reconstruct(self, bits):
        if len(bits) != self._wmLen:
            raise RuntimeError('bits are not matched with encryption length')

        if self._wmType == 'ipv4':
            return self.reconstruct_ipv4(bits)
        elif self._wmType == 'uuid':
            return self.reconstruct_uuid(bits)
        elif self._wmType == 'bits':
            return self.reconstruct_bits(bits)
        elif self._wmType == 'b16':
            return self.reconstruct_b16(bits)
        else:
            return self.reconstruct_bytes(bits)

    def decode(self, cv2Image, method='dwtDct', **configs):
        (r, c, channels) = cv2Image.shape
        if r*c < 256*256:
            raise RuntimeError('image too small, should be larger than 256x256')

        bits = []
        if method == 'dwtDct':
            embed = EmbedMaxDct(encryptions=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        elif method == 'dwtDctSvd':
            embed = EmbedDwtDctSvd(encryptions=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        elif method == 'rivaGan':
            embed = RivaEncryption(encryptions=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        else:
            raise NameError('%s is not supported' % method)
        return self.reconstruct(bits)

    @classmethod
    def loadModel(cls):
        RivaEncryption.loadModel()