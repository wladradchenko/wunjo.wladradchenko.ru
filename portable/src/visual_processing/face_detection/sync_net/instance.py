#!/usr/bin/python
# -*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import time, subprocess, os, math
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from .model import *


# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1

    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    dists = []

    for i in range(0, len(feat1)):
        dists.append(
            torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :]))

    return dists


# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout=0, num_layers_in_fc_layers=1024):
        super(SyncNetInstance, self).__init__()

        self.__S__ = S(num_layers_in_fc_layers=num_layers_in_fc_layers).cuda()

    def evaluate(self, tmp_dir: str, video_file: str, batch_size: int = 20, vshift: int = 15):
        self.__S__.eval()
        os.makedirs(tmp_dir, exist_ok=True)

        # ========== Extract frames and audio ==========
        cmd_img = f"ffmpeg -y -i {video_file} -threads 1 -f image2 {os.path.join(tmp_dir, '%06d.jpg')}"
        cmd_audio = f"ffmpeg -y -i {video_file} -ac 1 -vn -acodec pcm_s16le -ar 16000 {os.path.join(tmp_dir, 'audio.wav')}"
        for cmd in [cmd_img, cmd_audio]:
            if os.environ.get('DEBUG', 'False') == 'True':
                os.system(cmd)
            else:
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # ========== Load audio ==========
        sample_rate, audio = wavfile.read(os.path.join(tmp_dir, 'audio.wav'))
        mfcc_full = python_speech_features.mfcc(audio, sample_rate)
        mfcc_full = numpy.array(list(zip(*mfcc_full)))  # transpose to [feat, time]

        # ========== Prepare file list ==========
        flist = sorted([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.lower().endswith('.jpg')])
        lastframe = min(len(flist), len(audio) // 640) - 5

        dists = []
        with torch.no_grad():
            for i in range(0, lastframe, batch_size):
                im_batch_list = []
                cc_batch_list = []

                for vframe in range(i, min(i + batch_size, lastframe)):
                    # ========== Read 5-frame window ==========
                    frames5 = []
                    for fidx in range(vframe, vframe + 5):
                        img = cv2.imread(flist[fidx])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        frames5.append(img)
                    frames5 = numpy.stack(frames5, axis=0)  # [5,H,W,3]
                    frames5 = numpy.transpose(frames5, (3, 0, 1, 2))  # [C=3,5,H,W]
                    im_tensor = torch.from_numpy(frames5.astype(numpy.float32)).unsqueeze(0).cuda()
                    im_batch_list.append(im_tensor)

                    # ========== Slice audio MFCC ==========
                    start = vframe * 4
                    end = start + 20
                    cc_slice = torch.from_numpy(mfcc_full[:, start:end].astype(numpy.float32)).unsqueeze(0).unsqueeze(0).cuda()
                    cc_batch_list.append(cc_slice)

                # ========== Forward pass ==========
                im_in = torch.cat(im_batch_list, 0)
                cc_in = torch.cat(cc_batch_list, 0)

                im_out = self.__S__.forward_lip(im_in)
                cc_out = self.__S__.forward_aud(cc_in)

                dists.append((im_out.cpu(), cc_out.cpu()))

        # ========== Compute offset ==========
        im_feats = torch.cat([x[0] for x in dists], 0)
        cc_feats = torch.cat([x[1] for x in dists], 0)

        dists_list = calc_pdist(im_feats, cc_feats, vshift=vshift)
        mdist = torch.mean(torch.stack(dists_list, 1), 1)
        minval, minidx = torch.min(mdist, 0)

        offset = vshift - minidx
        conf = torch.median(mdist) - minval

        fdist = numpy.stack([dist[minidx].numpy() for dist in dists_list])
        fconfm = signal.medfilt(torch.median(mdist).numpy() - fdist, kernel_size=9)

        print('Framewise conf:\n', fconfm)
        print('AV offset: %d | Min dist: %.3f | Confidence: %.3f' % (offset, minval, conf))

        dists_npy = numpy.array([dist.numpy() for dist in dists_list])
        return offset.numpy(), conf.numpy(), dists_npy


    def evaluate_old(self, tmp_dir: str, video_file: str, batch_size: int = 20, vshift: int = 15):

        self.__S__.eval()

        # ========== ==========
        # Convert files
        # ========== ==========

        os.makedirs(tmp_dir, exist_ok=True)

        cmd = "ffmpeg -y -i %s -threads 1 -f image2 %s" % (video_file, os.path.join(tmp_dir, '%06d.jpg'))
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cmd = "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (video_file, os.path.join(tmp_dir, 'audio.wav'))
        if os.environ.get('DEBUG', 'False') == 'True':
            # not silence run
            os.system(cmd)
        else:
            # silence run
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # ========== ==========
        # Load video 
        # ========== ==========

        images = []

        flist = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.lower().endswith('.jpg')]
        flist.sort()

        for fname in flist:
            images.append(cv2.imread(fname))

        im = numpy.stack(images, axis=3)
        im = numpy.expand_dims(im, axis=0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(os.path.join(tmp_dir, 'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio)) / 16000) != (float(len(images)) / 25):
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different." % (
            float(len(audio)) / 16000, float(len(images)) / 25))

        min_length = min(len(images), math.floor(len(audio) / 640))

        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0, lastframe, batch_size):
            im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size))]
            im_in = torch.cat(im_batch, 0)
            im_out = self.__S__.forward_lip(im_in.cuda())
            im_feat.append(im_out.data.cpu())

            cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in range(i, min(lastframe, i + batch_size))]
            cc_in = torch.cat(cc_batch, 0)
            cc_out = self.__S__.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        # ========== ==========
        # Compute offset
        # ========== ==========

        print('Compute time %.3f sec.' % (time.time() - tS))

        dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)

        minval, minidx = torch.min(mdist, 0)

        offset = vshift - minidx
        conf = torch.median(mdist) - minval

        fdist = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf = torch.median(mdist).numpy() - fdist
        fconfm = signal.medfilt(fconf, kernel_size=9)

        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Framewise conf: ')
        print(fconfm)
        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset, minval, conf))

        dists_npy = numpy.array([dist.numpy() for dist in dists])
        return offset.numpy(), conf.numpy(), dists_npy

    def extract_feature(self, video_file: str, batch_size: int = 20):

        self.__S__.eval()

        # ========== ==========
        # Load video 
        # ========== ==========
        cap = cv2.VideoCapture(video_file)

        frame_num = 1
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = numpy.stack(images, axis=3)
        im = numpy.expand_dims(im, axis=0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images) - 4
        im_feat = []

        tS = time.time()
        for i in range(0, lastframe, batch_size):
            im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size))]
            im_in = torch.cat(im_batch, 0)
            im_out = self.__S__.forward_lipfeat(im_in.cuda())
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)

        # ========== ==========
        # Compute offset
        # ========== ==========

        print('Compute time %.3f sec.' % (time.time() - tS))

        return im_feat

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)

        self_state = self.__S__.state_dict()

        for name, param in loaded_state.items():
            self_state[name].copy_(param)