import os
import cv2
import struct
import subprocess
import numpy as np
from time import time
from typing import List
from tqdm import tqdm
from numba import njit

from diffusers.src.flow.flow_utils import FlowCalc
from diffusers.src.blender.video_sequence import VideoSequence
from diffusers.src.blender.poisson_fusion import poisson_fusion
import diffusers.src.blender.histogram_blend as histogram_blend
from diffusers.src.blender.guide import BaseGuide, ColorGuide, EdgeGuide, PositionalGuide, TemporalGuide


@njit
def g_error_mask_loop(H, W, dist1, dist2, output, weight1, weight2):
    for i in range(H):
        for j in range(W):
            if weight1 * dist1[i, j] < weight2 * dist2[i, j]:
                output[i, j] = 0
            else:
                output[i, j] = 1
            if weight1 == 0:
                output[i, j] = 0
            elif weight2 == 0:
                output[i, j] = 1


def g_error_mask(dist1, dist2, weight1=1, weight2=1):
    H, W = dist1.shape
    output = np.empty_like(dist1, dtype=np.byte)
    g_error_mask_loop(H, W, dist1, dist2, output, weight1, weight2)
    return output


@njit
def assemble_min_error_img_loop(H, W, a, b, error_mask, out):
    for i in range(H):
        for j in range(W):
            if error_mask[i, j] == 0:
                out[i, j] = a[i, j]
            else:
                out[i, j] = b[i, j]


def assemble_min_error_img(a, b, error_mask):
    H, W = a.shape[0:2]
    out = np.empty_like(a)
    assemble_min_error_img_loop(H, W, a, b, error_mask, out)
    return out


class Ebsynth:
    def __init__(self, gmflow_model_path, ebsynth_path):
        self.flow_calc = FlowCalc(gmflow_model_path)
        self.ebsynth_bin = ebsynth_path

    def processing_ebsynth(self, base_folder, input_subdir, frames_path, frames: list):
        video_sequence = self.create_sequence(base_folder=base_folder, input_subdir=input_subdir, frames_path=frames_path, frame_files_with_interval=frames)
        self.run_ebsynth(video_sequence)
        blend_histogram = True
        blend_gradient = True
        for i in range(video_sequence.n_seq):
            self.general_process_sequence(video_sequence, i, blend_histogram, blend_gradient)
        return video_sequence.blending_out_dir, video_sequence.output_format

    def general_process_sequence(self, video_sequence: VideoSequence, i, blend_histogram=True, blend_gradient=True):
        key_img_path = os.path.join(video_sequence.key_dir, video_sequence.frame_files[i])
        if not os.path.exists(key_img_path):
            return
        key1_img = cv2.imread(key_img_path)
        img_shape = key1_img.shape
        beg_id = video_sequence.get_sequence_beg_id(i)

        oas = video_sequence.get_output_sequence(i)
        obs = video_sequence.get_output_sequence(i, False)
        if oas is None or obs is None:
            return

        binas = [x.replace('jpg', 'bin') for x in oas]
        binbs = [x.replace('jpg', 'bin') for x in obs]

        obs = [obs[0]] + list(reversed(obs[1:]))
        inputs = video_sequence.get_input_sequence(i)
        oas = [cv2.imread(x) for x in oas if os.path.exists(x)]
        obs = [cv2.imread(x) for x in obs if os.path.exists(x)]
        inputs = [cv2.imread(x) for x in inputs if os.path.exists(x)]
        flow_seq = video_sequence.get_flow_sequence(i)

        dist1s = []
        dist2s = []
        for i in range(len(binbs) - 1):
            bin_a = binas[i + 1]
            bin_b = binbs[i + 1]
            dist1s.append(self.load_error(bin_a, img_shape))
            dist2s.append(self.load_error(bin_b, img_shape))

        lb = 0
        ub = 1
        beg = time()
        p_mask = None

        # write key img
        blend_out_path = video_sequence.get_blending_img(beg_id)
        cv2.imwrite(blend_out_path, key1_img)

        # Modify iteration to use flow_seq's length
        for i in range(len(flow_seq)):
            c_id = beg_id + i + 1
            blend_out_path = video_sequence.get_blending_img(c_id)

            dist1 = dist1s[i]
            dist2 = dist2s[i]
            oa = oas[i + 1]
            ob = obs[i + 1]
            weight1 = i / (len(binbs) - 1) * (ub - lb) + lb
            weight2 = 1 - weight1
            mask = g_error_mask(dist1, dist2, weight1, weight2)
            if p_mask is not None:
                # Use flow_seq[i] to get the correct flow path for the current iteration
                flow = self.flow_calc.get_flow(inputs[i], inputs[i + 1], flow_seq[i])
                p_mask = self.flow_calc.warp(p_mask, flow, 'nearest')
                mask = p_mask | mask
            p_mask = mask

            min_error_img = assemble_min_error_img(oa, ob, mask)
            if blend_histogram:
                hb_res = histogram_blend.blend(oa, ob, min_error_img, (1 - weight1), (1 - weight2))
            else:
                tmpa = oa.astype(np.float32)
                tmpb = ob.astype(np.float32)
                hb_res = (1 - weight1) * tmpa + (1 - weight2) * tmpb

            # gradient blend
            if blend_gradient:
                res = poisson_fusion(hb_res, oa, ob, mask)
            else:
                res = hb_res

            cv2.imwrite(blend_out_path, res)
        end = time()
        print(f'Ebsynth others: {round(end - beg)} sec')

    @staticmethod
    def load_error(bin_path, img_shape):
        img_size = img_shape[0] * img_shape[1]
        with open(bin_path, 'rb') as fp:
            bytes = fp.read()
        read_size = struct.unpack('q', bytes[:8])
        assert read_size[0] == img_size
        float_res = struct.unpack('f' * img_size, bytes[8:])
        res = np.array(float_res, dtype=np.float32).reshape(img_shape[0], img_shape[1])
        return res

    def create_sequence(self, base_folder, input_subdir, frames_path, frame_files_with_interval):
        sequence = VideoSequence(
            base_dir=base_folder, frames_path=frames_path, frame_files_with_interval=frame_files_with_interval,
            input_subdir=input_subdir, tmp_subdir="tmp", input_format="%04d.png", key_format="%04d.png"
        )
        return sequence

    def run_ebsynth(self, video_sequence: VideoSequence):
        """Run ebsynth in one process"""
        beg = time()
        i_arr = list(range(0, len(video_sequence.frame_files)))
        self.process_sequences(i_arr, video_sequence)
        end = time()
        print(f'Ebsynth process: {round(end - beg)} sec')

    def process_sequences(self, i_arr, video_sequence: VideoSequence):
        [self.process_one_sequence(i, video_sequence) for i in i_arr]

    def process_one_sequence(self, i, video_sequence: VideoSequence):
        frame_files = video_sequence.frame_files
        for is_forward in [True, False]:
            input_seq = video_sequence.get_input_sequence(i, is_forward)
            if not input_seq:
                continue
            output_seq = video_sequence.get_output_sequence(i, is_forward)
            if not output_seq:
                continue
            flow_seq = video_sequence.get_flow_sequence(i, is_forward)
            if not flow_seq:
                continue
            key_img_id = i if is_forward else i + 1
            if len(frame_files) - 1 < i + 1:
                continue
            key_img = os.path.join(video_sequence.key_dir, frame_files[key_img_id])
            for j in range(len(input_seq) - 1):
                i1 = cv2.imread(input_seq[j])
                i2 = cv2.imread(input_seq[j + 1])
                self.flow_calc.get_flow(i1, i2, flow_seq[j])

            print("Get guides for style")
            guides: List[BaseGuide] = [
                ColorGuide(input_seq, self.flow_calc),
                EdgeGuide(input_seq, video_sequence.get_edge_sequence(i, is_forward), self.flow_calc),
                TemporalGuide(key_img, output_seq, flow_seq, video_sequence.get_temporal_sequence(i, is_forward), self.flow_calc),
                PositionalGuide(flow_seq, video_sequence.get_pos_sequence(i, is_forward), self.flow_calc)
            ]
            weights = [6, 0.5, 0.5, 2]

            print("Run ebsynth on style")
            progress_bar = tqdm(total=len(input_seq), unit='it', unit_scale=True)
            for j in range(len(input_seq)):
                # key frame
                if j == 0:
                    img = cv2.imread(key_img)
                    cv2.imwrite(output_seq[0], img)
                else:
                    cmd = f'{self.ebsynth_bin} -style {os.path.abspath(key_img)}'
                    for g, w in zip(guides, weights):
                        cmd += ' ' + g.get_cmd(j, w)
                    cmd += f' -output {os.path.abspath(output_seq[j])} -searchvoteiters 12 -patchmatchiters 6'
                    # not silence run TODO remove
                    # os.system(cmd)
                    # silence run
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # update progress bar
                progress_bar.update(1)
            # close progress bar for key
            progress_bar.close()
