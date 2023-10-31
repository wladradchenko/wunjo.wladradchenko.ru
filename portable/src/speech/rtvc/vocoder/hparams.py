import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_path, "backend"))

from speech.rtvc.synthesizer.hparams import hparams as _syn_hp

sys.path.pop(0)


# Audio settings------------------------------------------------------------------------
# Match the values of the synthesizer
sample_rate = _syn_hp.sample_rate
n_fft = _syn_hp.n_fft
num_mels = _syn_hp.num_mels
hop_length = _syn_hp.hop_size
win_length = _syn_hp.win_size
fmin = _syn_hp.fmin
min_level_db = _syn_hp.min_level_db
ref_level_db = _syn_hp.ref_level_db
mel_max_abs_value = _syn_hp.max_abs_value
preemphasis = _syn_hp.preemphasis
apply_preemphasis = _syn_hp.preemphasize

bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode
                                    # below


# WAVERNN / VOCODER --------------------------------------------------------------------------------
voc_mode = 'RAW'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from 
# mixture of logistics)
voc_upsample_factors = (5, 5, 8)    # NB - this needs to correctly factorise hop_length
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 10

# Training
voc_batch_size = 100
voc_lr = 1e-4
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider 
                                    # than input length
voc_seq_len = hop_length * 5        # must be a multiple of hop_length

# Generating / Synthesizing
voc_gen_batched = True              # very fast (realtime+) single utterance batched generation
voc_target = 8000                   # target number of samples to be generated in each batch entry
voc_overlap = 400                   # number of samples for crossfading between batches

# Output Noise Reduce
prop_decrease_low_freq = 0.6        # prop decrease for low dominant frequency
prop_decrease_high_freq = 0.9        # prop decrease for high dominant frequency
dry = 0.1                              # dry ratio for facebook denoiser