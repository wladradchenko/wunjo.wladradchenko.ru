"""
Based on:
https://github.com/gaganbahga/time_stretch
https://github.com/sannawag/TD-PSOLA
"""

import librosa
import numpy as np
from numpy.fft import fft, ifft


def shift_pitch(signal, fs, factor, psola_params):
    """
    Changing speech tone in 'factor' times.

    :param signal:
    :param fs:
    :param factor:
    :param psola_params:
    :return:
    """
    if factor == 1:
        return signal

    peaks = find_peaks(signal, fs, psola_params)
    new_signal = psola(signal, peaks, factor)

    return new_signal


def find_peaks(signal, fs, psola_params):
    max_hz = psola_params['max_hz']
    min_hz = psola_params['min_hz']
    analysis_win_ms = psola_params['analysis_win_ms']
    max_change = psola_params['max_change']
    min_change = psola_params['min_change']

    N = len(signal)
    min_period = fs // max_hz
    max_period = fs // min_hz

    # compute pitch periodicity
    sequence_len = int(analysis_win_ms / 1000 * fs)  # analysis sequence length in samples
    periods = compute_periods_per_sequence(signal, sequence_len, min_period, max_period)

    # simple hack to avoid octave error: assume that the pitch should not vary much, restrict range
    mean_period = np.mean(periods)
    max_period = int(mean_period * 1.1)
    min_period = int(mean_period * 0.9)
    periods = compute_periods_per_sequence(signal, sequence_len, min_period, max_period)

    # find the peaks
    peaks = [np.argmax(signal[:int(periods[0] * 1.1)])]
    while True:
        prev = peaks[-1]
        idx = prev // sequence_len  # current autocorrelation analysis window
        if prev + int(periods[idx] * max_change) >= N:
            break
        # find maximum near expected location
        peaks.append(prev + int(periods[idx] * min_change) +
                np.argmax(signal[prev + int(periods[idx] * min_change): prev + int(periods[idx] * max_change)]))
    return np.array(peaks)


def compute_periods_per_sequence(signal, sequence_len, min_period, max_period):
    N = len(signal)
    offset = 0  # current sample offset
    periods = []  # period length of each analysis sequence

    while offset < N:
        frame = signal[offset:offset + sequence_len]
        if len(frame) < sequence_len:
            frame_padded = np.zeros((sequence_len, ))
            frame_padded[:len(frame)] = frame
            frame = frame_padded

        fourier = fft(frame)
        fourier[0] = 0  # remove DC component
        autoc = ifft(fourier * np.conj(fourier)).real
        autoc_peak = min_period + np.argmax(autoc[min_period:max_period])
        periods.append(autoc_peak)
        offset += sequence_len

    return periods


def psola(signal, peaks, f_ratio):
    N = len(signal)
    # Interpolate
    new_signal = np.zeros(N)
    new_peaks_ref = np.linspace(0, len(peaks) - 1, int(len(peaks) * f_ratio))
    new_peaks = np.zeros(len(new_peaks_ref)).astype(int)

    for i in range(len(new_peaks)):
        weight = new_peaks_ref[i] % 1
        left = np.floor(new_peaks_ref[i]).astype(int)
        right = np.ceil(new_peaks_ref[i]).astype(int)
        new_peaks[i] = int(peaks[left] * (1 - weight) + peaks[right] * weight)

    # PSOLA
    for j in range(len(new_peaks)):
        # find the corresponding old peak index
        i = np.argmin(np.abs(peaks - new_peaks[j]))
        # get the distances to adjacent peaks
        P1 = [new_peaks[j] if j == 0 else new_peaks[j] - new_peaks[j-1],
              N - 1 - new_peaks[j] if j == len(new_peaks) - 1 else new_peaks[j+1] - new_peaks[j]]
        # edge case truncation
        if peaks[i] - P1[0] < 0:
            P1[0] = peaks[i]
        if peaks[i] + P1[1] > N - 1:
            P1[1] = N - 1 - peaks[i]
        # linear OLA window
        window = list(np.linspace(0, 1, P1[0] + 1)[1:]) + list(np.linspace(1, 0, P1[1] + 1)[1:])
        # center window from original signal at the new peak
        new_signal[new_peaks[j] - P1[0]: new_peaks[j] + P1[1]] += window * signal[peaks[i] - P1[0]: peaks[i] + P1[1]]

    return new_signal


def stretch_wave(x, factor, phase_params):
    """
    Changing speech speed in 'factor' times, preserving its tone

    :param x:
    :param factor:
    :param phase_params:
    :return:
    """
    if factor == 1:
        return x

    nfft = phase_params['nfft']
    hop = phase_params['hop']

    stft = librosa.core.stft(x, n_fft=nfft).transpose()
    stft_cols = stft.shape[1]

    times = np.arange(0, stft.shape[0], factor)
    phase_adv = (2 * np.pi * hop * np.arange(0, stft_cols))/ nfft
    stft = np.concatenate((stft, np.zeros((1, stft_cols))), axis=0)

    indices = np.floor(times).astype(np.int)
    alpha = np.expand_dims(times - np.floor(times), axis=1)
    mag = (1. - alpha) * np.absolute(stft[indices, :]) + alpha * np.absolute(stft[indices + 1, :])

    dphi = np.angle(stft[indices + 1, :]) - np.angle(stft[indices, :]) - phase_adv
    dphi = dphi - 2 * np.pi * np.floor(dphi/(2 * np.pi))

    phase_adv_acc = np.matmul(np.expand_dims(np.arange(len(times) + 1),axis=1), np.expand_dims(phase_adv, axis=0))
    phase = np.concatenate( (np.zeros((1, stft_cols)), np.cumsum(dphi, axis=0)), axis=0) + phase_adv_acc
    phase += np.angle(stft[0, :])

    stft_new = mag * np.exp(phase[:-1, :] * 1j)

    return librosa.core.istft(stft_new.transpose())