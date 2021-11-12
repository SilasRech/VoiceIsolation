import numpy as np
import math


def sec_to_frame(x, sampling_rate, hop_size_samples):
    """
    Converts time in seconds to frame index.
    :param x:  time in seconds
    :param sampling_rate:  sampling frequency in hz
    :param hop_size_samples:    hop length in samples
    :return: frame index
    """
    return int(np.floor(sec_to_samples(x, sampling_rate) / hop_size_samples))


def divide_interval(num, start, end):
    """
    Divides the number of states equally to the number of frames in the interval.
    :param num:  number of states.
    :param start: start frame index
    :param end: end frame index
    :return starts: start indexes
    :return end: end indexes
    """
    interval_size = end - start
    # gets remainder
    remainder = interval_size % num
    # init sate count per state with min value
    count = [int((interval_size - remainder) / num)] * num
    # the remainder is assigned to the first n states
    count[:remainder] = [x + 1 for x in count[:remainder]]
    # init starts with first start value
    starts = [start]
    ends = []
    # iterate over the states and sets start and end values
    for c in count[:-1]:
        ends.append(starts[-1] + c)
        starts.append(ends[-1])

    # set last end value
    ends.append(starts[-1] + count[-1])

    return starts, ends


def hz_to_mel(x):
    """
    Converts a frequency given in Hz into the corresponding Mel frequency.
    :param x: input frequency in Hz.
    :return: frequency in mel-scale.
    """

    mel = 2595 * np.log10(1 + x / 700)

    return mel


def mel_to_hz(x):
    """
    Converts a frequency given in Mel back into the linear frequency domain in Hz.
    :param x: input frequency in mel.
    :return: frequency in Hz.
    """

    hz = (10 ** (x / 2595) - 1) * 700

    return hz


def sec_to_samples(x, sampling_rate):
    """
    Converts continuous time to sample index.
    :param x: scalar value representing a point in time in seconds.
    :param sampling_rate: sampling rate in Hz.
    :return: sample_index.
    """

    sample_index = int(x * sampling_rate)

    return sample_index


def next_pow2(x):
    """
    Returns the next power of two for any given positive number.
    :param x: scalar input number.
    :return: next power of two larger than input number.
    """

    return math.ceil(math.log(x, 2))


def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    """
    Returns the total number of frames for a given signal length with corresponding window and hop sizes.
    :param signal_length_samples: total number of samples.
    :param window_size_samples: window size in samples.
    :param hop_size_samples: hop size (frame shift) in samples.
    :return: total number of frames.
    """
    o = window_size_samples - hop_size_samples

    return math.ceil((signal_length_samples - o) / (window_size_samples - o))
