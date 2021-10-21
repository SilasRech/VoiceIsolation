import math
import numpy as np
import numpy as np
from scipy import fftpack
from scipy.io import wavfile
from scipy.fftpack import dct
import scipy.io.wavfile


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


def sec_to_frame(x, sampling_rate, hop_size_samples):
    """
    Converts time in seconds to frame index.
    :param x:  time in seconds
    :param sampling_rate:  sampling frequency in hz
    :param hop_size_samples:    hop length in samples
    :return: frame index
    """
    return int(np.floor(sec_to_samples(x, sampling_rate) / hop_size_samples))


def make_frames(audio_data, sampling_rate, window_size, hop_size):
    """
    Splits an audio signal into subsequent frames.

    :param audio_data: array representing the audio signal.
    :param sampling_rate: sampling rate in Hz.
    :param window_size: window size in seconds.
    :param hop_size: hop size (frame shift) in seconds.
    :return: n x m array of signal frames, where n is the number of frames and m is the window size in samples.
    """

    # transform window size in seconds to samples and calculate next higher power of two
    window_size_samples = sec_to_samples(window_size, sampling_rate)
    window_size_samples = 2 ** next_pow2(window_size_samples)

    # assign hamming window
    hamming_window = np.hamming(window_size_samples)

    # transform hop size in seconds to samples
    hop_size_samples = sec_to_samples(hop_size, sampling_rate)

    # get number of frames from function in tools.py
    n_frames = get_num_frames(len(audio_data), window_size_samples, hop_size_samples)

    # initialize nxm matrix (n is number of frames, m is window size)
    # initialize with zeros to avoid zero padding
    frames = np.zeros([n_frames, window_size_samples], dtype=float)

    # write frames in matrix
    for i in range(n_frames):
        start = i * hop_size_samples
        end = i * hop_size_samples + window_size_samples
        frames[i, 0:len(audio_data[start:end])] = audio_data[start:end]
        frames[i, :] = frames[i, :] * hamming_window

    return frames


def compute_absolute_spectrum(frames):
    """
    Calculates spectral analysis with Short-Time Fourier Transform.
    :param frames: framing matrix of audio data.
    :return: non-redundant part of absolute spectrum.
    """
    return abs(np.fft.rfft(frames))


def get_mel_filters(sampling_rate, window_size_sec, n_filters, f_min=0, f_max=8000):
    """
    Returns a mel filterbank for a given set of specifications.
    :param sampling_rate: sampling rate in Hz.
    :param window_size_sec: window size in seconds.
    :param n_filters: number of filters.
    :param f_min: minimum frequency covered by mel filterbank in Hz (default: 0).
    :param f_max: maximum frequency covered by mel filterbank in Hz (default: 8000).
    :return: m x d array representing the mel filterbank, where m is the FFT size and d is the number of mel filters.
    """

    # calculate max and min frequency in mel
    f_min_mel = hz_to_mel(f_min)
    f_max_mel = hz_to_mel(f_max)

    # create vector with frequency points for filterbank in mel scale (equidistant)
    freq_points_mel = np.linspace(f_min_mel, f_max_mel, n_filters + 2)
    # transform it into Hertz scale
    freq_points_hz = mel_to_hz(freq_points_mel)

    # calculate number of FFT frequency points
    fft_samples = int((2 ** next_pow2(sec_to_samples(window_size_sec, sampling_rate)) / 2) + 1)

    # find the corresponding indices for the filterbank in the FFT
    f = []
    for i in range(n_filters + 2):
        f.append(np.round((fft_samples) * freq_points_hz[i] / f_max))

    # initialize filterbank matrix H
    H = np.zeros((fft_samples, n_filters))

    # calculate filterbank matrix H
    for m in range(1, n_filters + 1):
        for k in range(fft_samples):
            if k < f[m - 1]:
                H[k, m - 1] = 0
            elif f[m - 1] <= k and k < f[m]:
                H[k, m - 1] = (2 * (k - f[m - 1])) / ((f[m + 1] - f[m - 1]) * (f[m] - f[m - 1]))
            elif f[m] <= k and k <= f[m + 1]:
                H[k, m - 1] = (2 * (f[m + 1] - k)) / ((f[m + 1] - f[m - 1]) * (f[m + 1] - f[m]))
            elif k > f[m + 1]:
                H[k, m - 1] = 0
    return H


def apply_mel_filters(abs_spectrum, filterbank):
    """
    Applies a mel filterbank to a given signal spectrum.
    :param abs_spectrum: signal spectrum obtained using compute_absolute_spectrum().
    :param filterbank: mel filterbank, obtained using get_mel_filters().
    :return: mel spectrum.
    """

    return np.dot(abs_spectrum, filterbank)


def compute_cepstrum(mel_spectrum, num_ceps):
    """
    Computes cepstrum out of mel spectrum

    :param mel_spectrum: mel spectrum
    :param num_ceps: number of cepstral coefficients
    :return: cepstrum
    """
    return dct(mel_spectrum, norm='ortho')[:, 0:num_ceps]


def get_delta(x):
    """
    Calculate first derivative of MFCC

    :param x: cepstrum
    :return: first derivative of cesptrum
    """

    # get size of cepstrum matrix
    x_len, y_len = x.shape
    # initialize delta x
    d_x = np.zeros([x_len, y_len])
    # initialize edges of delta x
    d_x[0, :] = x[1, :] - x[0, :]
    d_x[x_len - 1, :] = x[x_len - 1, :] - x[x_len - 2, :]

    # calculate first derivative
    for tau in range(1, x_len - 1, 1):
        d_x[tau, :] = 0.5 * (x[tau + 1, :] - x[tau - 1, :])

    return d_x


def append_delta(x, delta):
    """
    Concetenate MFCC spectrum and first derivative

    :param x: cepstrum
    :param delta: first derivative of cepstrum
    :return: concatenated array
    """
    return np.concatenate((x, delta), axis=1)


def compute_features_with_context(audio_file,
                                  window_size=25e-3,
                                  hop_size=10e-3,
                                  feature_type='STFT',
                                  n_filters=24,
                                  fbank_fmin=0,
                                  fbank_fmax=8000,
                                  num_ceps=13,
                                  left_context=4,
                                  right_context=4,
                                  ):
    """
    Extracts features and adds context to the features.
    :param audio_file: path to audio file.
    :param window_size: window size in seconds.
    :param hop_size: hop size (frame shift) in seconds.
    :param apply_fbank: flag which indicates if a mel filterbank should be used (default: True).
    :param n_filters: number of filters in mel filterbank (default: 24).
    :param fbank_fmin: minimum frequency covered by mel filterbank in Hz (default: 0).
    :param fbank_fmax: maximum frequency covered by mel filterbank in Hz (default: 8000).
    :param num_ceps: number of cepstral coefficients
    :param left_context: Number of predecessors.
    :param right_context: Number of succcessors.
    :return: Features with context.
    """
    # TODO  Implementieren Sie hier Aufgabe 7.2

    # load and read audio signal
    sampling_rate, signal = scipy.io.wavfile.read(audio_file)

    # normalize audio signal to -1 to 1
    signal_norm = signal / max(signal)

    # framing
    frames = make_frames(signal_norm, sampling_rate, window_size, hop_size)

    # compute absolute spectrum with STFT
    STFT = compute_absolute_spectrum(frames)

    filterbank = get_mel_filters(sampling_rate, window_size, n_filters, f_min=fbank_fmin, f_max=fbank_fmax)

    FBANK = apply_mel_filters(STFT, filterbank)
    FBANK = np.log(abs(FBANK))

    if feature_type == 'STFT':

        STFT = add_context(STFT, left_context=left_context, right_context=right_context)

        return STFT

    elif feature_type == 'FBANK':

        FBANK = add_context(FBANK, left_context=left_context, right_context=right_context)
        return FBANK

    else:
        # Compute Cepstral Coefficients
        MFCC = compute_cepstrum(FBANK, num_ceps)

        if feature_type == 'MFCC':

            MFCC = add_context(MFCC, left_context=left_context, right_context=right_context)

            return MFCC

        elif feature_type == 'MFCC_D':
            # Compute derivative of x_cep
            delta_x = get_delta(MFCC)
            MFCC_D = append_delta(MFCC, delta_x)

            MFCC_D = add_context(MFCC_D, left_context=left_context, right_context=right_context)

            return MFCC_D

        elif feature_type == 'MFCC_D_DD':

            # Compute derivative of x_cep
            delta_x = get_delta(MFCC)
            # Compute Second Derivative
            delta_delta_x = get_delta(delta_x)

            MFCC_D = append_delta(MFCC, delta_x)
            MFCC_D_DD = append_delta(MFCC_D, delta_delta_x)

            MFCC_D_DD = add_context(MFCC_D_DD, left_context=left_context, right_context=right_context)

            return MFCC_D_DD


def add_context(feats, left_context=6, right_context=6):
    """
    Adds context to the features.
    :param feats: extracted features of size (n x d) array of features, where n is the number of frames and d is the
           feature dimension.
    :param left_context: Number of predecessors.
    :param right_context: Number of successors.
    :return: Features with context of size (n x d x c), where c = left_context + right_context + 1
    """

    feats_pre = feats
    feats_post = feats

    for i in range(left_context):
        feats_pre = np.roll(feats_pre, 1, axis=0)
        feats_pre[0, :] = feats_pre[1, :]
        feats = np.dstack((feats_pre, feats))

    for m in range(right_context):
        feats_post = np.roll(feats_post, -1, axis=0)
        feats_post[-1, :] = feats_post[-2, :]
        feats = np.dstack((feats, feats_post))

    return feats