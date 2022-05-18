import glob
import tensorflow as tf
import os
import scipy.io.wavfile as sc
import numpy as np
from scipy.signal import butter, filtfilt, sosfilt


def butter_lowpass_filter(data, fs):
    cutoff = 70  # desired cutoff frequency of the filter, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 10

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients

    sos = butter(order, normal_cutoff, 'hp', fs=fs, output='sos')

    filtered = sosfilt(sos, data)

    return filtered


class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self, clean_dir, transform=None, target_transform=None):
        self.clean_dir = clean_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return int(len(glob.glob(self.clean_dir + "/*.wav")) / 2)

    def __getitem__(self, idx):
        target_speaker_path = os.path.join(self.clean_dir, 'SpeakerSet_CleanSpeech{}.wav'.format(idx))
        speech_mix_path = os.path.join(self.clean_dir, 'SpeakerSet_NearEnd{}.wav'.format(idx))

        target_speaker = path_to_audio(target_speaker_path)
        speech_mix = path_to_audio(speech_mix_path)

        target_speaker = tf.squeeze(tf.signal.frame(target_speaker, 512, 256, pad_end=False, pad_value=0, axis=-1, name=None))
        speech_mix = tf.squeeze(tf.signal.frame(speech_mix, 512, 256, pad_end=False, pad_value=0, axis=-1, name=None))

        dataset = tf.data.Dataset

        return [speech_mix, target_speaker]


def load_training_batch(clean_dir, sample_list, window, hop, sampling_rate=8000, mode='train'):

    target_speaker_list = []
    speech_mix_list = []
    clean_speaker_list = []
    clean_farend_list = []

    for i in range(len(sample_list)):
        clean_speaker_path = os.path.join(clean_dir, 'SpeakerSet_CleanSpeech{}.wav'.format(sample_list[i]))
        target_speaker_path = os.path.join(clean_dir, 'SpeakerSet_FarEnd{}.wav'.format(sample_list[i]))
        speech_mix_path = os.path.join(clean_dir, 'SpeakerSet_NearEnd{}.wav'.format(sample_list[i]))
        target_farend_path = os.path.join(clean_dir, 'SpeakerSet_CleanFarEnd{}.wav'.format(sample_list[i]))

        target_speaker = path_to_audio(target_speaker_path, sampling_rate)
        target_farend = path_to_audio(target_farend_path, sampling_rate)
        speech_mix = path_to_audio(speech_mix_path, sampling_rate)
        clean_speaker = path_to_audio(clean_speaker_path, sampling_rate)

        target_speaker = (target_speaker - np.mean(target_speaker)) / np.max(np.abs(target_speaker))
        target_farend = (target_farend - np.mean(target_farend)) / np.max(np.abs(target_farend))
        speech_mix = (speech_mix - np.mean(speech_mix)) / np.max(np.abs(speech_mix))
        clean_speaker = (clean_speaker - np.mean(clean_speaker)) / np.max(np.abs(clean_speaker))

        if np.isnan(target_speaker).any() or np.isnan(speech_mix).any() or np.isnan(clean_speaker).any() or np.isnan(target_farend).any():
            print('Faulty Input File Detected, Skipping')
        else:

            target_speaker = make_frames(target_speaker, window, hop, "HALF")
            speech_mix = make_frames(speech_mix, window, hop, "HALF")
            clean_speaker = make_frames(clean_speaker, window, hop, "HALF")
            clean_farend = make_frames(target_farend, window, hop, "HALF")

            target_speaker_list.append(target_speaker)
            speech_mix_list.append(speech_mix)
            clean_speaker_list.append(clean_speaker)
            clean_farend_list.append(clean_farend)

    if mode == 'train':
        target_speaker_list = np.concatenate([np.array(i) for i in target_speaker_list])
        speech_mix_list = np.concatenate([np.array(i) for i in speech_mix_list])
        clean_speaker_list = np.concatenate([np.array(i) for i in clean_speaker_list])
        clean_farend_list = np.concatenate([np.array(i) for i in clean_farend_list])
        # Reference: Target = FarEnd, SpeechMix = NearEnd, Clean = Clean
        return np.expand_dims(np.asarray(target_speaker_list), axis=1), np.expand_dims(np.asarray(speech_mix_list), axis=1), np.expand_dims(np.asarray(clean_speaker_list), axis=1), np.expand_dims(np.asarray(clean_farend_list), axis=1)
    else:
        return np.expand_dims(np.asarray(target_speaker_list), axis=2), np.expand_dims(np.asarray(speech_mix_list), axis=2), np.expand_dims(np.asarray(clean_speaker_list), axis=2), np.expand_dims(np.asarray(clean_farend_list), axis=2)


def construct_dataset(audio_dir):

    audio_paths = glob.glob(audio_dir + "/*.wav")

    clean_paths, near_paths = split_paths(audio_paths)

    path_ds_clean = tf.data.Dataset.from_tensor_slices(clean_paths)
    path_ds_near = tf.data.Dataset.from_tensor_slices(near_paths)

    audio_NE_ds = path_ds_near.map(lambda x: path_to_audio(x))
    audio_CS_ds = path_ds_clean.map(lambda x: path_to_audio(x))
    #audio_CS_ds = path_ds_clean.map(lambda x, y: path_to_audio(x, y))

    return audio_CS_ds, audio_NE_ds
    #return audio_CS_ds


def path_to_audio(path1, sampling_rate):
    """Reads and decodes an audio file."""
    fs, audio = sc.read(path1)

    if sampling_rate == 8000:
        audio = audio[::2]

    #audio1 = tf.io.read_file(path2)
    #audio1, _ = tf.audio.decode_wav(audio1, 1)

    #return audio, audio1
    return audio


def split_paths(path):

    clean_speech = []
    near_end = []

    for i in range(len(path)):
        if 'CleanSpeech' in os.path.basename(path[i]):
            clean_speech.append(path[i])
        else:
            near_end.append(path[i])

    return clean_speech, near_end


def make_frames(audio_data, window_size, hop_size, window):

    siglen = len(audio_data)
    step = int(hop_size)
    winlen = int(window_size)

    if window.upper() == "HAMMING":
        w = np.hamming(winlen)
    elif window.upper() == "HANN":
        w = np.hanning(winlen)
    elif window.upper() == "COSINE":
        w = np.sqrt(np.hamming(winlen))
    elif window.upper() == "NONE":
        w = np.ones(winlen)
    elif window.upper() == "HALF":
        w = np.ones(winlen)*1

    nWins = int(np.floor((siglen - winlen) / step)) + 1

    winSig = np.zeros((winlen, nWins))

    winIdxini = 0
    winIdxend = winlen

    for i in range(0, nWins):
        winSig[:, i] = np.multiply(audio_data[winIdxini:winIdxend], w)
        winIdxini += step
        winIdxend += step

    return winSig.T


def audiofiles_without_impulse(nearend1, farend1, win_length):

    fs = 16000

    nearend1 = np.squeeze(nearend1)
    farend1 = np.squeeze(farend1)

    min_length = np.min([len(nearend1), len(farend1)])

    if min_length > fs * 6:
        min_length = fs * 6

    nearend = nearend1[:min_length]
    farend = farend1[:min_length]

    clean_nearend = (nearend - np.mean(nearend)) / np.max(np.abs(nearend))
    clean_farend = (farend - np.mean(farend)) / np.max(np.abs(farend))

    speech_mix_near = 0.8 * clean_nearend + 0.3 * clean_farend
    speech_mix_near = (speech_mix_near - np.mean(speech_mix_near)) / np.max(np.abs(speech_mix_near))

    speech_mix_far = 0.1 * clean_nearend + 0.8 * clean_farend
    speech_mix_far = (speech_mix_far - np.mean(speech_mix_far)) / np.max(np.abs(speech_mix_far))

    if np.isnan(speech_mix_near).any() or np.isnan(speech_mix_far).any() or np.isnan(clean_nearend).any() or np.isnan(clean_farend).any():
        print('Faulty Input File Detected, Skipping')
    else:

        speech_mix_near = make_frames(speech_mix_near, win_length, win_length, "HALF")
        speech_mix_far = make_frames(speech_mix_far, win_length, win_length, "HALF")
        clean_nearend = make_frames(clean_nearend, win_length, win_length, "HALF")
        clean_farend = make_frames(clean_farend, win_length, win_length, "HALF")

    return np.expand_dims(np.asarray(speech_mix_far), axis=1), np.expand_dims(np.asarray(speech_mix_near), axis=1), np.expand_dims(np.asarray(clean_nearend), axis=1), np.expand_dims(np.asarray(clean_farend), axis=1)


def load_file(path):

    audio = path_to_audio(path, 16000)
    audio = (audio - np.mean(audio)) / np.max(abs(audio))
    audio = make_frames(audio, 1024, 1024, 'HALF')

    return audio


if __name__ == '__main__':

    batches = 100
    num_files_in_batch = 20000//batches
    clean_dir = 'C:/Users/silas/NoisyOverlappingSpeakers/Database'

    for i in range(batches):
        print('Batch: {} of 100'.format(i))

        writer = tf.io.TFRecordWriter('F:/TFRecordTraining/TrainBatch_{}.tfrecord'.format(i))
        for m in range(num_files_in_batch):

            current_sample = i*num_files_in_batch + m
            print('Sample: {}'.format(current_sample))
            clean_speaker_path = os.path.join(clean_dir, 'SpeakerSet_CleanSpeech{}.wav'.format(current_sample))
            target_speaker_path = os.path.join(clean_dir, 'SpeakerSet_FarEnd{}.wav'.format(current_sample))
            speech_mix_path = os.path.join(clean_dir, 'SpeakerSet_NearEnd{}.wav'.format(current_sample))

            clean = load_file(clean_speaker_path)
            near = load_file(speech_mix_path)
            far = load_file(target_speaker_path)

            for k in range(len(clean)):
                features = {
                    'clean': tf.train.Feature(float_list=tf.train.FloatList(value=clean[k])),
                    'near': tf.train.Feature(float_list=tf.train.FloatList(value=near[k])),
                    'far': tf.train.Feature(float_list=tf.train.FloatList(value=far[k]))
                }

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())

        writer.close()