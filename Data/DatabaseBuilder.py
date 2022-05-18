import numpy as np
import roomacoustics.room_builder as rb
import database_helper.data_helper as dh
from keras_data_generator import make_frames
import os
import glob
import random as rd
import soundfile as sf
from tqdm import tqdm
import math
from scipy.signal import butter, filtfilt, sosfilt
import matplotlib.pyplot as plt
import json
import time
import pyroomacoustics as pra
from database_helper.tools import subtract_elements_in_list


def reconstruct_speech(prediction, window, winlen, step):

    prediction = np.squeeze(prediction)
    step = int(step)
    winlen = int(winlen)

    num_f = len(prediction)

    # Initialise output signal in time domain.
    output_len = int((step * num_f) + winlen)

    out_sig = np.zeros((output_len,))

    # Generate window function
    if window.upper() == "COSINE":
        w = np.hamming(winlen)
    elif window.upper() == "HANN":
        w = 1 / np.hanning(winlen)
    elif window.upper() == "HAMMING":
        w = 1 / np.hamming(winlen)
    elif window.upper() == "HALF":
        w = np.ones(winlen) * 1
    else:
        w = np.ones(winlen)

    ini_frame = 0
    end_frame = winlen

    for j in range(num_f):
        out_sig[ini_frame:end_frame] = out_sig[ini_frame:end_frame] + prediction[j, :] * w
        ini_frame += step
        end_frame += step

    return out_sig


def butter_lowpass_filter(data, fs):
    cutoff = 100  # desired cutoff frequency of the filter, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 10

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients

    sos = butter(order, normal_cutoff, 'hp', fs=fs, output='sos')

    filtered = sosfilt(sos, data)

    return filtered


def get_random_room(n_samples, num_speakers):

    range_start_x = 5
    range_end_x = 10
    range_start_y = 5
    range_end_y = 10
    range_start_z = 2.5
    range_end_z = 5.5

    max_t60 = 1
    max_volume = range_end_x * range_end_y * range_end_z

    speaker_pos = np.zeros((n_samples, num_speakers), dtype=np.ndarray)

    distr_list = []
    t60_list = []
    for i in range(n_samples):
        x = rd.uniform(range_start_x, range_end_x)
        y = rd.uniform(range_start_y, range_end_y)
        z = rd.uniform(range_start_z, range_end_z)

        distr_list.append([x, y, z])

        volume_room = x*y*z
        t60_list.append(max_t60 *(volume_room/max_volume))

        for m in range(num_speakers):
            speaker_pos[i, m] = np.array(get_speaker_position(x, y, 1.5))

    return distr_list, t60_list, speaker_pos


def get_speaker_position(x, y, z):

    x_new = np.random.uniform(0, x)
    y_new = np.random.uniform(0, y)
    z_new = 1.5

    return [x_new, y_new, z_new]


def get_random_speakers(n_samples):

    speaker_set = [{'Speaker1': 'M', 'Speaker2': 'M'}, {'Speaker1': 'M', 'Speaker2': 'F'}, {'Speaker1': 'F', 'Speaker2': 'M'}, {'Speaker1': 'F', 'Speaker2': 'F'}]

    return np.random.choice(speaker_set, n_samples)


def cut_and_add_acoustic_scenario(clean, near_end, far_end, audio_speaker2, sampling_rate):

    near_end = np.squeeze(near_end)
    far_end = np.squeeze(far_end)
    clean = np.squeeze(clean)
    clean_T = np.squeeze(audio_speaker2)

    if sampling_rate == 8000:
        near_end = near_end[::2]
        far_end = far_end[::2]
        clean = clean[::2]
        clean_T = clean_T[::2]

    elif sampling_rate == 16000:
        near_end = near_end
        far_end = far_end
        clean = clean
        clean_T = clean_T

    min_length = np.min([len(near_end), len(clean), len(far_end), len(clean_T)])

    if min_length > sampling_rate*6:
        min_length = sampling_rate*6

    near_end = near_end[:min_length]
    far_end = far_end[:min_length]
    clean = clean[:min_length]
    clean_T = clean_T[:min_length]

    near_end = (near_end - np.mean(near_end)) / np.max(near_end)
    far_end = (far_end - np.mean(far_end)) / np.max(far_end)
    clean = (clean - np.mean(clean)) / np.max(clean)
    clean_T = (clean_T - np.mean(clean_T)) / np.max(clean_T)

    return near_end, clean, far_end, clean_T


def get_audiofiles_from_acoustic_environment(index_acoustic_set):

    parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent_path, 'NoisyOverlappingSpeakers', 'Inside')

    #path = 'C:/Users/silas/NoisyOverlappingSpeakers/NoisyOverlappingSpeakers/Inside'

    scenarios = list(os.walk(path))

    one_scenario = scenarios[index_acoustic_set]

    path_audio_dir = os.path.join(path, one_scenario[0])

    environment_audio = glob.glob(path_audio_dir + "/*.wav")

    env_audio, fs = sf.read(environment_audio[0])

    return env_audio, fs


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


def room_builder(audio_speaker1, audio_speaker2, t60, room_dim, speaker1_pos, speaker2_pos):

    fs = 16000
    audio_speaker1 = (audio_speaker1 / np.max(abs(audio_speaker1))) * 0.1
    audio_speaker2 = (audio_speaker2 / np.max(abs(audio_speaker2))) * 0.1
    delay = 0

    room = pra.ShoeBox(room_dim, fs=fs)

    room.add_source(speaker1_pos, signal=audio_speaker1)
    room.add_source(speaker2_pos, signal=audio_speaker2, delay=delay)

    room.add_microphone(subtract_elements_in_list(speaker1_pos, [0, 0, 0.2]))
    room.add_microphone(subtract_elements_in_list(speaker2_pos, [0, 0, 0.2]))
    room.simulate()

    near_end = room.mic_array.signals[0]
    far_end = room.mic_array.signals[1]

    return near_end, far_end


def room_builder_more_speakers(audio_speaker_list, t60, room_dim, speaker_pos_list):

    fs = 16000
    room = pra.ShoeBox(room_dim, fs=fs)

    for i in range(len(audio_speaker_list)):
        one_speaker = (audio_speaker_list[i] / np.max(abs(audio_speaker_list[i]))) * 0.1
        if i == 1:
            room.add_source(speaker_pos_list[i], signal=one_speaker)

    room.add_microphone(subtract_elements_in_list(speaker_pos_list[0], [0, 0, 0.2]))
    room.add_microphone(subtract_elements_in_list(speaker_pos_list[1], [0, 0, 0.2]))
    room.simulate()

    near_end = room.mic_array.signals[0]
    far_end = room.mic_array.signals[1]

    return near_end, far_end


if __name__ == '__main__':

    n_samples = 1
    fs = 16000
    num_speakers = 2

    path = os.path.join('C:\\', 'Users', 'silas', 'Documents', 'Python Scripts', 'LibriSpeech')

    # Build Distributions

    room_distr,  t60_list, speaker1_pos = get_random_room(n_samples, num_speakers)

    speaker_set = get_random_speakers(n_samples)

    counter = 0

    # Iterate through data and build dataset
    while counter < n_samples:

        print('Sample {} of {}'.format(counter, n_samples))
        # Set of speakers
        dataset = dh.LibriData(speaker_set[counter], path)

        audio_speaker_list, fs = dataset.load_random_speaker_files(4)

        for i in range(len(audio_speaker_list)):
            audio_speaker_list[i] = np.squeeze(audio_speaker_list[i].detach().cpu().numpy())

        #room = rb.PyRoomBuilder(audio_speaker1, audio_speaker2, t60_list[counter], room_distr[counter], speaker1_pos[counter], speaker2_pos[counter])

        # audio_far_end = torch.tensor(room.get_audio_output('FarEnd'))
        #audio_near_end = room.get_audio_output('NearEnd')
        #audio_far_end = room.get_audio_output('FarEnd')

        audio_near_end, audio_far_end = room_builder_more_speakers(audio_speaker_list, t60_list[counter], room_distr[counter], speaker1_pos[counter])

        complete_scenario, clean, far_end, clean_T = cut_and_add_acoustic_scenario(audio_speaker_list[0], audio_near_end, audio_far_end, audio_speaker_list[1], sampling_rate=fs)

        filtered_farend = butter_lowpass_filter(far_end, fs)
        filtered_clean = butter_lowpass_filter(clean, fs)
        filtered_completeScenario = butter_lowpass_filter(complete_scenario, fs)
        filtered_clean_T = butter_lowpass_filter(clean_T, fs)

        if np.all(filtered_farend) and np.all(filtered_clean) and np.all(filtered_completeScenario) and np.all(filtered_clean_T) and not np.isnan(filtered_farend).any() and not np.isnan(filtered_clean).any() and not np.isnan(filtered_completeScenario).any() and not np.isnan(filtered_clean_T).any():

            sf.write('C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase_Silence/SpeakerSet_NearEnd{}.wav'.format(counter), filtered_completeScenario, fs)
            sf.write('C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase_Silence/SpeakerSet_CleanSpeech{}.wav'.format(counter), filtered_clean, fs)
            sf.write('C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase_Silence/SpeakerSet_FarEnd{}.wav'.format(counter), filtered_farend, fs)
            sf.write('C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase_Silence/SpeakerSet_CleanFarEnd{}.wav'.format(counter), clean_T, fs)

            counter += 1

    print('Database Finished')



