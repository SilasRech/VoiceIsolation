import numpy as np

import roomacoustics.room_builder as rb
import database_helper.data_helper as dh

from torch.utils.data import Dataset
import torchaudio
import torch
import os
import glob
import random as rd
from HelperFunctions import OA_Windowing, make_frames
from database_helper.tools import walk_through_files
import soundfile as sf
from tqdm import tqdm


class NoisyOverlappingSpeakerSet(Dataset):
    def __init__(self, clean_dir, transform=None, target_transform=None):
        self.clean_dir = clean_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return int(len(glob.glob(self.clean_dir + "/*.wav"))/2)

    def __getitem__(self, idx):
        target_speaker_path = os.path.join(self.clean_dir, 'SpeakerSet_NearEnd{}.wav'.format(idx))
        speech_mix_path = os.path.join(self.clean_dir, 'SpeakerSet_CleanSpeech{}.wav'.format(idx))

        target_speaker = read_audio(target_speaker_path)
        speech_mix = read_audio(speech_mix_path)

        return speech_mix, target_speaker


def read_audio(audio_dir):
    audio_file, fs = torchaudio.load(audio_dir)

    audio_file = torch.nn.functional.normalize(audio_file, dim=1)
    #audio_file = audio_file - torch.mean(audio_file)

    #audio_file = OA_Windowing(audio_file, 'HAMMING', fs*0.032, fs*0.016).transpose(0, 1)

    return audio_file


def get_random_room(n_samples):

    range_start_x = 5
    range_end_x = 12
    range_start_y = 5
    range_end_y = 12
    range_start_z = 2.5
    range_end_z = 5.5

    max_t60 = 1
    max_volume = range_end_x * range_end_y * range_end_z

    speaker1_pos = []
    speaker2_pos = []

    distr_list = []
    t60_list = []
    for i in range(n_samples):
        x = rd.uniform(range_start_x, range_end_x)
        y = rd.uniform(range_start_y, range_end_y)
        z = rd.uniform(range_start_z, range_end_z)

        distr_list.append([x, y, z])

        volume_room = x*y*z
        t60_list.append(max_t60* (volume_room/max_volume))

        speaker1_pos.append(get_speaker_position(x, y, 1.5))
        speaker2_pos.append(get_speaker_position(x, y, 1.5))

    return distr_list, t60_list, speaker1_pos, speaker2_pos


def get_speaker_position(x, y,z):

    x_new = np.random.uniform(0, x)
    y_new = np.random.uniform(0, y)
    z_new = np.random.uniform(0, z)

    return [x_new, y_new, z_new]


def get_random_speakers(n_samples):

    speaker_set = [{'Speaker1': 'M', 'Speaker2': 'M'}, {'Speaker1': 'M', 'Speaker2': 'F'}, {'Speaker1': 'F', 'Speaker2': 'M'}, {'Speaker1': 'F', 'Speaker2': 'F'}]

    return np.random.choice(speaker_set, n_samples)


def cut_and_add_acoustic_scenario(clean, near_end, scenario):

    near_end = np.squeeze(near_end)
    clean = np.squeeze(clean)

    env_audio, fs = get_audiofiles_from_acoustic_environment(scenario)
    env_audio = np.squeeze(env_audio)

    min_length = np.min([len(near_end), len(clean), len(env_audio)])

    if min_length > fs*6:
        min_length = fs*6

    near_end = near_end[:min_length]
    clean = clean[:min_length]
    env_audio = env_audio[:min_length]/np.max(abs(env_audio))*0.8

    speaker_mix = np.add(near_end, clean)

    return np.add(speaker_mix, env_audio), clean


def get_audiofiles_from_acoustic_environment(index_acoustic_set):

    path = os.path.join('C:\\', 'Users', 'silas', 'NoisyOverlappingSpeakers', 'NoisyOverlappingSpeakers', 'Inside')

    scenarios = list(os.walk(path))

    one_scenario = scenarios[index_acoustic_set]

    path_audio_dir = os.path.join(path, one_scenario[0])

    environment_audio = glob.glob(path_audio_dir + "/*.wav")

    env_audio, fs = sf.read(environment_audio[0])

    return env_audio, fs


def test_audio(audio_speaker1, audio_speaker2):

    fs = 16000
    room_distr, t60_list, speaker1_pos, speaker2_pos = get_random_room(1)
    scenario = np.random.randint(1, 7, 1)

    audio_speaker1 = np.squeeze(audio_speaker1.detach().cpu().numpy())
    audio_speaker2 = np.squeeze(audio_speaker2.detach().cpu().numpy())

    room = rb.PyRoomBuilder(audio_speaker1, audio_speaker2, t60_list[0], room_distr[0], speaker1_pos[0],
                            speaker2_pos[0])

    # audio_far_end = torch.tensor(room.get_audio_output('FarEnd'))
    audio_near_end = room.get_audio_output('NearEnd')

    complete_scenario, clean = cut_and_add_acoustic_scenario(audio_speaker1, audio_near_end, scenario[0])

    clean = torch.unsqueeze(torch.from_numpy(clean).float().cuda(), 0)
    clean = OA_Windowing(clean, 'HAMMING', fs * 0.032, fs * 0.016)
    clean = torch.unsqueeze(clean, 1)

    complete_scenario = torch.unsqueeze(torch.from_numpy(complete_scenario).float().cuda(), 0)
    complete_scenario = OA_Windowing(complete_scenario, 'HAMMING', fs * 0.032, fs * 0.016)
    complete_scenario = torch.unsqueeze(complete_scenario, 1)

    return complete_scenario, clean


if __name__ == '__main__':

    n_samples = 5
    fs = 16000

    path = os.path.join('C:\\', 'Users', 'silas', 'Documents', 'Python Scripts', 'LibriSpeech')

    # Build Distributions

    room_distr, t60_list, speaker1_pos, speaker2_pos = get_random_room(n_samples)
    # There are 7 possible scenarios
    scenario = np.random.randint(1, 7, n_samples)

    speaker_set = get_random_speakers(n_samples)

    counter = 0

    # Iterate through data and build dataset
    for i in tqdm(range(n_samples)):

        # Set of speakers
        dataset = dh.LibriData(speaker_set[i], path)

        audio_speaker1, audio_speaker2, fs = dataset.load_random_speaker_files()

        # Init Room for Impulse Responses

        audio_speaker1 = np.squeeze(audio_speaker1.detach().cpu().numpy())
        audio_speaker2 = np.squeeze(audio_speaker2.detach().cpu().numpy())

        room = rb.PyRoomBuilder(audio_speaker1, audio_speaker2, t60_list[i], room_distr[i], speaker1_pos[i], speaker2_pos[i])

        #audio_far_end = torch.tensor(room.get_audio_output('FarEnd'))
        audio_near_end = room.get_audio_output('NearEnd')

        complete_scenario, clean = cut_and_add_acoustic_scenario(audio_speaker1, audio_near_end, scenario[i])

        complete_scenario = make_frames(complete_scenario, fs, 0.032, 0.016)
        clean = make_frames(clean, fs,  0.032, 0.016)

        for k in range(len(clean)):
            sf.write('C:/Users/silas/NoisyOverlappingSpeakers/Database/SpeakerSet_NearEnd{}.wav'.format(counter), complete_scenario[k, :], fs)#

            sf.write('C:/Users/silas/NoisyOverlappingSpeakers/Database/SpeakerSet_CleanSpeech{}.wav'.format(counter),
                    clean[k, :], fs)
            counter += 1
        #sf.write('C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase/SpeakerSet_NearEnd{}.wav'.format(i), complete_scenario, fs)

        #sf.write('C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase/SpeakerSet_CleanSpeech{}.wav'.format(i),
       #             clean, fs)




    print('Database Finished')
# Random Roomsize DONE
# Random Environment Setting DONE
# Choose Random Speakers DONE
# Add all of them
# Save in folder


