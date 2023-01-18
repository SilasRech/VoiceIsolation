import torch
from torch.utils.data import Dataset
import torchaudio
import os
import random
import numpy as np
import soundfile as sf
from roomacoustics import PyRoomBuilder


class LibriSpeech(Dataset):
    def __init__(self, file_list, root_dir, noise_files=None, fs=16000, length=4,
                 win_length=2048, hop_length=2048, length_input=16000, batchsize=1, num_speaker=2):
        self.file_list = file_list
        self.root_dir = root_dir
        self.noise_files = noise_files
        self.fs = fs
        self.length = length
        self.win_length = win_length
        self.hop_length = hop_length
        self.input_length = length_input
        self.batchsize = batchsize
        self.num_speaker = num_speaker

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
  
        target_path = self.file_list[idx].split()[0]
        interferer_audio_all = torch.zeros((self.length*self.fs))
        for k in range(1, self.num_speaker):
            interferer_path = self.file_list[idx].split()[k]
            interferer_audio, fs = torchaudio.load(interferer_path, normalize=True)
            interferer_audio = self.cut_and_normalize(interferer_audio)
            interferer_audio = interferer_audio / torch.max(torch.abs(interferer_audio))
            interferer_audio_all += interferer_audio

        target_audio, fs = torchaudio.load(target_path, normalize=True)
        target_audio = self.cut_and_normalize(target_audio)
        target_audio = target_audio / torch.max(torch.abs(target_audio))

        leaked_target, leaked_interferer = self.withSNR(target_audio, interferer_audio_all)

        return self.bandfilter(leaked_target), self.bandfilter(leaked_interferer), self.bandfilter(target_audio)

    def withSNR(self, target_audio, interferer_audio):
        snr_db = np.random.randint(-15, high=-5, size=1)
    
        snr = torch.as_tensor((10 ** (snr_db / 20)), dtype=torch.float)

        target_rms = target_audio.norm(p=2)
        interferer_rms = interferer_audio.norm(p=2)
        scale_target = snr * (interferer_rms / target_rms)
        
        return (scale_target * interferer_audio + target_audio) / 2, (scale_target * target_audio + interferer_audio) / 2


    def withRoomImpulse(self, target_audio, interferer_audio):
        t60 = np.random.uniform(low=0.5, high=1.1, size=1)
        room_dim = [float(np.random.uniform(low=5, high=8, size=1)), float(np.random.uniform(low=5, high=8, size=1)), float(np.random.uniform(low=2.5, high=3.5, size=1))]
        speaker1_pos = [float(np.random.uniform(low=0, high=4.8, size=1)), float(np.random.uniform(low=5, high=4.8, size=1)), 1.5]
        speaker2_pos = [float(np.random.uniform(low=0, high=4.8, size=1)), float(np.random.uniform(low=5, high=4.8, size=1)), 1.5]

        room = PyRoomBuilder(target_audio, interferer_audio, t60, room_dim, speaker1_pos, speaker2_pos)

        simulated_audio = room.get_audio_output()

        leaked_target = torch.from_numpy(simulated_audio[0][:int(self.fs*self.length)]).to(dtype=torch.float32)
        leaked_target = leaked_target / torch.max(torch.abs(leaked_target))

        leaked_interferer = torch.from_numpy(simulated_audio[1][:int(self.fs*self.length)]).to(dtype=torch.float32)
        leaked_interferer = leaked_interferer / torch.max(torch.abs(leaked_interferer))

        leaked_target = torch.reshape(leaked_target, (self.length, self.input_length))
        leaked_interferer = torch.reshape(leaked_interferer, (self.length, self.input_length))

        return leaked_target, leaked_interferer

    def bandfilter(self, audio):

        audio_lowpass = torchaudio.functional.lowpass_biquad(audio, sample_rate=self.fs, cutoff_freq=16000, Q = 0.707)
        audio_highpass = torchaudio.functional.highpass_biquad(audio, sample_rate=self.fs, cutoff_freq=70, Q = 0.707)
        return audio_highpass

    def cut_and_normalize(self, audio_in):
        num_samples = self.length * self.fs
        audio_in = audio_in[:, :int(num_samples)]
        audio_in = torch.squeeze(audio_in)
        audio_nomean = audio_in - torch.mean(audio_in)
        audio_nomean_unitstd = audio_nomean / torch.std(audio_nomean)

        return audio_nomean_unitstd

    def add_noise(self, audio):
        noise_sample = int(np.random.randint(0, high=len(self.noise_files), size=1, dtype=int))
        noise, _ = torchaudio.load(self.noise_files[noise_sample])
        noise = noise[:, : audio.shape[1]]

        snr_db = torch.from_numpy(np.random.randint(3, high=15, size=1, dtype=int))

        speech_rms = audio.norm(p=2)
        noise_rms = noise.norm(p=2)

        snr = 10 ** (snr_db / 20)
        scale = snr * noise_rms / speech_rms

        return (scale * audio + noise) / 2


def walk_through_files(path):
    return_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".flac"):
                return_list.append(os.path.join(root, file))

    list_possible_audio_files = []
    for path in return_list:
        minimum_file_length = 96000
        audiofile, sample_rate = sf.read(path)

        if len(audiofile) >= minimum_file_length:
            list_possible_audio_files.append(path)

    random_sample = random.choice(list_possible_audio_files)

    return random_sample
