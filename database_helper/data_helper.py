import soundfile as sf
import os
import numpy as np
from torch.utils.data import Dataset
import torchaudio
from database_helper.tools import transform_speaker_text
from database_helper.tools import walk_through_files


class LibriData(Dataset):
    """
    Helper Function for laoding the audio files from libri dataset
    path: Path to root folder of libri dataset
    speakers_distr: What speakers are selected for transmission testing (m, m), (f, m), (m, f), (f, f)

    """

    def __init__(self, speakers_distr: dict=None, path: str=None):
        self.data = {}
        self.path = path
        self.speakers_distr = speakers_distr

        self.sample_rate = 16000
        self.dataset = torchaudio.datasets.LIBRISPEECH(self.path, url="dev-clean")

    def __len__(self) -> int:
        """Returns length of dataset.
        Returns:
            int: Length of dataset
        """
        return len(self.data)

    def load_random_speaker_files(self):
        """
        Randomly selects speakers 1 and 2 sample based on the distribution of speakers

        :return: idx for speakers
        """

        speaker_list = self.get_speakers_from_subset(self.path, 'dev-clean')

        speaker1 = speaker_list.loc[speaker_list['SEX'] == self.speakers_distr['Speaker1'].capitalize()].sample()
        speaker2 = speaker_list.loc[speaker_list['SEX'] == self.speakers_distr['Speaker2'].capitalize()].sample()

        index_speaker1 = int(speaker1['ID'].values)
        index_speaker2 = int(speaker2['ID'].values)

        audio_speaker1, fs = self.get_audiofiles_from_libri(index_speaker1)
        audio_speaker2, fs = self.get_audiofiles_from_libri(index_speaker2)

        self.sample_rate = fs

        return audio_speaker1, audio_speaker2, fs

    def get_speakers_from_subset(self, path, subset):

        with open(os.path.join(path, 'speakers.txt')) as f:
            meta_speakers = f.readlines()

        return transform_speaker_text(meta_speakers, subset)

    def get_audiofiles_from_libri(self, index_speaker):

        speaker_path = os.path.join(self.path, "dev-clean", str(index_speaker))

        audiofile_path = walk_through_files(speaker_path)

        audio_file_speaker1, fs = torchaudio.load(audiofile_path)

        return audio_file_speaker1, fs


def loader():

    audio_file1, fs = torchaudio.load('./Data/1919-142785-0007.flac')
    audio_file2, fs = torchaudio.load('./Data/6313-66125-0027.flac')

    return audio_file1, audio_file2, fs

