import pyroomacoustics as pra
import numpy as np
from database_helper.tools import subtract_elements_in_list


class PyRoomBuilder():
    """
    Helper Function for laoding the audio files from libri dataset
    path: Path to root folder of libri dataset
    speakers_distr: What speakers are selected for transmission testing (m, m), (f, m), (m, f), (f, f)

    """

    def __init__(self, audio_speaker1, audio_speaker2, t60, room_dim, speaker1_pos, speaker2_pos):
        self.rt60 = t60
        self.room_dim = room_dim
        self.fs = 16000
        self.pos_speaker1 = speaker1_pos
        self.pos_speaker2 = speaker2_pos
        self.audio_speaker1 = (audio_speaker1/np.max(abs(audio_speaker1))) * 0.1
        self.audio_speaker2 = (audio_speaker2/np.max(abs(audio_speaker2))) * 0.1
        self.room = 0
        self.delay = 0.0

        #self.e_absorption, self.max_order = pra.inverse_sabine(self.rt60, self.room_dim)

        room = pra.ShoeBox(self.room_dim, fs=self.fs)

        room.add_source(self.pos_speaker1, signal=self.audio_speaker1)
        room.add_source(self.pos_speaker2, signal=self.audio_speaker2, delay=self.delay)
        self.room = room

    def add_microphone(self, room, speaker_pos):
        room.add_microphone(subtract_elements_in_list(speaker_pos, [0, 0, 0.2]))

    def get_audio_output(self, speaker):

        if speaker == 'NearEnd':
            self.add_microphone(self.room, self.pos_speaker1)
        else:
            self.add_microphone(self.room, self.pos_speaker2)

        self.room.simulate()
        #room.mic_array.to_wav(
        #    "./data/output/Output_For_{}_Speaker.wav".format(speaker),
        #    norm=True,
        #    bitdepth=np.int16,
        #    mono=True,
        #)
        return self.room.mic_array.signals
