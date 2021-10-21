import pyroomacoustics as pra
import numpy as np
from database_helper.tools import subtract_elements_in_list


class PyRoomBuilder():
    """
    Helper Function for laoding the audio files from libri dataset
    path: Path to root folder of libri dataset
    speakers_distr: What speakers are selected for transmission testing (m, m), (f, m), (m, f), (f, f)

    """

    def __init__(self, audio_speaker1: None, audio_speaker2, parameter: dict=None):
        self.rt60 = parameter['rt60']
        self.room_dim = parameter['room_dim']
        self.fs = parameter['fs']
        self.pos_speaker1 = parameter['pos_speaker1']
        self.pos_speaker2 = parameter['pos_speaker2']
        self.audio_speaker1 = audio_speaker1
        self.audio_speaker2 = audio_speaker2

        self.delay = 0.0

        self.e_absorption, self.max_order = pra.inverse_sabine(self.rt60, self.room_dim)

    def build_room(self):
        room = pra.ShoeBox(self.room_dim, fs=self.fs, materials=pra.Material(self.e_absorption),
                                 max_order=self.max_order)

        #room.add_source(self.pos_speaker1, signal=self.audio_speaker1)
        room.add_source(self.pos_speaker2, signal=self.audio_speaker2, delay=self.delay)
        return room

    def add_microphone(self, room, speaker_pos):
        room.add_microphone(subtract_elements_in_list(speaker_pos, [0, 0, 0.1]))

    def get_audio_output(self, speaker):

        room = self.build_room()

        if speaker == 'NearEnd':
            self.add_microphone(room, self.pos_speaker1)
        else:
            self.add_microphone(room, self.pos_speaker2)

        room.simulate()
        room.mic_array.to_wav(
            "./data/output/Output_For_{}_Speaker.wav".format(speaker),
            norm=True,
            bitdepth=np.int16,
            mono=True,
        )
        return room.mic_array.signals
