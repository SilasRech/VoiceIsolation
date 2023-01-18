import pyroomacoustics as pra
import numpy as np


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
        self.audio_speaker1 = audio_speaker1*0.1
        self.audio_speaker2 = audio_speaker2*0.1

        #e_absorption, max_order = pra.inverse_sabine(self.rt60, self.room_dim)
        room = pra.ShoeBox(self.room_dim, fs=self.fs)

        room.add_source(self.pos_speaker1, signal=self.audio_speaker1)
        room.add_source(self.pos_speaker2, signal=self.audio_speaker2)

        room.add_microphone(list(np.asarray(self.pos_speaker1) - np.asarray([0, 0, 0.2])))
        room.add_microphone(list(np.asarray(self.pos_speaker2) - np.asarray([0, 0, 0.2])))

        self.room = room

    def get_audio_output(self):

        self.room.simulate()

        return self.room.mic_array.signals