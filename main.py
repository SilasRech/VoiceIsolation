from database_helper import data_helper
from roomacoustics import room_builder
import voice_isolation as vi
import echo_canceller.echo_can as ec
import os
import numpy as np
import torchaudio
import torch
import scipy.io.wavfile as sc
import pytorch_lightning as pl
import asteroid
from asteroid.models import BaseModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import DataLoader, random_split

path = os.path.join('C:\\','Users', 'silas', 'Documents','Python Scripts', 'LibriSpeech')

#Define audio backend to enable use of torchaudio
if os.name == 'nt':
    torchaudio.set_audio_backend('soundfile')
elif os.name == 'posix':
    torchaudio.set_audio_backend('sox_io')
else:
    print('OS not supported by torchaudio')
    exit()


libri_data = torchaudio.datasets.LIBRISPEECH(path, download=False)
data_loader = torch.utils.data.DataLoader(libri_data,
                                          batch_size=1,
                                          shuffle=True)

model = BaseModel.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean")

if __name__ == '__main__':

    # Default for Development = False to have static audiofiles
    load_random_speaker = False

    parameters = {
        'speaker_distr' : {'Speaker1': 'M', 'Speaker2': 'M'},
    }

    # Load audiofiles for testing
    if load_random_speaker:
        dataset = data_helper.LibriData(parameters['speaker_distr'], path)
        audio_speaker1, audio_speaker2, fs = dataset.load_random_speaker_files()
    else:
        audio_speaker1, audio_speaker2, fs = data_helper.loader()

    # Init Room for Impulse Responses
    parameters_room = {
        'rt60': 0.6,
        'room_dim': [10, 7.5, 3.5],
        'fs': fs,
        'pos_speaker1': [2.5, 4.5, 1.6],
        'pos_speaker2': [1.5, 2, 1.6],
    }

    audio_speaker1 = np.squeeze(audio_speaker1.detach().cpu().numpy())
    audio_speaker2 = np.squeeze(audio_speaker2.detach().cpu().numpy())

    room = room_builder.PyRoomBuilder(audio_speaker1, audio_speaker2, parameters_room)

    audio_far_end = torch.tensor(room.get_audio_output('FarEnd'))
    audio_near_end = torch.tensor(room.get_audio_output('NearEnd'))

    #audio_near_end = torch.squeeze(audio_near_end)
    #audio_near_end = torch.unsqueeze(audio_near_end, dim=-1).cpu().detach().numpy()
    audio_near_end = audio_near_end.float()
    outwave = model.separate(audio_near_end)

    #test1 = outwave.cpu().detach().numpy()
    test1 = outwave.cpu().detach().numpy()[0, 0, :]
    test2 = outwave.cpu().detach().numpy()[0, 1, :]

    sc.write('data/TestSeparation.wav', fs,  test1)
    sc.write('data/TestSeparation1.wav', fs,  test2)

    #Echo Cancellation
    winlen = int(0.04 * fs)
    winstep = int(0.02 * fs)
    nfft = winlen
    n_delay_blocks = 16

    #train, val = random_split(libri_data, [28500, 39])

    #autoencoder = vi.LitAutoEncoder()
    #autoencoder.summarize()
    #trainer = pl.Trainer()
    #trainer.fit(autoencoder, DataLoader(train), DataLoader(val))

    out_sig = ec.echo_can(torch.reshape(audio_near_end, (-1, 1)), torch.reshape(audio_far_end, (-1, 1)), winlen, winstep, nfft,
                             'FT', n_delay_blocks=n_delay_blocks)

    # torchaudio.save(filepath = os.path.normpath('./Samples/EchoSample.wav'),src = out_sig, sample_rate = fs_target, format = "wav")
    sc.write(os.path.normpath('./data/output/EchoSample2.wav'), fs, out_sig.numpy())

    test = 1

