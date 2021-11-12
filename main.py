from database_helper import data_helper
import echo_canceller.echo_can as ec
import os
import numpy as np
import torchaudio
import torch
import scipy.io.wavfile as sc
import pytorch_lightning as pl
import asteroid
import Data.DatabaseBuilder
import voice_isolation as vi
from solver import Solver
from HelperFunctions import make_frames
from torch.utils.data import DataLoader
from Data.DatabaseBuilder import test_audio
from HelperFunctions import OA_Windowing, OA_Reconstruct


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import DataLoader, random_split

path = os.path.join('C:\\','Users', 'silas', 'Documents', 'Python Scripts', 'LibriSpeech')

#model = BaseModel.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean")


if __name__ == '__main__':

    parameters = {
        'speaker_distr': {'Speaker1': 'M', 'Speaker2': 'M'},
    }

    load_random_speaker = False
    if load_random_speaker:
        dataset = data_helper.LibriData(parameters['speaker_distr'], path)
        audio_speaker1, audio_speaker2, fs = dataset.load_random_speaker_files()
    else:
        audio_speaker1, audio_speaker2, fs = data_helper.loader()

    complete_scenario, clean = test_audio(audio_speaker1, audio_speaker2)

    #test_mixture = torch.from_numpy(complete_scenario).float().cuda()
    #test_mixture = torch.unsqueeze(test_mixture, 1)

    #test_clean = torch.from_numpy(clean).float().cuda()
    #test_clean = torch.unsqueeze(test_clean, 1)

    path = 'C:/Users/silas/NoisyOverlappingSpeakers/Database/'

    dataset = Data.DatabaseBuilder.NoisyOverlappingSpeakerSet('C:/Users/silas/NoisyOverlappingSpeakers/Database/')
    dataset_eval = Data.DatabaseBuilder.NoisyOverlappingSpeakerSet('C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase/')

    train_dataloader = DataLoader(dataset, batch_size=374, shuffle=False)
    eval_dataloader = DataLoader(dataset_eval, batch_size=374, shuffle=False)

    data = {'tr_loader': train_dataloader, 'eval_loader': eval_dataloader}

    # Load audiofiles for testing
    parameters = {
        'speaker_distr': {'Speaker1': 'M', 'Speaker2': 'M'},
    }

    M = 2  # BatchSize

    N = 512  # enc_dim / dec_dim

    L = int(16000 * 0.032)  # filter size # kernel size

    T = 512  # Samples

    K = 2 * T // L - 1

    B = 128  # Input Channels for ConvLayer
    H, P, X, R, C, norm_type, causal = 3, 3, 3, 2, 1, "gLN", False

    print(f'There are {len(dataset)} samples in the dataset')
    # Default for Development = False to have static audiofiles

    load_random_speaker = False
    if load_random_speaker:
        dataset = data_helper.LibriData(parameters['speaker_distr'], path)
        audio_speaker1, audio_speaker2, fs = dataset.load_random_speaker_files()
    else:
        audio_speaker1, audio_speaker2, fs = data_helper.loader()

    # model
    model = vi.ConvTasNet(N, L, B, H, P, X, R,
                       C, norm_type=norm_type, causal=causal)
    #print(model)

    optimizier = torch.optim.Adam(model.parameters(),
                                      lr=1e-3,
                                      weight_decay=1e-3)

    # solver
    solver = Solver(data, model, optimizier, epochs=1)
    solver.train()

    print('Training Complete - Start Masking Audio')
    trans_complete = complete_scenario.permute(2, 1, 0).cuda()
    trans_clean = clean.permute(2, 1, 0).cuda()

    #Predict Model
    test_out = model(trans_complete, trans_clean).float().cuda()

    test_out = test_out.permute(0, 1, 2).cuda()
    test_out = OA_Reconstruct(torch.squeeze(test_out), 'HAMMING', 16000*0.032, 16000*0.016)

    test_clean = trans_clean.permute(0, 1, 2).cuda()
    test_clean = OA_Reconstruct(torch.squeeze(test_clean), 'HAMMING', 16000*0.032, 16000*0.016)

    #complete_scenario = complete_scenario.permute(0, 1, 2).cuda()
    complete_scenario = OA_Reconstruct(torch.squeeze(trans_complete), 'HAMMING', 16000*0.032, 16000*0.016)

    test_out = test_out.cpu().detach().numpy()
    test_clean = torch.squeeze(test_clean)
    test_clean = test_clean.cpu().detach().numpy()
    complete_scenario = complete_scenario.cpu().detach().numpy()

    sc.write('data/TestSeparation.wav', fs,  test_out)
    sc.write('data/TestSeparation_clean.wav', fs,  test_clean)
    sc.write('data/TestSeparation_CompleteScenario.wav', fs, complete_scenario)

    test = 1

