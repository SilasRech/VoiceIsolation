import os
import glob
import sys
import random
import torchaudio
import numpy as np
from tqdm import tqdm

import torch
import torchmetrics
from torchsummary import summary

from torch.utils.data import DataLoader
from training import train, model_eval

from metafiles import generate_meta_files

from model import ConvTasNet, ConvTasNet_Original
from dataloader import LibriSpeech
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required Arguments
    parser.add_argument('--datapath', help='root path to Librispeech database', type=str, required=False, default='C:/Users/rechs1/LibriSpeech/dev-clean')
    parser.add_argument('--mode', help='Should the model train or evaluate', type=str, required=False, default='train')
    parser.add_argument('--loss', help='Which loss function to use', type=str, required=False, default='si_snr')

    # Arguments for training
    parser.add_argument('--epochs', help='Should the model train or evaluate', type=int, required=False, default=100)

    args = parser.parse_args()
    #model = ConvTasNet(L=64, N=512, X=8, R=1, B=256, H=512, P=3, norm="cLN", non_linear="relu", causal=True).cuda()
    #summary(model, [(5, 1, 16000), (5, 1, 16000)])
    #out1 = model(in1, in2)

    #print(model)

    root = os.getcwd()
    mode = args.mode

    num_speaker = 4
    generate_meta_files(root, datapath=args.datapath, database_length=20000, overwrite=False, num_speakers=num_speaker)

    torch.backends.cudnn.benchmark = True
    model_name = 'IsoNet.pt'

    BATCH_SIZE = 16

    EPOCHS = 100
    addnoise = False
    fs = 16000

    total_length = 4
    input_length = 16000  # in samples
    win_length = int(fs * total_length)

    with open(os.path.join(root, 'training_list.txt')) as f:
        training_files = f.read().splitlines()

    print(mode)

    eval_dataset = LibriSpeech(training_files[:50], root_dir=root, fs=fs, length=total_length,
                               length_input=input_length, batchsize=1, num_speaker=num_speaker)
    eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=4, prefetch_factor=8, pin_memory=True)

    if mode == 'train':
        print("-----------------Training Mode Started-----------------")
        print("-----------------Building Database-----------------")

        train_dataset = LibriSpeech(training_files, root_dir=root, fs=fs, length=total_length,
                                    length_input=input_length, batchsize=1, num_speaker=num_speaker)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, prefetch_factor=8,
                                  pin_memory=True)

        print("-----------------Loading Model-----------------")

        #model = ConvTasNet_Original()
        model = ConvTasNet()
        model.cuda()
        #summary(model, [(16, 64000), (16, 64000)])
        # criterion = torchmetrics.ScaleInvariantSignalNoiseRatio(full_state_update=False).cuda()

        criterion = torch.nn.MSELoss()

        optim_params = model.parameters()
        optimizer = torch.optim.Adam(optim_params, 0.001)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                               verbose=True, min_lr=0)
        print("-----------------Starting Training------------------")
        for epoch in range(EPOCHS):
            model = train(train_loader, model, criterion, optimizer, epoch, scheduler, loss_mode="mse")

            torch.save(model, model_name)

            print("\n -----------------Starting Evaluation----------------")

            isolated_predictions = model_eval(eval_loader, model, fs, total_length)
    else:
        print("-----------------Evaluation Mode Started-----------------")

        leaked_target, leaked_interferer, target_audio = next(iter(eval_loader))

        target_mix = torch.reshape(torch.squeeze(leaked_target), shape=(1, int(fs * total_length)))
        interfering_mix = torch.reshape(torch.squeeze(leaked_interferer), shape=(1, int(fs * total_length)))
        target = torch.reshape(torch.squeeze(target_audio), shape=(1, int(fs * total_length)))

        for k in range(len(target_mix)):
            torchaudio.save('./evaluation_sample/target_mix{}.wav'.format(k),
                            target_mix, 16000)
            torchaudio.save('./evaluation_sample/target{}.wav'.format(k),
                            target, 16000)
            torchaudio.save('./evaluation_sample/interferer_mix{}.wav'.format(k),
                            interfering_mix, 16000)
        print("-----------------Loading Model---------------------")
        loaded_model = torch.load(model_name).cuda()

        print("-----------------Loading Completed, starting evaluation...-------------")
        isolated_predictions = model_eval(eval_loader, loaded_model, fs, total_length)
        print("Files Successfully Written, Finished")