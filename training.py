import os

import torch
import torchaudio
import time
import torchmetrics

from tools import AverageMeter, ProgressMeter


def train(train_loader, model, criterion, optimizer, epoch, scheduler, loss_mode):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (target_mix, interferer_mix, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = torch.squeeze(target).to('cuda:0', non_blocking=True)
        target_mix = torch.squeeze(target_mix).to('cuda:0', non_blocking=True)
        interferer_mix = torch.squeeze(interferer_mix).to('cuda:0', non_blocking=True)

        # compute output and loss
        isolated = model(target=target_mix, interferer=interferer_mix)
        loss_mode = 'sisnr'
        if loss_mode == 'mse':
            loss = criterion(isolated, target)
        else:
            loss = -torch.mean(torchmetrics.functional.scale_invariant_signal_noise_ratio(isolated, target))
        losses.update(loss.item(), target.size(0))

        # compute gradient and do SGD step
        loss.backward()
        
        if (i+1) % 20 == 0 or (i+1) == len(train_loader):
          # Update weights
          optimizer.step()        # Reset the gradients to None
          optimizer.zero_grad(set_to_none=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

    scheduler.step(metrics=loss)
    return model


def model_eval(eval_loader, model, fs, length):

    # switch to eval mode
    model.eval()

    for i, (target_mix, interferer_mix, target) in enumerate(eval_loader):

        target = target.to('cuda:0', non_blocking=True)
        target_mix = target_mix.to('cuda:0', non_blocking=True)
        interferer_mix = interferer_mix.to('cuda:0', non_blocking=True)

        # compute output and loss
        isolated = model(target=target_mix, interferer=interferer_mix)
        isolated = torch.unsqueeze(isolated, dim=1)

        target_mix = torch.reshape(torch.squeeze(target_mix), shape=(1, int(fs * length))).detach().cpu()
        interfering_mix = torch.reshape(torch.squeeze(interferer_mix), shape=(1, int(fs * length))).detach().cpu()
        isolated = torch.reshape(torch.squeeze(isolated), shape=(1, int(fs * length))).detach().cpu()
        target = torch.reshape(torch.squeeze(target), shape=(1, int(fs * length))).detach().cpu()

        torchaudio.save(os.getcwd()+'/evaluation_sample/Target_mix{}.wav'.format(i), target_mix, 16000)
        torchaudio.save(os.getcwd()+'/evaluation_sample/Interfering_mix{}.wav'.format(i), interfering_mix, 16000)
        torchaudio.save(os.getcwd()+'/evaluation_sample/Isolated{}.wav'.format(i), isolated, 16000)
        torchaudio.save(os.getcwd()+'/evaluation_sample/TargetReference{}.wav'.format(i), target, 16000)
    return isolated


def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = torch.sum(s1 * s2, -1, keepdim=True)
    s2_s2_norm = torch.sum(s2* s2, -1, keepdim=True)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = torch.sum(s_target * s_target, -1, keepdim=True)
    noise_norm = torch.sum(e_nosie * e_nosie, -1, keepdim=True)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return -torch.mean(snr)
