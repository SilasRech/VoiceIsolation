import numpy as np
import sys
import torch


def OA_Windowing(inSig, window, winlen, step):
    """
    Overlap-add windowing
    """

    siglen = inSig.size(1)
    step = int(step)
    winlen = int(winlen)
    if window.upper() == "HAMMING":
        w = np.hamming(winlen)
    elif window.upper() == "HANN":
        w = np.hanning(winlen)
    elif window.upper() == "COSINE":
        w = np.sqrt(np.hanning(winlen))
    else:
        sys.exit("Window not supported")

    nWins = int(np.floor((siglen - winlen) / step)) + 1

    zpSig = torch.nn.functional.pad(inSig, (0, 0, 0, winlen), mode='constant', value=0)
    #zpSig = torch.from_numpy(np.pad(inSig.numpy(), (0, winlen), 'constant', constant_values=(0, 0)))
    winSig = torch.zeros((winlen, nWins))

    winIdxini = 0
    winIdxend = winlen
    zpSig = torch.reshape(zpSig, (-1, 1))

    for i in range(0, nWins):
        winSig[:, i] = torch.reshape(torch.multiply(torch.reshape(zpSig[winIdxini:winIdxend],(winlen,)), w),(winlen,))
        winIdxini += step
        winIdxend += step

    return winSig


def OA_Reconstruct(in_frames, window, winlen, step):
    """
    Overlap-Add reconstruction of the signal.
    """

    in_frames = torch.squeeze(in_frames)

    step = int(step)
    winlen = int(winlen)

    num_f = in_frames.size(0)

    # Initialise output signal in time domain.

    output_len = int((step * num_f) + winlen)

    out_sig = torch.zeros((output_len,)).cuda()

    # Generate window function

    if window.upper() == "HAMMING":
        w = torch.hamming_window(winlen)
    elif window.upper() == "HANN":
        w = torch.hann_window(winlen)
    elif window.upper() == "COSINE":
        w = torch.sqrt(torch.hann_window(winlen))
    elif window.upper() == "RECT":
        w = torch.ones((winlen,))
    elif window.upper() == "FT":
        w = flat_top_win(winlen, step)
    else:
        sys.exit("Window not supported")

    w = w.cuda()
    ini_frame = 0
    end_frame = winlen

    for i in range(num_f):
        out_sig[ini_frame:end_frame] += in_frames[i, :] * w
        ini_frame += step
        end_frame += step

    return out_sig


def flat_top_win(wlen,wstep):
    """
    Flat-top window function
    """

    flatlen = int(wlen - wstep)

    ramp_win = np.sin(np.pi*np.arange(0.5,wstep/2)/wstep).reshape(np.int(wstep/2),1)
    hwin = np.concatenate((ramp_win, np.ones((flatlen,1)), np.flip(ramp_win)))

    return hwin.reshape((-1,))

def win_spec(inSig, nfft, winlen, step, window):
    """
    Window signal and calculate spectrogram
    """

    win_sig = OA_Windowing(inSig, window, winlen, step)
    dftsig = np.fft.fft(win_sig, n=nfft, axis=0, norm='ortho')
    dftsig = dftsig[:np.int((nfft / 2) + 1), :]
    return dftsig




