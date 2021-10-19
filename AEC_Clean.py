import os, sys
import numpy as np
from scipy.io.wavfile import write
import HelperFunctions as hf
import matplotlib.pyplot as plt
import torch
import torchaudio

#Define audio backend to enable use of torchaudio
if os.name == 'nt':
    torchaudio.set_audio_backend('soundfile')
elif os.name == 'posix':
    torchaudio.set_audio_backend('sox_io')
else:
    print('OS not supported by torchaudio')
    exit()


def main(signal_nearend, signal_farend, winlen, winstep, nfft, window, n_delay_blocks = 4):

    # Signals windowing
    nfft = int(np.power(2,np.ceil(np.log2(winlen))))

    near_windowed = hf.OA_Windowing(signal_nearend, window = window, winlen = winlen, step = winstep)
    far_windowed = hf.OA_Windowing(signal_farend, window = window, winlen = winlen, step = winstep)

    nlms = hf.NLMS(n_bands = int(nfft/2) + 1, p = n_delay_blocks, mu_max = 1.1)

    n_frames = near_windowed.size(1)

    #Initialize tools and debug variables
    out_sig = torch.zeros((int(nfft/2 * n_frames),1))

    aux_y = torch.zeros((int(nfft/2 * n_frames),1))

    out_spec = torch.zeros((int(nfft/2) + 1, 1))

    out_frame = torch.zeros((near_windowed.size(0),1))
    out_frames = torch.zeros(near_windowed.size())
    far_freq_mem = torch.zeros((n_delay_blocks,int(nfft/2) + 1))

    ctd_out = torch.zeros((n_frames,1))

    ctd_aux = ncc_CT_detect_freq(forg_factor = 0.8)

    ctd_result = torch.zeros((n_frames,1))
    abs_err = torch.zeros((n_frames,1))

    Adapt_flag = True

    Sdd = torch.zeros((n_frames,1))
    Syy = torch.zeros((n_frames,1))
    Sdy = torch.zeros((n_frames,1))

    silences = torch.ones((n_frames,1))

    Filter_learning_rate = np.zeros((n_frames,1))

    count = 0

    for i in range(n_frames):

        new_near_frame = near_windowed[:,i]
        new_far_frame = far_windowed[:,i]

        new_near_spec = torch.fft.rfft(new_near_frame,n = nfft)
        new_far_spec = torch.fft.rfft(new_far_frame,n = nfft)

        # Check silence in echo signal and reset adaptation accordingly
        Syy[i] = torch.abs(torch.matmul(torch.transpose(new_far_frame.reshape((-1,1)),0,1), new_far_frame.reshape((-1,1)).conj()))

        if Syy[i] < 0.005:
            count += 1
            silences[i] = 0
            out_spec = new_near_spec
            if count > n_delay_blocks:
                nlms.reset()
                ctd_aux.reset()
        else:
            count = 0

        #Double talk detection. Using previous error frame in this case (Probably re-checking after filter update works better)
        Adapt_flag,ctd_result[i] = ctd_aux.detect(new_near_spec.reshape((-1,1)), out_spec)
        #Update memory for signal filtering (This is done inside the filter update too)
        far_freq_mem = torch.cat((torch.reshape(new_far_spec,(1,-1)),far_freq_mem[0:-1, :]),dim = 0)

        #Scale learning rate based on double-talk detector result
        scale_factor = ctd_result[i].item()

        nlms.mu = min(nlms.mu_max,nlms.mu_max * scale_factor)

        #Update filter
        Filter_learning_rate[i] = nlms.mu
        nlms.update(new_far_spec.numpy(), new_near_spec.numpy())

        #Filter signal

        y_spec = torch.from_numpy(np.diag(np.dot(nlms.W.conj().T, far_freq_mem.numpy())))
        out_spec = new_near_spec - y_spec
        out_frame = torch.reshape(torch.fft.irfft(out_spec,n = nfft),(-1,))

        abs_err[i] = torch.max(torch.abs(out_spec))

        ctd_out[i] = Adapt_flag

        out_frames[:,i] = out_frame[:winlen]

    #Reconstruct output signal
    out_sig = hf.OA_Reconstruct(torch.transpose(out_frames,0,1), window='cosine', winlen = winlen, step = winstep)
    aux_y = None
    return out_sig,aux_y

class check_silence:
    def __init__(self, mem_length, ratio):
        self.mem = torch.zeros((mem_length,1))
        self.ratio = ratio

    def check(self, Syy, Sdd):
        self.mem = torch.cat((Syy.reshape((-1,1)), self.mem[0:-1,:]))
        max_Syy = torch.max(self.mem)
        if Sdd > self.ratio * max_Syy:
            return True

        return False

class ncc_CT_detect_freq:
    def __init__(self, forg_factor = 0.9):
        self.forg_factor = forg_factor

        self.decision = True

        self.red = 0
        self.sigma_d = 0
        self.sigma_e = 0
        self.decision_ratio = 1

        self.count = 0

    def detect(self, near_frame, error_frame):
        newred = np.abs(np.matmul(error_frame.T.numpy(), near_frame.numpy().conj()))
        newsigmad = np.abs(np.matmul(near_frame.T.numpy(), near_frame.numpy().conj()))
        newsigmae = np.abs(np.matmul(error_frame.T.numpy(), error_frame.numpy().conj()))
        self.red = self.forg_factor * self.red + (1 - self.forg_factor) * newred
        self.sigma_d = self.forg_factor * self.sigma_d + (1 - self.forg_factor) * newsigmad
        self.sigma_e = self.forg_factor * self.sigma_e + (1 - self.forg_factor) * newsigmae

        aux_ratio = (self.red / (np.sqrt(self.sigma_d*self.sigma_e) + 1e-6))
        self.decision_ratio = 1 - aux_ratio


        return self.decision, self.decision_ratio.item()

    def reset(self):
        self.red = 0
        self.sigma_d = 0
        self.sigma_e = 0
        self.decision_ratio = 1
        self.count = 0
        self.decision = True

if __name__ == '__main__':

    fs_target = 16000
    winlen = int(0.09 * fs_target)
    winstep = int(0.06 * fs_target)
    nfft = winlen

    n_delay_blocks = 16

    near_path = os.path.normpath('F:/Mac_input_mic_ref.wav')
    far_path = os.path.normpath('F:/Mac_input_echo_ref.wav')
    near_clean_path = os.path.normpath('./Samples/Nearend/nearend_speech_fileid_24.wav')

    near_sig,fs = torchaudio.load(near_path)
    far_sig,fs_far = torchaudio.load(far_path)
    clean_sig, fs_clean = torchaudio.load(near_clean_path)

    assert(fs == fs_far)

    resampler = torchaudio.transforms.Resample(orig_freq = fs, new_freq = fs_target, resampling_method = 'sinc_interpolation')

    if fs_target != fs:
        near_sig = resampler(near_sig)
        far_sig = resampler(far_sig)

    out_sig,filt_sig = main(torch.reshape(near_sig,(-1,1)), torch.reshape(far_sig,(-1,1)), winlen, winstep, nfft, 'FT', n_delay_blocks = n_delay_blocks)

    #torchaudio.save(filepath = os.path.normpath('./Samples/EchoSample.wav'),src = out_sig, sample_rate = fs_target, format = "wav")
    write(os.path.normpath('./Samples/EchoSample2.wav'), fs_target, out_sig.numpy())
