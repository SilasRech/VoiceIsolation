import numpy as np
import torch
import HelperFunctions as hf
import matplotlib.pyplot as plt


def echo_can(signal_nearend, signal_farend, winlen, winstep, nfft, window, n_delay_blocks=8):

    # Signals windowing

    nfft = int(np.power(2, np.ceil(np.log2(winlen))))

    near_windowed = hf.OA_Windowing(signal_nearend, window=window, winlen=winlen, step=winstep)
    far_windowed = hf.OA_Windowing(signal_farend, window=window, winlen=winlen, step=winstep)

    nlms_bg = hf.NLMS(n_bands=int(nfft / 2) + 1, p=n_delay_blocks, mu_max=0.95)
    nlms_fg = hf.NLMS(n_bands=int(nfft / 2) + 1, p=n_delay_blocks, mu_max=0)
    out_nlms = hf.NLMS(n_bands=int(nfft / 2) + 1, p=n_delay_blocks, mu_max=0.95)

    n_frames = near_windowed.size(1)

    out_frames = torch.zeros(near_windowed.size())
    far_freq_mem = torch.zeros((n_delay_blocks, int(nfft / 2) + 1))

    ctd_out = torch.zeros((n_frames, 1))

    ctd_before_update = ncc_CT_detect_time(winlen=winlen, nfft=nfft, forg_factor=0.8)
    ctd_after_update = ncc_CT_detect_time(winlen=winlen, nfft=nfft, forg_factor=0.8)

    ctd_result_before_update = torch.zeros((n_frames, 1))
    ctd_result_after_update = torch.zeros((n_frames, 1))

    abs_err = torch.zeros((n_frames, 1))

    Adapt_flag = True

    Syy = torch.zeros((n_frames, 1))

    silences = torch.ones((n_frames, 1))

    Filter_learning_rate = np.zeros((n_frames, 1))

    count = 0
    other_count = 0

    for i in range(n_frames):

        new_near_frame = near_windowed[:, i]
        new_far_frame = far_windowed[:, i]
        # new_ref_frame = ref_windowed[:,i]

        new_near_spec = torch.fft.rfft(new_near_frame, n=nfft)
        new_far_spec = torch.fft.rfft(new_far_frame, n=nfft)
        # new_ref_spec = torch.fft.rfft(new_ref_frame,n = nfft)

        Syy[i] = torch.abs(torch.matmul(torch.transpose(new_far_spec.reshape((-1, 1)), 0, 1),
                                        new_far_spec.reshape((-1, 1)).conj()))

        # Sref[i] = torch.abs(torch.matmul(torch.transpose(new_ref_spec.reshape((-1,1)),0,1), new_ref_spec.reshape((-1,1)).conj())) > 0.05

        # Detect silence
        if Syy[i] < 0.005:
            count += 1
            silences[i] = 0
            out_spec = new_near_spec
            if count > n_delay_blocks:
                nlms_fg.mu = 0
                nlms_bg.mu = 0
                out_nlms.mu = 0
                nlms_bg.reset()
                nlms_fg.reset()
                ctd_before_update.reset()
                ctd_after_update.reset()
        else:
            count = 0
            nlms_fg.mu = nlms_fg.mu_max
            nlms_bg.mu = nlms_bg.mu_max
            out_nlms.mu = out_nlms.mu_max

        # Filter memory
        far_freq_mem = torch.cat((torch.reshape(new_far_spec, (1, -1)), far_freq_mem[0:-1, :]), dim=0)

        # Update memory and filter with the foreground filter (slow adapting)
        nlms_fg.update_mem(new_far_spec.numpy())

        y_spec = torch.from_numpy(np.diag(np.dot(nlms_fg.W.conj().T, far_freq_mem.numpy())))
        out_spec_before = new_near_spec - y_spec

        # Update memory, adapt and filter with the background filter (fast adapting)
        _, ctd_result_before_update[i] = ctd_before_update.detect(new_near_spec.reshape((-1, 1)), out_spec_before)

        nlms_bg.update(new_far_spec.numpy(), new_near_spec.numpy())

        y_spec = torch.from_numpy(np.diag(np.dot(nlms_bg.W.conj().T, far_freq_mem.numpy())))
        out_spec_after_update = new_near_spec - y_spec

        decision, ctd_result_after_update[i] = ctd_after_update.detect(new_near_spec.reshape((-1, 1)),
                                                                       out_spec_after_update)

        # Make Decision if Double Talk is used
        if (ctd_result_after_update[i] < 0.4 and ctd_result_before_update[i] < 0.2):
            detected = True
        elif (ctd_result_after_update[i] > 0.5 and ctd_result_before_update[i] > 0.4):
            detected = False

        # Apply actions depending on double talk detection. Let the filter adapt at the beginning of the recording for 2*filter_length regardless of double talk
        if detected and (i >= 2 * n_delay_blocks):
            other_count = 0
            out_nlms.mu = 0.1
        else:
            other_count += 1
            if other_count >= 2 or (i < 2 * n_delay_blocks):
                other_count = 1
                out_nlms.mu = out_nlms.mu_max
                Filter_learning_rate[i] = 1

        if i < 2 * n_delay_blocks:
            out_nlms.mu = out_nlms.mu_max
            Filter_learning_rate[i] = 1

        out_nlms.update(new_far_spec.numpy(), new_near_spec.numpy())
        y_spec = torch.from_numpy(np.diag(np.dot(out_nlms.W.conj().T, far_freq_mem.numpy())))
        out_spec = new_near_spec - y_spec

        # Copy weights from output filter to foreground
        nlms_fg.W = out_nlms.W

        out_frame = torch.reshape(torch.fft.irfft(out_spec, n=nfft), (-1,))

        abs_err[i] = torch.max(torch.abs(out_spec))

        ctd_out[i] = Adapt_flag

        out_frames[:, i] = out_frame[:winlen]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(signal_nearend)
    plt.subplot(3, 1, 2)
    plt.plot(ctd_result_after_update)
    plt.plot(ctd_result_before_update, 'g')
    plt.plot(ctd_result_before_update / ctd_result_after_update, 'r')
    plt.ylim([0, 1])
    plt.plot(Filter_learning_rate, 'y')
    plt.subplot(3, 1, 3)
    plt.plot(signal_farend)
    plt.show(block=False)

    out_sig = hf.OA_Reconstruct(torch.transpose(out_frames, 0, 1), window='cosine', winlen=winlen, step=winstep)
    aux_y = None
    return out_sig

class check_silence:
    def __init__(self, mem_length, ratio):
        self.mem = torch.zeros((mem_length, 1))
        self.ratio = ratio

    def check(self, Syy, Sdd):
        self.mem = torch.cat((Syy.reshape((-1, 1)), self.mem[0:-1, :]))
        max_Syy = torch.max(self.mem)
        if Sdd > self.ratio * max_Syy:
            return True

        return False

class ncc_CT_detect_freq:
    def __init__(self, forg_factor=0.9):
        self.forg_factor = forg_factor

        self.decision = True

        self.red = 0
        self.sigma_d = 1
        self.sigma_e = 1
        self.decision_ratio = 1

        self.count = 0

    def detect(self, near_frame, error_frame):
        newred = np.matmul(np.abs(error_frame.T.numpy()), np.abs(near_frame.numpy()))
        newsigmad = np.abs(np.matmul(near_frame.T.numpy().conj(), near_frame.numpy()))
        newsigmae = np.abs(np.matmul(error_frame.T.numpy().conj(), error_frame.numpy()))
        self.red = self.forg_factor * self.red + (1 - self.forg_factor) * newred
        self.sigma_d = self.forg_factor * self.sigma_d + (1 - self.forg_factor) * newsigmad
        self.sigma_e = self.forg_factor * self.sigma_e + (1 - self.forg_factor) * newsigmae

        aux_ratio = (self.red / (np.sqrt(self.sigma_d * self.sigma_e) + 1e-6))
        self.decision_ratio = 1 - aux_ratio

        if self.decision_ratio.item() < 0.5:
            self.decision = False
        else:
            self.decision = True

        return self.decision, self.decision_ratio.item()

    def reset(self):
        self.red = 1
        self.sigma_d = 1
        self.sigma_e = 1
        self.decision_ratio = 0
        self.count = 0
        self.decision = True

class ncc_CT_detect_time:
    def __init__(self, winlen, nfft, forg_factor=0.9):
        self.forg_factor = forg_factor

        self.winlen = winlen
        self.nfft = nfft
        self.decision = True

        self.red = 1
        self.sigma_d = 1
        self.sigma_e = 1
        self.decision_ratio = 1

        self.count = 0

    def detect(self, near_spec, error_spec):

        near_frame = torch.fft.irfft(near_spec.reshape((-1,)), n=self.nfft)[:self.winlen]
        error_frame = torch.fft.irfft(error_spec, n=self.nfft)[:self.winlen]

        newred = np.sum(error_frame.numpy() * near_frame.numpy())
        newsigmad = np.sum(near_frame.numpy() * near_frame.numpy())
        newsigmae = np.sum(error_frame.numpy() * error_frame.numpy())
        self.red = self.forg_factor * self.red + (1 - self.forg_factor) * newred
        self.sigma_d = self.forg_factor * self.sigma_d + (1 - self.forg_factor) * newsigmad
        self.sigma_e = self.forg_factor * self.sigma_e + (1 - self.forg_factor) * newsigmae

        aux_ratio = (self.red / (np.sqrt(self.sigma_d * self.sigma_e) + 1e-6))
        self.decision_ratio = 1 - aux_ratio

        if self.decision_ratio.item() < 0.5:
            self.decision = False
        else:
            self.decision = True

        return self.decision, self.decision_ratio.item()

    def reset(self):
        self.red = 1
        self.sigma_d = 1
        self.sigma_e = 1
        self.decision_ratio = 1
        self.count = 0
        self.decision = True
