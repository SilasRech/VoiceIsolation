import numpy as np
from scipy.signal import lfilter
import sys
import torch
import matplotlib.pyplot as plt

def OA_Windowing(inSig, window, winlen, step):
    """
    Overlap-add windowing
    """

    siglen = inSig.size(0)
    step = int(step)
    winlen = int(winlen)
    if window.upper() == "HAMMING":
        #w = np.hamming(winlen)
        w = torch.hamming_window(winlen)
    elif window.upper() == "HANN":
        #w = np.hanning(winlen)
        w = torch.hann_window(winlen)
    elif window.upper() == "COSINE":
        #w = np.sqrt(np.hanning(winlen))
        w = torch.sqrt(torch.hann_window(winlen))
    elif window.upper() == "RECT":
        w = torch.ones((winlen,))
    elif window.upper() == "FT":
        w = torch.from_numpy(flat_top_win(winlen, step))
    else:
        sys.exit("Window not supported")

    #nWins = int(np.floor((siglen - winlen) / step) + 1)
    nWins = int(np.floor((siglen - winlen) / step) + 1)

    zpSig = torch.nn.functional.pad(inSig, (0, 0, 0, winlen), mode='constant', value=0)
    #zpSig = torch.from_numpy(np.pad(inSig.numpy(), (0, winlen), 'constant', constant_values=(0, 0)))
    winSig = torch.zeros((winlen, nWins))

    winIdxini = 0
    winIdxend = winlen
    zpSig = torch.reshape(zpSig,(-1, 1))

    for i in range(0, nWins):
        winSig[:, i] = torch.reshape(torch.multiply(torch.reshape(zpSig[winIdxini:winIdxend],(winlen,)), w),(winlen,))
        winIdxini += step
        winIdxend += step

    return winSig


def OA_Reconstruct(in_frames, window, winlen, step):
    """
    Overlap-Add reconstruction of the signal.
    """

    step = int(step)
    winlen = int(winlen)

    num_f = in_frames.size(0)

    # Initialise output signal in time domain.

    output_len = int((step * num_f) + winlen)

    out_sig = torch.zeros((output_len,))

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


def ener_bands(spec_in, nbands):
    """
    Group frequency samples into energy bands
    """

    nfft = len(spec_in)
    nframes = len(spec_in[0, :])
    bands_per_ener = np.int(np.floor(nfft / nbands))
    ener_out = np.zeros((nbands, nframes))
    band_start = 0
    for i in range(nbands):
        band_stop = band_start + bands_per_ener
        ener_out[i, :] = np.sum(np.power(np.absolute(spec_in[band_start:band_stop, :]), 2), axis=0)
        # ener_out[i, :] = np.mean(np.absolute(spec_in[band_start:band_stop, :]), axis=0)
        # ener_out[i, :] = np.sum(spec_in[band_start:band_stop, :], axis=0)
        band_start = band_stop
    return ener_out


def spectral_subtraction(x, interference):
    """
    Return spectral subtraction coefficients
    """
    out = np.sqrt(np.divide(np.maximum(np.power(x, 2) - np.power(interference, 2), np.zeros(x.shape)), np.power(x, 2)))

    return out


def spectral_subtraction_ener(x, interference):
    """
    Return spectral subtraction coefficients (Trying adaptive scaling)
    """
    """
    scale = np.amax(np.maximum(interference / x, np.ones(x.shape,dtype=np.float64)),axis=0)
    print(scale.shape)

    scale = mtl.repmat(scale,x.shape[0],1)
    plt.figure()
    plt.imshow(scale, aspect = 'auto')
    print(scale.shape)

    interference /= scale
    """
    out = np.sqrt(np.divide(np.maximum(x - interference, np.zeros(x.shape)), x))

    return out

def linear_sub(x, interference):
    """
    Return linear subtraction coefficients
    """
    out = np.divide(np.maximum(x - interference, np.zeros(x.shape)), x)

    return out


def wiener_ener(x, interference):
    """
    Return Wiener Filter coefficients (Tryint adaptive scaling)
    """
    """
    scale = np.amax(np.maximum(interference / x, np.ones(x.shape,dtype=np.float64)),axis=0)
    print(scale.shape)

    scale = mtl.repmat(scale,x.shape[0],1)
    plt.figure()
    plt.imshow(scale, aspect = 'auto')
    print(scale.shape)

    interference /= scale
    """
    out = np.divide(np.maximum(x - interference, np.zeros(x.shape)), x)

    return out

def wiener(x, interference):
    """
    Return Wiener Filter coefficients (Tryint adaptive scaling)
    """
    """
    scale = np.amax(np.maximum(interference / x, np.ones(x.shape,dtype=np.float64)),axis=0)
    print(scale.shape)

    scale = mtl.repmat(scale,x.shape[0],1)
    plt.figure()
    plt.imshow(scale, aspect = 'auto')
    print(scale.shape)

    interference /= scale
    """
    out = np.divide(np.maximum(np.power(np.absolute(x),2) - np.power(np.absolute(interference),2), np.zeros(x.shape)), np.power(np.absolute(x),2))

    return out


def expand_spectrum(in_ener, nfft):
    out_spec = np.zeros((nfft, in_ener.shape[1]))
    n_ener_bands = in_ener.shape[0]
    bands_per_ener = np.int(np.floor(nfft / n_ener_bands))
    freq_band = 0
    for b in range(n_ener_bands):
        for i in range(bands_per_ener):
            out_spec[freq_band, :] = in_ener[b, :]
            freq_band += 1

    if freq_band < nfft:
        for f in range(freq_band, nfft):
            out_spec[freq_band, :] = in_ener[n_ener_bands - 1, :]

    return out_spec

class Noise_Gate:
    #Noise Gate with hystheresys
    #Params:
    #   holdtime    - Number of frames the sound level has to be below the threshold value before the gate is activated
    #   ltrhold     - Threshold value for activating the gate.
    #   utrhold     - Threshold value for deactivating the gate.
    #   release     - Number of frames before the sound level reaches zero
    #   attack      - Number of frames before the output sound level is the same as the input level after deactivating the gate
    #   a           - Pole placement of the envelope detecting filter < 1
    #   Fs          - Sampling frequency
    def __init__(self, holdtime, ltrhold, utrhold, release, attack, a = None):
        self.holdtime = holdtime
        self.ltrhold = ltrhold
        self.utrhold = utrhold
        self.lthcnt = 0
        self.uthcnt = 0
        self.release = release
        self.attack = attack
        self.a = a
        self.g = 1
    
    def filter(self, x):
        xrms = RMS_dB(x)
        g = self.g
        if (xrms <= self.ltrhold) or ((xrms < self.utrhold) and (self.lthcnt > 0)):
            #value below the lower threshold?
            self.lthcnt = self.lthcnt + 1
            self.uthcnt = 0
            if self.lthcnt > self.holdtime:
                #Time below the lower threshold longer than the hold time?
                if self.lthcnt > (self.release + self.holdtime):
                    g = 0
                else:
                    g = 1
        elif (xrms >= self.utrhold) or ((xrms > self.ltrhold) and (self.uthcnt > 0)):
            #Value above the upper threshold or is the signal being faded in?
            self.uthcnt = self.uthcnt + 1
            if (g<1):
                #Has the gate been activated or isn't the signal faded in yet?
                g = max(self.uthcnt/self.attack,g)
            else:
                g = 1
            
            self.lthcnt = 0
        else:
            g = self.g
            self.lthcnt = 0
            self.uthcnt = 0
        y = x * g
        self.g = g
        return y

class NG_freq:
    """
    Noise gate class applied bandwise on frequency domain for higher quality.
    """
    def __init__(self, holdtime, ltrhold, utrhold, release, attack, nfft, memory_len):
        self.holdtime = holdtime
        self.ltrhold = ltrhold
        self.utrhold = utrhold
        self.lthcnt = np.zeros((nfft,))
        self.uthcnt = np.zeros((nfft,))
        self.release = release
        self.attack = attack
        self.nfft = nfft
        self.g = np.zeros((nfft,))
        self.mem_dB = np.zeros((nfft,memory_len))
        self.mov_avg = np.zeros((nfft,))
        self.avg_rms = 0
        self.is_speech = False

    
    def filter(self,x):

        x_freq = np.fft.fft(x.numpy(),n=self.nfft)

        x_freq_db = 20 * np.log10(np.abs(x_freq))

        
        #rms_th_up = self.avg_rms + 3
        #rms_th_dn = self.avg_rms - 15
        #if self.is_speech:
        #    self.is_speech = np.mean(x_freq_db) > rms_th_dn
        #else:
        #    self.is_speech = np.mean(x_freq_db) > rms_th_up
        #self.avg_rms = 0.99*self.avg_rms + 0.01*np.mean(x_freq_db)
        #mean_energy_dB = np.mean(x_freq_db)
        #threshold_dB = mean_energy_dB + 12.
        #speech_active = bin_energy_dB > threshold_dB

        #self.mem_dB = np.concatenate((self.mem_dB[:,:-1],x_freq_db))

        #for window_ix in range(window_count):
        #    range_start = max(0,window_ix-hysteresis_time)
        #speech_active_hysteresis = np.max(int(speech_active))#[range(range_start,window_ix+1)])
        
        #speech_active_sloped = np.zeros([memory_length])
        #for frame_ix in range(window_count):
        #if speech_active_hysteresis:
        #    speech_active_sloped = np.mean(speech_active_hysteresis[:,-self.fade_in_time:])#[range(range_start,frame_ix+1)])
        #else:
        #    speech_active_sloped = np.mean(speech_active_hysteresis[:,-self.fade_out_time:])#[range(range_start,frame_ix+1)])

        self.update_gate(x_freq_db)

        spectrogram_binwise = x_freq*self.g

        #if not self.is_speech:
        #    spectrogram_binwise *= 0

        out_frame = torch.from_numpy(np.real(np.fft.ifft(spectrogram_binwise,n=self.nfft)))
        return out_frame
    
    def update_gate(self, xrms):

        mem_avg = 0.9*self.mov_avg + 0.1*xrms#np.mean(self.mem_dB,axis=1)
        self.mov_avg = mem_avg

        for f_band in range(self.nfft):

            g = self.g[f_band]

            th_up = mem_avg[f_band] + self.utrhold
            th_dn = mem_avg[f_band] - self.ltrhold
            if (xrms[f_band] <= th_dn) or ((xrms[f_band] < th_up) and (self.lthcnt[f_band] > 0)):
                #value below the lower threshold?
                self.lthcnt[f_band] = self.lthcnt[f_band] + 1
                self.uthcnt[f_band] = 0
                if self.lthcnt[f_band] > self.holdtime:
                    #Time below the lower threshold longer than the hold time?
                    if self.lthcnt[f_band] > (self.release + self.holdtime):
                        g = 0
                    else:
                        g = 1
            elif (xrms[f_band] >= th_up) or ((xrms[f_band] > th_dn) and (self.uthcnt[f_band] > 0)):
                #Value above the upper threshold or is the signal being faded in?
                self.uthcnt[f_band] = self.uthcnt[f_band] + 1
                if (g<1):
                    #Has the gate been activated or isn't the signal faded in yet?
                    g = max(self.uthcnt[f_band]/self.attack,g)
                else:
                    g = 1
                
                self.lthcnt[f_band] = 0
            else:
                g = self.g[f_band]
                self.lthcnt[f_band] = 0
                self.uthcnt[f_band] = 0
            
            self.g[f_band] = g
            self.mem_dB = np.concatenate((self.mem_dB[:,1:],xrms.reshape((-1,1))),axis=1) 


        #if not self.speech_active:
        #    ths =  + self.th_up
        #    if xrms > ths:
        #        self.speech_active = True
        #else:
        #    ths = np.mean(self.mem_dB,axis=1) - self.th_dn
        #    if xrms < ths:
        #        self.speech_active = False



class compexp:
    #COMPEXP Compressor and expander function
    #Params:
    #   CT: Compressor Threshold
    #   CS: Compressor Slope 0 < CS < 1
    #   ET: Expander Threshold
    #   ES: Expander Slope ES < 0
    def __init__(self, CT, CS, ET, ES, memory, tav = 0.01, at = 0.03, rt = 0.003):
        self.CT = CT
        self.CS = CS
        self.ET = ET
        self.ES = ES
        self.buffer = torch.zeros((memory,1)) # np.zeros((memory,1))
        self.buffer_idx = memory
        self.tav = tav
        self.at = at
        self.rt = rt
        self.g = 1

    def filter(self, x):

        x = torch.reshape(x,(-1,1))
        self.buffer = torch.cat((x,self.buffer[0:-1]),dim=0) # np.concatenate(x,self.buffer[0:-1])
        xrms = torch.sum(torch.pow(self.buffer,2))/self.buffer.size(0) # np.sum(np.power(self.buffer,2)) / self.buffer.shape[0]
        X = 10*torch.log10(xrms) # 10*np.log10(xrms)
        G = torch.min(torch.tensor((0,self.CS*(self.CT - X), self.ES*(self.ET - X)))) # np.amin(np.array((0,self.CS*(self.CT - X), self.ES*(self.ET - X)))) ?????
        f = 10**(G/20)
        if f < self.g:
            coeff = self.at
        else:
            coeff = self.rt
        
        self.g = (1-coeff) * self.g + coeff * f
        y = self.g * self.buffer[0]
        
        return y
    
def RMS_dB(in_sig):
    RMS = 10*torch.log10(torch.mean(torch.square(torch.abs(in_sig))))
    return RMS


class MDF_filter():
    """
    Multi-delay adaptive filter class
    """
    def __init__(self, nfft, num_blocks, block_size, learning_step = 0.25):
        self.nfft = nfft
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.e_freq = torch.zeros((1,self.block_size), dtype = torch.cfloat)

        self.mu = learning_step # p in c code

        self.memory_blocks = torch.zeros((self.num_blocks,self.block_size), dtype = torch.cfloat)
        self.filters = torch.zeros((self.num_blocks,self.block_size),dtype = torch.cfloat)

        self.Zk = torch.zeros((1,self.block_size))
        self.Pk = torch.zeros((self.num_blocks,self.block_size))

        self.beta = 0.8

    def process_frame_update_weights(self, new_frame, d_frame):

        y_freq = torch.zeros((1,self.block_size),dtype = torch.cfloat)

        new_frame_freq = torch.reshape(torch.fft.rfft(new_frame,n=self.nfft, norm = 'ortho'),(1,-1))

        self.memory_blocks = torch.cat((new_frame_freq,self.memory_blocks[:-1,:]))

        for k in range(self.num_blocks):
            y_freq += self.memory_blocks[k,:] * self.filters[k,:]
        
        y = torch.fft.irfft(y_freq,n=self.nfft, norm='ortho')[:,int(self.nfft / 2):]
        e = (d_frame - y)
        e_full = torch.cat((torch.zeros((1,int(self.nfft/2))),e),dim=1)

        self.e_freq = torch.fft.rfft(e_full, n=self.nfft, norm = 'ortho')

        self._update_weights()

        return e,y
    
    def _update_weights(self):

        M = self.num_blocks # Define gain factor in filter memory

        for k in range(1,self.num_blocks):
            self.Pk[k,:] = self.Pk[k-1,:]
        self.Pk[0,:] = torch.conj(self.memory_blocks[0,:])*self.memory_blocks[0,:]

        self.Zk = self.beta * self.Zk + (1 - self.beta) * torch.sum(self.Pk,dim=0)

        muk = (M*self.mu)/self.Zk

        for k in range(self.num_blocks):
            phi_aux = torch.conj(self.memory_blocks[k,:]) * self.e_freq
            phi = torch.fft.irfft(phi_aux,n=self.nfft, norm='ortho')[:,:int(self.nfft / 2)]
            phi_zp = torch.cat((phi,torch.zeros((1,int(self.nfft/2)))),dim=1)
            phi_freq = torch.fft.rfft(phi_zp, n=self.nfft, norm='ortho')
            self.filters[k,:] = self.filters[k,:] + muk*phi_freq

class NG_band:
    """
    Noise gate class applied bandwise on frequency domain for higher quality.
    """
    def __init__(self, holdtime, ltrhold, utrhold, release, attack, nfft, ener_bands, ener_type, fs, memory_len):
        self.holdtime = holdtime
        self.ltrhold = ltrhold
        self.utrhold = utrhold
        self.release = release
        self.attack = attack
        self.nfft = nfft
        self.fs = fs
        self.n_freq = int(nfft/2 + 1)
        self.ener_bands = ener_bands
        self.ener_type = ener_type.upper()

        self.lthcnt = np.zeros((self.ener_bands,))
        self.uthcnt = np.zeros((self.ener_bands,))
        
        self.g = np.zeros((self.ener_bands,))
        self.mem_dB = np.zeros((self.ener_bands,memory_len))
        self.mov_avg = np.zeros((self.ener_bands,1))
        self.linear_avg = np.zeros((self.ener_bands,1))
        self.avg_rms = 0
        self.is_speech = False
        self.normalizer = Normalizer(holdtime, ltrhold, utrhold, release, attack)
        self.compressor = Compressor(delta_th = 6)
        self.compressor = LinearCompressor(-23, 3, 20)

    
    def filter(self,x):

        x_freq = np.fft.fft(x.numpy(),n=self.nfft,norm='ortho')

        x_freq_cut = x_freq[0:self.n_freq]
        #Calculate energy bands
        x_ener = self.calc_energy_bands(x_freq_cut)

        x_freq_db = 10 * np.log10(np.abs(x_ener))

        self.update_gate(x_freq_db)
        
        #self.normalizer.update(np.abs(x_ener))

        #g = self.g * self.normalizer.g
        comp_g = self.compressor.filter(x_freq_cut)
        g = self.g * comp_g

        #g = self.g
        #self.n_active_bands = 10
        #if np.sum(self.g) > self.n_active_bands:
        #    g_aux = - 10 - 10*np.log10(np.mean(self.linear_avg*self.g))
        #    g = self.g * np.power(10,g_aux/20)
        #else:
        #    g = self.g * 0


        g = self.unfold_gain_bands(g)

        spectrogram_binwise = x_freq*g

        if np.sum(x_ener) > np.sum(self.linear_avg):
            self.linear_avg = 0.9 * self.linear_avg + 0.1 * x_ener
        else:
            self.linear_avg = 0.95 * self.linear_avg + 0.05 * x_ener

        #x_ener_norm = 10 * np.log10(np.mean(self.calc_energy_bands(spectrogram_binwise)))

        #self.check_speech(x_ener_norm)
        #if self.is_speech == False:
        #    spectrogram_binwise *= 0

        out_frame = torch.from_numpy(np.real(np.fft.ifft(spectrogram_binwise,n=self.nfft,norm='ortho')))
        return out_frame, x_freq_db.reshape((-1,))
    
    def update_gate(self, xrms):
        
        mem_avg = self.mov_avg

        for f_band in range(self.ener_bands):

            g = self.g[f_band]

            th_up = mem_avg[f_band] + self.utrhold
            th_dn = mem_avg[f_band] - self.ltrhold
            if (xrms[f_band] <= th_dn) or ((xrms[f_band] < th_up) and (self.lthcnt[f_band] > 0)):
                #value below the lower threshold?
                self.lthcnt[f_band] = self.lthcnt[f_band] + 1
                self.uthcnt[f_band] = 0
                if self.lthcnt[f_band] > self.holdtime:
                    #Time below the lower threshold longer than the hold time?
                    if self.lthcnt[f_band] > (self.release + self.holdtime):
                        g = 0
                    else:
                        g = 1
            elif (xrms[f_band] >= th_up) or ((xrms[f_band] > th_dn) and (self.uthcnt[f_band] > 0)):
                #Value above the upper threshold or is the signal being faded in?
                self.uthcnt[f_band] = self.uthcnt[f_band] + 1
                if (g<1):
                    #Has the gate been activated or isn't the signal faded in yet?
                    g = max(self.uthcnt[f_band]/self.attack,g)
                else:
                    g = 1
                
                self.lthcnt[f_band] = 0
            else:
                g = self.g[f_band]
                self.lthcnt[f_band] = 0
                self.uthcnt[f_band] = 0

            self.g[f_band] = g

            #np.mean(self.mem_dB,axis=1)
            self.mov_avg = 0.95*mem_avg + 0.05*xrms.reshape((-1,1))
            self.mem_dB = np.concatenate((self.mem_dB[:,1:],xrms.reshape((-1,1))),axis=1) 

    def calc_energy_bands(self, frame):
        ener_out = np.zeros((self.ener_bands,1))
        if self.ener_type == "UNIFORM":
            freq_per_band = np.int(np.floor(self.n_freq / self.ener_bands))
            band_start = 0
            for i in range(self.ener_bands):
                band_stop = band_start + freq_per_band
                ener_out[i] = np.sum(np.power(np.absolute(frame[band_start:band_stop]), 2), axis=0)
                band_start = band_stop
        elif self.ener_type == "MEL":
            ener_out = get_mel_bands(frame,self.nfft,self.ener_bands,self.fs)
        else:
            print("Energy band grouping not supported")
            exit()
        
        return ener_out

    def unfold_gain_bands(self,g_in):

        g_out = np.zeros((self.nfft,))
        if self.ener_type == "UNIFORM":
            freq_per_band = np.int(np.floor(self.n_freq / self.ener_bands))
            band_start = 0
            for i in range(self.ener_bands):
                band_stop = band_start + freq_per_band
                if g_in[i] > 0:
                    g_out[band_start:band_stop] = g_in[i]
                elif g_in[max(i-1,0)] > 0 or g_in[min(i+1,self.ener_bands-1)] > 0:
                    g_out[band_start:band_stop] = 0.5 * (g_in[max(i-1,0)] + g_in[min(i+1,self.ener_bands-1)])
                
                band_start = band_stop
            
            for i in range(band_stop,self.n_freq):
                g_out[i] = g_in[self.ener_bands - 1]
            
            g_out[self.n_freq:] = np.flip(g_out[1:self.n_freq-1])
        elif self.ener_type == "MEL":
            g_out[:self.n_freq] = mel_to_freq(self.g,self.nfft, self.ener_bands,self.fs)
            g_out[self.n_freq:] = np.flip(g_out[1:self.n_freq-1])
        else:
            print("Energy band grouping not supported")
            exit()
        
        return g_out

    def check_speech(self,x_freq_ener):
        avg_mem = np.mean(self.mov_avg)
        avg_ener = np.mean(x_freq_ener)

        uth = avg_mem + self.utrhold
        lth = avg_mem - self.ltrhold

        if avg_ener > uth:
            self.is_speech = True
        elif avg_ener < lth:
            self.is_speech = False


def create_mel_matrix(nfft,n_coeffs,fs, f_ini, f_end):
    """
    Generate transformation matrix from linear frequency scale to Mel scale using triangular filters
    Params:
        - nfft: Number of FFT coefficients.
        - n_coeffs: Number of Mel coefficients.
        - fs: Sampling frequency.
        - f_ini: Frequency value, center of the first mel coefficient.
        - f_end: Frequency value, center of the last mel coefficient.
    Returns:
        - transf_mat: Transformation matrix. Dimensions -> (n_coeffs, (nfft/2 + 1)).
    """

    mel_bands = np.zeros((2,n_coeffs))
    transf_mat = np.zeros((n_coeffs,int(nfft/2 + 1)))

    mel_ini = 1125*np.log(1+(f_ini/700))
    mel_end = 1125*np.log(1+(f_end/700))
 
    mel_cent = np.linspace(mel_ini, mel_end,(n_coeffs + 2))
    mel_bands[0,:] = mel_cent[0:n_coeffs]
    mel_bands[1,:] = mel_cent[2:(n_coeffs+2)]
    mel_cent = mel_cent[1:(n_coeffs+1)]

    hz_bands = 700 * np.expm1(mel_bands/1125)
    hz_cent = 700 * np.expm1(mel_cent/1125)

    samp_cent = np.floor((nfft/2 + 1) * hz_cent / (fs/2))
    samp_bands = np.floor((nfft/2 + 1) * hz_bands / (fs/2))

    for i in range(n_coeffs):
        for j in range(int(samp_bands[0,i]),int(samp_bands[1,i])):
            if j < samp_cent[i]:
                transf_mat[i,j] = (j - samp_bands[0,i])/(samp_cent[i] - samp_bands[0,i])
            else:
                transf_mat[i,j] = (samp_bands[1,i] - j)/(samp_bands[1,i] - samp_cent[i])

    return transf_mat

def get_mel_bands(spect, nfft, n_mel, fs, f_ini=20, f_end=4000):
    """
    Transform signal spectrum to mel energy bands
    Params:
        - spect: ((nfft/2 + 1), n_frames) spectrogram of input frames.
        - nfft: Number of FFT coefficients.
        - n_coeffs: Number of Mel coefficients.
        - fs: Sampling frequency.
        - f_ini: Frequency value, center of the first mel coefficient.
        - f_end: Frequency value, center of the last mel coefficient.
    Returns:
        - Mel frequency bands.
    """

    transf_mat = create_mel_matrix(nfft, n_mel, fs, f_ini, f_end)

    mel_spect = np.matmul(transf_mat,np.abs(spect))

    return mel_spect

def mel_to_freq(mel_coeff, nfft, n_mel, fs,  f_ini=20, f_end=4000):
    """
    Transform signal spectrum to mel energy bands
    Params:
        - spect: ((nfft/2 + 1), n_frames) spectrogram of input frames.
        - nfft: Number of FFT coefficients.
        - n_coeffs: Number of Mel coefficients.
        - fs: Sampling frequency.
        - f_ini: Frequency value, center of the first mel coefficient.
        - f_end: Frequency value, center of the last mel coefficient.
    Returns:
        - Power spectrum of Mel frequency bands.
    """

    transf_mat = create_mel_matrix(nfft, n_mel, fs, f_ini, f_end)

    freq_out = np.matmul(transf_mat.T,mel_coeff)/2 # Normalize output dividing by 2

    return freq_out

class Normalizer:
    """
    Use leaky integrator to calculate the gated energy and normalize the spoken parts
    """

    def __init__(self,holdtime, ltrhold, utrhold, release, attack, forg_factor = 0.99, norm_level = -10):
        self.forg_factor = forg_factor
        self.holdtime = holdtime
        self.ltrhold = ltrhold
        self.utrhold = utrhold
        self.release = release
        self.attack = attack
        self.norm_level = norm_level
        self.uthcnt = 0
        self.lthcnt = 0
        self.mem = np.finfo(float).eps
        self.g = 0
        self.is_speech = False
    
    def update(self,in_frame):
        ener_linear = np.amax(10 * np.log10(in_frame))
        ener = ener_linear
        #mem_log = 10 * np.log10(self.mem)

        speech_on = self.is_speech
        th_up = self.mem + self.utrhold
        th_dn = self.mem - self.ltrhold

        if (ener <= th_dn) or ((ener < th_up) and (self.lthcnt > 0)):
            #value below the lower threshold?
            self.lthcnt = self.lthcnt + 1
            self.uthcnt = 0
            if self.lthcnt > (self.release + self.holdtime):
                speech_on = False
        elif (ener >= th_up) or ((ener > th_dn) and (self.uthcnt > 0)):
            #Value above the upper threshold or is the signal being faded in?
            self.uthcnt = self.uthcnt + 1
            speech_on = True
            
            self.lthcnt = 0
        else:
            self.lthcnt = 0
            self.uthcnt = 0
        
        self.is_speech = speech_on

        self.mem = (self.forg_factor) * self.mem + (1 - self.forg_factor) * ener#_linear

        if self.is_speech:
            self.calc_gain()
        else:
            self.g = 0

    def calc_gain(self):
        self.g = np.power(10,(self.norm_level - self.mem)/20)

class Compressor:
    def __init__(self, delta_th, forg_factor = 0.95, ini_ener_mem = -60):
        self.delta_th = delta_th
        self.ener_mem = ini_ener_mem
        self.forg_factor = forg_factor

    def filter(self, frame):
        frame_ener = np.mean(np.abs(frame)**2)
        frame_ener_dB = 10 * np.log10(frame_ener)
        if frame_ener_dB > self.ener_mem:
            self.ener_mem = 0.8 * self.ener_mem + 0.2 * frame_ener_dB
        else:
            self.ener_mem = self.forg_factor * self.ener_mem + (1 - self.forg_factor) * frame_ener_dB

        threshold_dB = self.ener_mem + self.delta_th
        threshold = 10.**(threshold_dB/10)
        compression_gain = np.power(frame_ener/threshold,-0.8)

        # Fancy way of thresholding at 1, that is, min(1,x)
        compression_gain -= 1
        compression_gain = 1.-0.5*(-compression_gain + np.abs(-compression_gain))
        #compression_gain *= 2
        return compression_gain

class LinearCompressor:
    def __init__(self, ener_th, gain_one_th, gain_two_th, forg_factor = 0.95, ini_ener_mem = -60):
        self.ener_th = ener_th
        self.gain_one_th = self.ener_th - gain_one_th
        self.gain_two_th = self.ener_th - gain_two_th
        self.ener_mem = ini_ener_mem
        self.forg_factor = forg_factor

    def filter(self, frame):
        frame_ener = np.mean(np.abs(frame)**2)
        frame_ener_dB = 10 * np.log10(frame_ener)
        if frame_ener_dB > self.ener_mem:
            self.ener_mem = 0.8 * self.ener_mem + 0.2 * frame_ener_dB
        else:
            self.ener_mem = self.forg_factor * self.ener_mem + (1 - self.forg_factor) * frame_ener_dB

        if self.ener_th < self.ener_mem:
            compression_gain = self.ener_th - self.ener_mem
        elif self.gain_one_th < self.ener_mem:
            compression_gain = 0.0
        elif self.gain_two_th < self.ener_mem:
            compression_gain = min(self.gain_one_th - self.ener_mem, 6.0)
        else:
            compression_gain = 0.0

        compression_gain = np.power(10,compression_gain/20)
        return compression_gain

#########################################################################################################

class NLMS:

    def __init__(self, n_bands=32, p=8, mu_max=0):
        self.n_bands = n_bands
        self.filt_len = p
        self.W = np.zeros((p, self.n_bands), dtype=np.complex128)
        self.x_mem = np.zeros((p, self.n_bands), dtype=np.complex128)
        self.D = np.zeros((1, self.n_bands), dtype=np.complex128)
        self.E = np.zeros((1, self.n_bands), dtype=np.complex128)
        self.mu = mu_max
        self.mu_max = mu_max

    def update(self, X_n, D_n):
        # update buffers
        self.x_mem[1:, :] = self.x_mem[0:-1, :]
        self.x_mem[0, :] = X_n
        self.D = D_n

        # a priori error
        self.E = self.D - np.diag(np.dot(self.W.conj().T, self.x_mem))

        # compute update
        update = self.mu * np.tile(self.E.conj(), (self.filt_len, 1)) * self.x_mem

        update /= np.tile(np.diag(np.dot(self.x_mem.conj().T, self.x_mem)),
                            (self.filt_len, 1)) + 1e-6

        # update filter coefficients
        self.W += update

    def reset(self):
        self.W = np.zeros((self.filt_len, self.n_bands), dtype=np.complex64)
        #self.x_mem = np.zeros((self.filt_len,self.n_bands), dtype = np.complex64)

    def update_mem(self, x_mem_new):
        for i in reversed(range(1, self.filt_len)):
            self.x_mem[i, :] = self.x_mem[i - 1, :]
        self.x_mem[0, :] = x_mem_new


class TwoPathFreq_filter:

    def __init__(self, n_bands=32, p=8, mu_max=0, beta=0.8, gamma=0.1, hang_t =2):
        self.R_ey = 0
        self.R_yy = 0
        self.n_bands = n_bands
        self.filt_len = p
        self.W = np.zeros((p,self.n_bands), dtype = np.complex64)
        self.x_mem = np.zeros((p,self.n_bands), dtype = np.complex64)
        self.D = np.zeros((1,self.n_bands), dtype = np.complex64)
        self.mu = mu_max
        self.background = NLMS(n_bands = self.n_bands, p = self.filt_len, mu_max = self.mu)
        self.foreground = NLMS(n_bands = self.n_bands, p = self.filt_len, mu_max = self.mu)

        self.e_f = np.zeros((1,self.n_bands), dtype = np.complex64)
        self.e_b = np.zeros((1,self.n_bands), dtype = np.complex64)

        self.beta = beta
        self.gamma = gamma
        self.hang_t = hang_t

        self.update_back = True
        self.update_count = 0
    
    def update(self, x_n, d_n):
        # update buffers
        self.x_mem[1:, :] = self.x_mem[0:-1, :]
        self.x_mem[0, :] = x_n
        self.D = d_n

        self.background.update(x_n, d_n)
        self.e_b = self.D - np.diag(np.dot(self.background.W.conj().T, self.x_mem))

        if self.update_back:
            self.foreground.W = self.background.W# update(x_n, d_n)
        self.e_f = self.D - np.diag(np.dot(self.foreground.W.conj().T, self.x_mem))
        self.W = self.foreground.W

        self.transfer_control()      

    def L_j(self,A):
        out = np.sum(np.abs(A))
        return out

    def transfer_control(self):
        x_control = self.x_mem[self.hang_t,:]

        #Condition 1
        c_i = (self.L_j(self.e_b) < self.beta * self.L_j(self.e_f))
        #Condition 2
        c_ii = (self.L_j(self.e_b) < self.gamma * self.L_j(self.D))
        #Condition 3
        c_iii = (self.L_j(self.D) < self.L_j(x_control))

        if (c_i and c_ii and c_iii):
            self.update_count += 1
        else:
            self.update_count = 0

        if self.update_count >= 3:
            self.update_count = 2
            self.update_back = True
        else:
            self.update_back = False


class CT_detect:
    def __init__(self, num_frames, frame_len, ratio_th = None):
        self.num_frames = num_frames
        self.ratio_th = ratio_th if not (ratio_th == None) else 0.2
        self.frame_len = frame_len

        self.mem_farend = np.zeros((int(self.num_frames*self.frame_len),1))

        self.state = 0

        self.count = 0
        self.cnt_coef = 0

    def detect(self, near_frame, far_frame):
        self.mem_farend = np.concatenate((self.mem_farend[self.frame_len:,:], far_frame.reshape((-1,1))))

        maxval_near = np.amax(np.abs(near_frame))
        maxval_far = np.amax(np.abs(self.mem_farend))

        if self.state == 0:
            if maxval_near > (self.ratio_th * maxval_far):
                self.state = 1
                self.count = 0
            
            self.count += 1

        elif self.state == 1:
            if maxval_near < (0.35 * self.ratio_th * maxval_far):
                self.state = 0
                self.count = 0

            self.count += 1
        
        self.count = min(self.count,10)
        self.cnt_coef = np.clip(self.count / 10, 0, 1)

        return self.state, maxval_near/maxval_far


class xcorr_CT_detect:
    def __init__(self, num_frames, frame_len, ratio_th = None):
        self.num_frames = num_frames
        self.ratio_th = ratio_th if not (ratio_th == None) else 0.2
        self.frame_len = frame_len

        self.mem_farend = np.zeros((int(self.num_frames*self.frame_len),1))

        self.state = 0

        self.count = 0
        self.cnt_coef = 0

    def detect(self, near_frame, far_frame):
        self.mem_farend = np.concatenate((self.mem_farend[self.frame_len:,:], far_frame.reshape((-1,1))))

        xcorr = np.correlate(near_frame.squeeze(),self.mem_farend.squeeze()) / (np.sqrt(np.sum(np.square(near_frame))) * np.sqrt(np.sum(np.square(self.mem_farend))))

        maxxcorr = np.amax(np.abs(xcorr))
        if self.state == 0:
            if maxxcorr > (self.ratio_th * maxxcorr):
                self.state = 1
                self.count = 0
            
            self.count += 1

        elif self.state == 1:
            if maxxcorr < (0.5 * self.ratio_th * maxxcorr):
                self.state = 0
                self.count = 0

            self.count += 1
        
        self.count = min(self.count,10)
        self.cnt_coef = np.clip(self.count / 10, 0, 1)

        return self.state, maxxcorr


class ncc_CT_detect:
    def __init__(self, forg_factor=0.9, ratio_th=None):
        self.ratio_th = ratio_th if not (ratio_th == None) else 0.2
        self.forg_factor = forg_factor

        self.decision = True

        self.red = 0
        self.sigma_d = 0
        self.sigma_e = 0
        self.decision_ratio = 0

        self.count = 0

    def detect(self, near_frame, error_frame):
        newred = np.abs(np.matmul(error_frame.T.numpy(), near_frame.numpy()))
        newsigmad = np.abs(np.matmul(near_frame.T.numpy(), near_frame.numpy()))
        newsigmae = np.abs(np.matmul(error_frame.T.numpy(), error_frame.numpy()))
        self.red = self.forg_factor * self.red + (1 - self.forg_factor) * newred
        self.sigma_d = self.forg_factor * self.sigma_d + (1 - self.forg_factor) * newsigmad
        self.sigma_e = self.forg_factor * self.sigma_e + (1 - self.forg_factor) * newsigmae

        aux_ratio = (self.red / (np.sqrt(self.sigma_d*self.sigma_e) + 1e-6))
        self.decision_ratio = 1 - aux_ratio
        
        if self.decision_ratio < self.ratio_th:
            self.decision = False
            self.count = 0
        else:
            self.count += 1
            if self.count > 5:
                self.decision = True

        return self.decision, self.decision_ratio.item()


def norm_xcorr(a,b):
    return np.sum(a * b)/(np.sqrt(np.sum(np.square(a)))*np.sqrt(np.sum(np.square(b))) + 1e-12)

class TP_NLMS:
    def __init__(self, num_bands, filter_length, mu_max, beta, gamma, hang_t):
        self.num_bands = num_bands
        self.filt_length = filter_length
        self.mu_max = mu_max
        self.beta = beta
        self.gamma = gamma
        self.hang_t = hang_t

        self.w_bg = np.zeros((filter_length, num_bands), dtype = np.complex64)
        self.w_fg = np.zeros((filter_length, num_bands), dtype = np.complex64)
        self.e_bg = np.zeros((1, num_bands), dtype = np.complex64)
        self.e_fg = np.zeros((1, num_bands), dtype = np.complex64)

        self.x_mem = np.zeros((filter_length, num_bands), dtype = np.complex64)
        self.d = np.zeros((1,num_bands))

    def update_memory(self,new_x):
        # update buffers
        self.x_mem[1:, :] = self.x_mem[0:-1, :]
        self.x_mem[0, :] = new_x

    def update_bg_weights(self):
        update = self.mu_max * np.tile(self.e_bg.conj(), (self.filt_length, 1)) * self.x_mem

        update /= np.tile(np.diag(np.dot(self.x_mem.conj().T, self.x_mem)),
                            (self.filt_length, 1)) + 1e-6

        # update filter coefficients
        self.w_bg += update
    
    def transfer_bg_to_fg(self):
        self.w_fg = self.w_bg

    def transfer_fg_to_bg(self):
        self.w_bg = self.w_fg
    
    def filter_bg(self):
        self.e_bg = self.d - np.diag(np.dot(self.w_bg.conj().T, self.x_mem))
    
    def filter_fg(self):
        self.e_fg = self.d - np.diag(np.dot(self.w_fg.conj().T, self.x_mem))

    def process_frame(self,new_d, new_x, override_update):
        self.d = new_d
        self.update_memory(new_x)
        #print(override_update)
        self.filter_fg()
        self.filter_bg()
        self.update_bg_weights()
        self.filter_bg()

        decision, a, b, c = self.transfer_control()

        if (not override_update):
            self.transfer_bg_to_fg()
            return self.e_bg, a, b, c
        else:
            #self.transfer_fg_to_bg()
            return self.e_fg, a, b, c

    def L_j(self,A):
        out = np.sum(np.sqrt(np.abs(np.dot(A.T.conj(),A))))
        return out

    def transfer_control(self):
        x_control = self.x_mem[self.hang_t,:]

        leb = self.L_j(self.e_bg).item()
        lef = self.L_j(self.e_fg).item()
        ld = self.L_j(self.d).item()
        lx = self.L_j(x_control).item()

        print(leb, lef, ld, lx)

        #Condition 1
        c_i = (leb < self.beta * lef)
        #Condition 2
        c_ii = (leb < self.gamma * ld)
        #Condition 3
        c_iii = (ld < lx)

        if (c_i and c_ii and c_iii):
            self.update_count += 1
        else:
            self.update_count = 0

        if self.update_count >= 2:
            self.update_count = 1
            self.update_back = True
        else:
            self.update_back = False

        return self.update_back, leb/(lef + 1e-12), leb/(ld + 1e-12), ld/(lx + 1e-12)
