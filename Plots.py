import soundfile as sf
import numpy as np
from Data.DatabaseBuilder import make_frames
from Data.DatabaseBuilder import reconstruct_speech
from scipy.signal import butter, filtfilt, welch, stft, istft
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
import os
import glob
import pandas as pd
import json
from database_helper import data_helper


def plot_regression_line(y, x, color):

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, color)

    return m, b


def plot_mos(path):

    mos_results = pd.read_csv(path, sep=',')
    mos_results['rating_score'] = mos_results['rating_score']
    mos_results = mos_results[["rating_stimulus", "rating_score"]]

    reference = mos_results.loc[mos_results['rating_stimulus'] == 'reference']
    IRM = mos_results.loc[mos_results['rating_stimulus'] == 'C3']
    Mixture = mos_results.loc[mos_results['rating_stimulus'] == 'C2']
    Iso_Net = mos_results.loc[mos_results['rating_stimulus'] == 'C1']

    y = np.linspace(0, 25, 26)
    reference_np = np.mean(np.reshape(reference['rating_score'].to_numpy(), (-1, 26)), axis=0)
    IRM_np = np.mean(np.reshape(IRM['rating_score'].to_numpy(), (-1, 26)), axis=0)
    Mixture_np = np.mean(np.reshape(Mixture['rating_score'].to_numpy(), (-1, 26)), axis=0)
    Iso_Net_np = np.mean(np.reshape(Iso_Net['rating_score'].to_numpy(), (-1, 26)), axis=0)

    columns = ['Reference', 'Mixture', 'IRM', 'Iso-Net']
    test = reference["rating_score"].reset_index(drop=True)
    mos_results_plot = pd.DataFrame(data={'Reference': reference["rating_score"].reset_index(drop=True), 'Mixture': Mixture["rating_score"].reset_index(drop=True), 'IRM': IRM["rating_score"].reset_index(drop=True), 'Iso-Net': Iso_Net["rating_score"].reset_index(drop=True)})

    mean_mos = mos_results_plot.mean()
    var_mos = mos_results_plot.std()

    error_ref = 1.95 * np.std(reference) / np.sqrt(len(reference))
    error_ref = error_ref.iloc[0]

    error_IRM = 1.95 * np.std(IRM) / np.sqrt(len(IRM))
    error_IRM = error_IRM.iloc[0]

    error_mix = 1.95 * np.std(Mixture) / np.sqrt(len(Mixture))
    error_mix = error_mix.iloc[0]

    error_iso = 1.95 * np.std(Iso_Net) / np.sqrt(len(Iso_Net))
    error_iso = error_iso.iloc[0]

    samples = np.linspace(0, 25, 26)
    plt.plot(reference_np, 'orange', label='Reference')
    plt.fill_between(samples, reference_np - error_ref, reference_np + error_ref, alpha=0.1, color='orange')
    plot_regression_line(reference_np, y, 'orange')

    plt.plot(IRM_np, 'g', label='IRM')
    plt.fill_between(samples, IRM_np - error_IRM, IRM_np + error_IRM, alpha=0.1, color='g')
    plot_regression_line(IRM_np, y, 'g')

    plt.plot(Mixture_np, 'b', label='Mixture')
    plt.fill_between(samples, Mixture_np - error_mix, Mixture_np + error_mix, alpha=0.1, color='b')
    plot_regression_line(Mixture_np, y, 'b')

    plt.plot(Iso_Net_np, 'r', label='Iso-Net')
    plt.fill_between(samples, Iso_Net_np - error_iso, Iso_Net_np + error_iso, alpha=0.1, color='r')
    plot_regression_line(Iso_Net_np, y, 'r')

    plt.xlabel('Listening Sample', fontsize=18)
    plt.ylabel('MUSHRA Score', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc=2)
    plt.grid(True)
    plt.show()
    plt.savefig('Data/Plots_Thesis/MUSHRAOverTime.pdf')

    boxprops = dict(linewidth=2)
    ax = mos_results_plot.plot.box(figsize=(10, 8), boxprops=boxprops, showfliers=False, whis=0)
    ax.plot(linewidth=2)
    ax.set_ylabel('MUSHRA Score', fontsize=18)
    ax.set_ylim(0, 110)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    #plt.grid(True)
    plt.show()
    plt.savefig('Data/Plots_Thesis/MUSHRAListeningTest.pdf')

    test = 1


def plot_speech_mixture(nearend, farend):
    window = 512
    hop = 256
    fs = 16000

    clean_nearend = (nearend - np.mean(nearend)) / np.max(np.abs(nearend))
    clean_farend = (farend - np.mean(farend)) / np.max(np.abs(farend))

    return clean_nearend, clean_farend


def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    y = filtfilt(b, a, data)
    return y


# IBM
def tests():
    file1, samplerate = sf.read('C:/Users/silas/PycharmProjects/VoiceIsolation/Data/ForPLot/TestSeparation_clean27.wav')
    file2, _ = sf.read('C:/Users/silas/PycharmProjects/VoiceIsolation/Data/ForPLot/TestSeparation_FarEndClean27.wav')

    file3, _ = sf.read('C:/Users/silas/PycharmProjects/VoiceIsolation/Data/ForPLot/TestSeparation_CompleteScenario27.wav')

    file1 = file1 / max(abs(file1))
    file2 = file2 / max(abs(file2))
    file3 = file3 / max(abs(file3))

    ref_signal1 = stft(file1, samplerate, nfft=1024, nperseg=512)
    ref_signla2 = stft(file2, samplerate, nfft=1024, nperseg=512)
    ref_signal3 = stft(file3, samplerate, nfft=1024, nperseg=512)

    ref_signal3 = abs(ref_signal3[2]) / np.matrix(abs(ref_signal3[2])).max()
    ref_signal1 = abs(ref_signal1[2]) / np.matrix(abs(ref_signal1[2])).max()
    ref_signla2 = abs(ref_signla2[2]) / np.matrix(abs(ref_signla2[2])).max()

    # IBM
    threshold = 0.08
    diff_signal = (abs(ref_signal1) - abs(ref_signla2))
    diff_signal = diff_signal / np.max(np.abs(diff_signal))
    diff_signal[diff_signal > threshold] = 1
    diff_signal[diff_signal < threshold] = 0
    plt.imshow(abs(diff_signal))
    plt.xticks(np.linspace(0, 377, 7).astype(int), np.linspace(0, 6, 7).astype(int))
    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Frequency [Hz]', fontsize=14)
    plt.yticks(np.linspace(0, 513, 9).astype(int), np.linspace(0, 8000, 9).astype(int))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
    plt.savefig('Data/Plots_Thesis/IBMMask.pdf')

    #IRM

    mask_IRM = ((abs(ref_signal1) / (abs(ref_signal1) + abs(ref_signla2))))**0.5
    masked_sig = ref_signal3 * mask_IRM
    reconstructed_sig = istft(masked_sig, fs=samplerate, nfft=1024, nperseg=512)
    reconstruct_test = istft(ref_signal3, fs=samplerate, nfft=1024, nperseg=512)

    sf.write('Data/EvalSamples/Test21111.wav', reconstruct_test[1], samplerate)
    sf.write('Data/EvalSamples/IRMMasked.wav', reconstructed_sig[1], samplerate)

    test = np.round(np.linspace(0, 8000, 9))

    plt.imshow(abs(mask_IRM))
    plt.xticks(np.linspace(0, 377, 7).astype(int), np.linspace(0, 6, 7).astype(int))
    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Frequency [Hz]', fontsize=14)
    plt.yticks(np.linspace(0, 513, 9).astype(int), np.linspace(0, 8000, 9).astype(int))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
    plt.savefig('Data/Plots_Thesis/MASKIRM.pdf')

    plt.imshow(abs(ref_signal1))
    plt.xticks(np.linspace(0, 377, 7).astype(int), np.linspace(0, 6, 7).astype(int))
    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Frequency [Hz]', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(np.linspace(0, 513, 9).astype(int), np.linspace(0, 8000, 9).astype(int))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
    plt.savefig('Data/Plots_Thesis/SampleSpectrum.pdf')

    tst = 1

    audio_speaker1, audio_speaker2, fs = data_helper.loader()
    plot_speech_mixture(audio_speaker1, audio_speaker2)

    test_win_inv = 1 / np.hanning(512)

    # Filter requirements.
    T = 6         # Sample Period
    fs = 8000       # sample rate, Hz
    cutoff = 150     # desired cutoff frequency of the filter, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 10
    n = int(T * fs) # total number of samples

    results_pesq = []
    results_stoi = []
    path = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamples'

    eval_samples = len(glob.glob(path + "/*.wav")) // 4

    file1, samplerate = sf.read('C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamples/TestSeparation0.wav')
    file2, samplerate = sf.read('C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamples/TestSeparation_clean0.wav')
    file1 = file1 / max(abs(file1))


    file2, samplerate = sf.read('C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase/SpeakerSet_NearEnd20.wav')

    test_subtraction = file2 + file2 * (-0.9)
    sf.write('Data/SubtractionTest.wav', test_subtraction, 16000)

    f, Pxx_den = welch(file1, samplerate, nperseg=1024)

    plt.semilogy(f, Pxx_den)
    plt.ylim([0.5e-8, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

    sf.write('Data/TestSeparation_filtered.wav', file_out, 16000)

    for i in range(eval_samples):

        reconstruct_signal, samplerate = sf.read(os.path.join(path, 'TestSeparation{}.wav'.format(i)))
        reconstruct_signal = reconstruct_signal / max(abs(reconstruct_signal))

        reconstruct_clean, samplerate = sf.read(os.path.join(path, 'TestSeparation_clean{}.wav'.format(i)))
        reconstruct_clean = reconstruct_clean / max(abs(reconstruct_clean))

        result = pesq(8000, reconstruct_clean, reconstruct_signal, 'nb')

        stoi_1 = stoi(reconstruct_clean, reconstruct_signal, 8000, extended=False)

        results_pesq.append(result)
        results_stoi.append(stoi_1)

    #Calculate Standard Deviation

    variance_pesq = np.std(results_pesq)
    variance_stoi = np.std(results_stoi)


    print('Average PESQ: {}'.format(np.mean(results_pesq)))
    print('Standard Dev PESQ: {}'.format(variance_pesq))

    print('Average STOI: {}'.format(np.mean(results_stoi)))
    print('Standard Dev STOI: {}'.format(variance_stoi))

    # Test Windowing

    file_ones = np.ones(48000)

    frames_ones = make_frames(file_ones, 512, 256, 'Half')

    test_hamming = frames_ones[0, 0:256] + frames_ones[0, 256:512]

    frames_recon = reconstruct_speech(frames_ones, 'Half', 512, 256)
    frames_recon = frames_recon[:len(file_ones)]

    t_compare = np.linspace(0, len(file_ones) / fs, num=len(file_ones))
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t_compare, file_ones)
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Reference Signal')
    axs[0].grid(True)

    axs[1].set_ylabel('Network Output')
    axs[1].set_xlabel('Time [s]')
    axs[1].plot(t_compare, frames_recon)
    axs[1].grid(True)

    fig.tight_layout()
    plt.show()

    filtered_signal = butter_lowpass_filter(reconstruct_signal, cutoff, fs, order)

    sf.write('Data/TestSeparation_filtered.wav', filtered_signal, 8000)

    frames = make_frames(reconstruct_signal, 512, 256)
    frames = np.squeeze(frames)

    test = 1


def plot_windowing_function():
    test_inverse = np.ones(512)
    test_window = np.hanning(512)
    test_trian = np.bartlett(512)
    test_hann = np.hamming(512)

    plt.plot(test_inverse)
    plt.plot(test_window)
    plt.plot(test_trian)
    plt.plot(test_hann)
    plt.xticks(np.linspace(0, 512, 9).astype(int), np.linspace(-256, 256, 9).astype(int))
    plt.legend(['Rectangular', 'Hann', 'Triangular', 'Hamming'])
    plt.grid(True)
    plt.xlabel('Sample index from Center', fontsize=14)
    plt.ylabel('Magnitude', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    plt.savefig('Data/Plots_Thesis/WindowingFunction.pdf')


def calculuate_sisnr(mix, clean):

    EPS = 1e-8
    clean_zero_mean = clean - np.mean(clean)

    mix_zero_mean = mix - np.mean(mix)

    pair_wise_dot = np.sum(clean_zero_mean * mix_zero_mean)

    s_target_energy = np.sum(clean_zero_mean ** 2) + EPS
    pair_wise_proj = pair_wise_dot * clean_zero_mean / s_target_energy

    e_noise = mix_zero_mean - pair_wise_proj

    pair_wise_sdr = np.sum(pair_wise_proj ** 2) / (np.sum(e_noise ** 2) + EPS)
    si_snr = 10 * np.log10(pair_wise_sdr + EPS)

    return si_snr


def plot_database_distribution():

    f = open('C:/Users/silas/NoisyOverlappingSpeakers/Database/database_overview.json')
    database = json.load(f)

    rooms = database['rooms']

    volumes = [rooms[i][0]*rooms[i][1]*rooms[i][2] for i in range(len(rooms))]

    plt.hist(volumes, bins=300)
    plt.xlabel('Room Volume [m3]', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('Data/Plots_Thesis/DatabaseDistribution.pdf')
    plt.show()

    speaker_set = [1, 2, 3, 4]

    delay_times = []

    for k in range(len(rooms)):
        delay_time = (np.sqrt(sum((np.asarray(database['Near End Speaker 1 Position'][k]) - np.asarray(database['Far End Speaker 2 Position'][k])) ** 2)) / 343) * 1000
        delay_times.append(delay_time)

    plt.hist(delay_times, bins=300)
    plt.xlabel('Delay between speakers [ms]', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('Data/Plots_Thesis/DelayTimes.pdf')
    plt.show()


    speakers = np.random.choice(speaker_set, 20000)

    plt.hist(speakers, bins=4, rwidth=0.5)
    plt.xlabel('Distribution', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(np.linspace(1.5, 3.5, 4).astype(int), ['M-M', 'F-F', 'F-M', 'M-F'])
    plt.show()
    plt.savefig('Data/Plots_Thesis/MFDistribution.pdf')


def plot_activationFunctions():
    x = np.linspace(-2, 2, 100)

    sigmoid = []
    relu = []
    tanh = []
    prelu = []

    for i in range(len(x)):
        sigmoid.append(1 / (1 + np.exp(-x[i])))
        relu.append(np.maximum(0, x[i]))
        tanh.append(np.tanh(x[i]))

        if x[i] >= 0:
            prelu.append(x[i])
        else:
            prelu.append(0.3*x[i])

    plt.plot(x, sigmoid, linewidth=2)
    plt.plot(x, relu, linewidth=2)
    plt.plot(x, tanh, linewidth=2)
    plt.plot(x, prelu, linewidth=2)
    plt.legend(['Sigmoid', 'ReLU', 'TanH', 'PReLU'], prop={'size': 12})
    plt.xlabel('$\phi(x_{\lambda})$', fontsize=14)
    plt.ylabel('${\phi}\'(\phi)$', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()
    plt.savefig('Data/Plots_Thesis/ActivationFunction.pdf')


def calculatePESQandStoi(path):

    results_pesq = []
    results_stoi = []
    results_sisnr = []

    eval_samples = len(glob.glob(path + "/*.wav")) // 4

    for i in range(1, 149):
        print('Sample: {}'.format(i))
        reconstruct_signal, samplerate = sf.read(os.path.join(path, 'TestSeparation{}.wav'.format(i)))
        reconstruct_signal = reconstruct_signal / max(abs(reconstruct_signal))

        reconstruct_clean, samplerate = sf.read(os.path.join(path, 'TestSeparation_clean{}.wav'.format(i)))
        reconstruct_clean = reconstruct_clean / max(abs(reconstruct_clean))

        result = pesq(16000, reconstruct_clean, reconstruct_signal, 'nb')

        stoi_1 = stoi(reconstruct_clean, reconstruct_signal, 16000, extended=False)

        sisnr = calculuate_sisnr(reconstruct_clean, reconstruct_signal)

        results_pesq.append(result)
        results_stoi.append(stoi_1)
        results_sisnr.append(sisnr)

        # Calculate Standard Deviation

    variance_pesq = np.std(results_pesq)
    variance_stoi = np.std(results_stoi)
    variance_sisnr = np.std(results_sisnr)

    print('Average PESQ: {}'.format(np.mean(results_pesq)))
    print('Standard Dev PESQ: {}'.format(variance_pesq))

    print('Average STOI: {}'.format(np.mean(results_stoi)))
    print('Standard Dev STOI: {}'.format(variance_stoi))

    print('Average SiSNR: {}'.format(np.mean(sisnr)))
    print('Standard Dev SiSNR: {}'.format(variance_sisnr))

    return np.mean(results_pesq), variance_pesq, np.mean(results_stoi), variance_stoi, np.mean(sisnr), variance_sisnr


def calculate_MI(path):
    mutual_information_network = []
    mutual_information_mixture = []

    eval_samples = len(glob.glob(path + "/*.wav")) // 5

    for i in range(148):
        print('Sample: {}'.format(i))

        target, samplerate = sf.read(os.path.join(path, 'TestSeparation{}.wav'.format(i)))
        target = np.expand_dims(target[:, 1] / max(abs(target[1])), axis=1)

        clean, samplerate = sf.read(os.path.join(path, 'TestSeparation_clean{}.wav'.format(i)))
        clean = clean[:, 1] / max(abs(clean[:, 1]))

        mixture, samplerate = sf.read(os.path.join(path, 'TestSeparation_CompleteScenario{}.wav'.format(i)))
        mixture = np.expand_dims(mixture[:, 1] / max(abs(mixture[:, 1])), axis=1)

        mi_before = mutual_info_regression(mixture, clean)
        mi_after = mutual_info_regression(target, clean)

        mutual_information_network.append(mi_before)
        mutual_information_mixture.append(mi_after)

    mean_before = np.mean(mutual_information_mixture)
    mean_after = np.mean(mutual_information_network)
    var_before = np.var(mutual_information_mixture)
    var_after = np.var(mutual_information_network)

    print('BeforeNetwork Mean MI: {}'.format(mean_before))
    print('AfterNetwork Mean MI: {}'.format(mean_after))
    print('BeforeNetwork Var MI: {}'.format(var_before))
    print('AfterNetwork Var MI: {}'.format(var_after))

    return {'BeforeNetwork Mean MI': mean_before, 'AfterNetwork Mean MI': mean_after, 'BeforeNetwork Var MI':var_before, 'AfterNetwork Var MI':var_after}


def time_series_network(path):

    target, _ = sf.read(os.path.join(path, 'TestSeparation5.wav'))
    clean_near, _ = sf.read(os.path.join(path, 'TestSeparation_clean5.wav'))
    clean_far, samplerate = sf.read(os.path.join(path, 'TestSeparation_FarEnd5.wav'))

    min_length = np.min([len(target), len(clean_far), len(clean_near)])

    if min_length > samplerate * 6:
        min_length = samplerate * 6

    nearend = clean_near[:min_length]
    farend = clean_far[:min_length]
    target = target[:min_length]

    near_end, far_end = plot_speech_mixture(nearend, farend)
    time_steps = np.linspace(0, 6, num=samplerate * 6)
    #f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    f, (ax1, ax2) = plt.subplots(2, 1)
    line1 = ax1.plot(time_steps, 0.4 * far_end, 'r', alpha=0.4)
    line2 = ax1.plot(time_steps, near_end, 'b', alpha=0.4)
    #line3 = ax3.plot(time_steps, 0.4 * near_end, 'b', alpha=0.4)
    #line4 = ax3.plot(time_steps, far_end, 'r', alpha=0.4)

    line1[0].set_label('Interferer')
    line2[0].set_label('Target')
    ax1.set_ylabel('Amplitude', fontsize=14)
    ax1.grid(True)
    ax1.set_title('Speech Mixture Signal', fontsize=14)

    ax2.plot(time_steps, target, 'b', alpha=0.4)
    ax2.set_xlabel('Time [s]', fontsize=14)
    ax2.set_ylabel('Amplitude', fontsize=14)
    ax2.grid(True)
    ax2.set_title('Isolated Speech Signal', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    #ax3.set_xlabel('Time [s]', fontsize=14)
    #ax3.set_ylabel('Amplitude', fontsize=14)
    #ax3.grid(True)
    #ax3.set_title('FarEnd Speech Signal', fontsize=14)

    #ax4.plot(time_steps, farend, 'r', alpha=0.4)
    #ax4.set_xlabel('Time [s]', fontsize=14)
    #ax4.set_ylabel('Amplitude', fontsize=14)
    #ax4.grid(True)
    #ax2.set_title('CleanFarEnd', fontsize=14)

    handles, labels = ax1.get_legend_handles_labels()
    f.legend(handles, labels, prop={"size": 14})

    plt.show()
    plt.savefig('Data/Plots_Thesis/TimeSeries.pdf')
    tst = 1


def get_spectrogram(path):
    target, _ = sf.read(os.path.join(path, 'TestSeparation5.wav'))
    clean_near, _ = sf.read(os.path.join(path, 'TestSeparation_clean5.wav'))
    clean_far, samplerate = sf.read(os.path.join(path, 'TestSeparation_FarEnd5.wav'))

    min_length = np.min([len(target), len(clean_far), len(clean_near)])

    if min_length > samplerate * 6:
        min_length = samplerate * 6

    nearend = clean_near[:min_length]
    farend = clean_far[:min_length]
    target = target[:min_length]

    near_end, far_end = plot_speech_mixture(nearend, farend)

    mixture = near_end + 0.4 * far_end
    spectrum_mix = stft(mixture[:, 0], samplerate, nfft=1024, nperseg=512)
    specrum_near_end = stft(near_end[:, 0], samplerate, nfft=1024, nperseg=512)
    spectrum_far_end = stft( 0.4 * far_end[:, 0], samplerate, nfft=1024, nperseg=512)

    spectrum_isonet = stft(target[:, 0], samplerate, nfft=1024, nperseg=512)

    plt.imshow(abs(spectrum_far_end[2]), cmap='Reds', alpha=0.5)
    plt.imshow(abs(specrum_near_end[2]), cmap='Blues', alpha=0.5)
    plt.clim(0, 0.05)
    plt.xticks(np.linspace(0, 377, 7).astype(int), np.linspace(0, 6, 7).astype(int))
    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Frequency [Hz]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(prop={'size': 12})
    plt.yticks(np.linspace(0, 513, 9).astype(int), np.linspace(0, 8000, 9).astype(int))
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

    plt.imshow(abs(spectrum_isonet[2]))
    plt.xticks(np.linspace(0, 377, 7).astype(int), np.linspace(0, 6, 7).astype(int))
    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Frequency [Hz]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(prop={'size': 12})
    plt.yticks(np.linspace(0, 513, 9).astype(int), np.linspace(0, 8000, 9).astype(int))
    plt.gca().invert_yaxis()
    plt.colorbar()

    plt.show()
    plt.savefig('Data/Plots_Thesis/TimeSeries.pdf')
    test = 1


def plot_preprocessing():

    example_window = np.hamming(1024)
    file2, samplerate = sf.read('C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase/SpeakerSet_NearEnd20.wav')

    file2 = file2[6000:10100]

    framed_audio = make_frames(file2, 1024, 1024, 'HAMMING')

    x_time = np.linspace(0, 0.28, 4100)
    x_frame = np.linspace(0, 1023/16000, 1024)
    x_frame1 = np.linspace(1024/16000, 2047/16000, 1024)
    x_frame2 = np.linspace(2048/16000, 3071/16000, 1024)
    x_frame3 = np.linspace(3072/16000, 4095/16000, 1024)

    plt.plot(x_time, file2+10, label='Input Signal')

    plt.plot(x_frame, framed_audio[0] + 8, color='r', alpha=0.6, label='Subframe')
    plt.plot(x_frame, example_window + 8, '--', color='k', alpha=0.6, label='Windowing')

    plt.plot(x_frame1, framed_audio[1] + 6, color='r', alpha=0.6)
    plt.plot(x_frame1, example_window + 6, '--', color='k', alpha=0.6)

    plt.plot(x_frame2, framed_audio[2] + 4, color='r', alpha=0.6)
    plt.plot(x_frame2, example_window + 4, '--', color='k', alpha=0.6)

    plt.plot(x_frame3, framed_audio[3] + 2, color='r', alpha=0.6)
    plt.plot(x_frame3, example_window + 2, '--', color='k', alpha=0.6)

    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(prop={'size': 12})

    ax = plt.gca()
    ax.axes.yaxis.set_ticks([])

    plt.show()
    plt.savefig('Data/Plots_Thesis/PreProcessing.pdf')
    test = 1


def plot_for_presi():
    fontsize = 25

    convtas = [5.1, 3.22, 4.03]

    percep = [8.5, 2.41, 2.4]
    iso = [3.7, 3.7, 0]
    labels = ['Network Size [mio.]', 'PESQ', 'MOS']

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, convtas, width, label='ConvTas-Net', color='b')
    rects2 = ax.bar(x, percep, width, label='personalized Percep-Net', color='r')
    rects3 = ax.bar(x + width, iso, width, label='Iso-Net', color='g')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores', fontsize=fontsize)
    # ax.set_title('Scores by PESQ and Si-SNR')
    ax.set_xticks(x, )
    ax.set_ylim([0, 10])
    ax.set_xticklabels(labels, fontsize=fontsize)

    for item in ([ax.yaxis.label] +
                ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    ax.legend(prop={'size': fontsize})

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=fontsize)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    plt.show()
    plt.savefig('Data/Plots_Thesis/BarPlot.pdf')

    x_time = [128, 256, 512]
    filters_PESQ = [2.6, 2.9, 3.7]
    filters_snr = [11, 17.5, 26.58]
    network_size = [1.101, 4.372, 17.394]

    plt.legend(['Si-SNR', 'PESQ'], prop={'size': 18})

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of filters for each convolutional layer', fontsize=18)
    ax1.set_ylabel('Si-SNR', color=color, fontsize=18)
    #ax1.set_xticks(fontsize=18)
    #ax1.set_yticks(fontsize=18)
    ax1.plot(x_time, filters_snr, color='b', label='Si-SNR', linewidth=2, marker='s')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 30])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('PESQ Score', color=color, fontsize=18)  # we already handled the x-label with ax1
    ax2.plot(x_time, filters_PESQ, color='r', label='PESQ', linewidth=2, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 4.5])

    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.set_ylabel('Network Size [mio.]', color=color, fontsize=18)  # we already handled the x-label with ax1
    ax3.plot(x_time, network_size, color='g', label='Network Size [mio.]', linewidth=2, marker='s')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_ylim([0, 20])
    ax3.spines['right'].set_position(('data', 0.2))

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(18)

    for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
                 ax3.get_xticklabels() + ax3.get_yticklabels()):
        item.set_fontsize(18)


    fig.tight_layout()
    plt.grid(True)
    plt.savefig('Data/Plots_Thesis/FiltersizedPrese.pdf')
    plt.show()

    alignment_mask = [3.3]
    sub_mask = [3.4]
    inverse_mask = [3.7]

    #X = np.arange(2)
    #fig = plt.figure()
    #ax = fig.add_axes([0, 0, 1, 1])
    #ax.bar(X + 0.00, alignment_mask, color='b', width=0.25)
    #ax.bar(X + 0.25, sub_mask, color='g', width=0.25)
    #ax.bar(X + 0.50, inverse_mask, color='r', width=0.25)
    #ax.set_xticks(X, labels=['PESQ', 'Si-SNR'])
    #plt.legend(['Alignment Mask', 'Subtractive Mask', 'Inverse Mask'])
    #plt.ylabel('Score', fontsize=14)
    #plt.grid(True)
    #plt.show()
    #plt.savefig('Data/Plots_Thesis/BarDiagramm_ThreeArchitectures.pdf')

    labels = ['PESQ']

    x = np.arange(1, 2, 1) # the label locations
    x_plot = np.arange(1)
    width = 1  # the width of the bars

    fig, ax = plt.subplots(figsize=(7,10))
    rects1 = ax.bar(0, alignment_mask, width, label='Alignment Mask', color='b')
    rects2 = ax.bar(1, sub_mask, width, label='Subtractive Mask', color='r')
    rects3 = ax.bar(2, inverse_mask, width, label='Inverse Mask', color='g')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('PESQ Score', fontsize=25)
    #ax.set_title('Scores by PESQ and Si-SNR')
    ax.set_xticks(x, )
    ax.set_xticklabels(['Architecture'], fontsize=25)
    #ax.set_yticklabels(np.linspace(0, 30, 7), fontsize=18)
    ax.set_ylim([0, 4.5])

    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)

    ax.legend(loc='lower left', prop={'size': 25})
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=25)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    plt.show()
    plt.savefig('Data/Plots_Thesis/BarPlot.pdf')

    alignment_mask = [22.3]
    sub_mask = [24.4]
    inverse_mask = [26.58]

    labels = ['Si-SNR']

    x = np.arange(1, 2, 1)   # the label locations
    width = 1  # the width of the bars

    fig, ax = plt.subplots(figsize=(7,10))
    rects1 = ax.bar(0, alignment_mask, width, label='Alignment Mask', color='b')
    rects2 = ax.bar(1, sub_mask, width, label='Subtractive Mask', color='r')
    rects3 = ax.bar(2, inverse_mask, width, label='Inverse Mask', color='g')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Si-SNR', fontsize=25)
    # ax.set_title('Scores by PESQ and Si-SNR')
    ax.set_xticks(x, )
    ax.set_xticklabels(['Architecture'], fontsize=25)
    ax.set_yticklabels(np.linspace(0, 30, 7), fontsize=25)
    ax.legend(loc='lower left', prop={'size': 25})
    ax.set_ylim([0, 30])

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=25)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    plt.show()
    plt.savefig('Data/Plots_Thesis/BarPlot.pdf')

    x_time = [256, 512, 1024, 2048]
    filters_PESQ = [3.22, 3.46, 3.7, 3.92]
    filters_snr = [19.2, 21.4, 26.58, 28.3]

    plt.legend(['Si-SNR', 'PESQ'], prop={'size': 18})

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of input samples', fontsize=18)
    ax1.set_ylabel('Si-SNR', color=color, fontsize=18)
    # ax1.set_xticks(fontsize=18)
    # ax1.set_yticks(fontsize=18)
    ax1.plot(x_time, filters_snr, color='b', label='Si-SNR', linewidth=2, marker='s')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 30])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('PESQ Score', color=color, fontsize=18)  # we already handled the x-label with ax1
    ax2.plot(x_time, filters_PESQ, color='r', label='PESQ', linewidth=2, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 4.5])

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(18)

    fig.tight_layout()
    plt.grid(True)
    plt.savefig('Data/Plots_Thesis/FiltersizedPrese.pdf')
    plt.show()

    x_time = [128, 256, 512]
    bottleneck_PESQ = [3.7, 3.8, 4.1]
    bottleneck_snr = [26.58, 25.4, 30.3]
    network_size = [3.7, 7.944, 17.394]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of filters in bottleneck layer', fontsize=18)
    ax1.set_ylabel('Si-SNR', color=color, fontsize=18)
    # ax1.set_xticks(fontsize=18)
    # ax1.set_yticks(fontsize=18)
    ax1.plot(x_time, bottleneck_snr, color='r', label='Si-SNR', linewidth=2, marker='s')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 32])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('PESQ Score', color=color, fontsize=18)  # we already handled the x-label with ax1
    ax2.plot(x_time, bottleneck_PESQ, color='b', label='PESQ', linewidth=2, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 4.5])
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax3.set_ylabel('Network Size [mio.]', color=color, fontsize=18)  # we already handled the x-label with ax1
    ax3.plot(x_time, network_size, color='g', label='Network Size [mio.]', linewidth=2, marker='s')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_ylim([0, 20])
    ax3.spines['right'].set_position(('data', 0.2))

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(18)

    for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
                 ax3.get_xticklabels() + ax3.get_yticklabels()):
        item.set_fontsize(18)

    fig.tight_layout()
    plt.grid(True)
    plt.savefig('Data/Plots_Thesis/FiltersizedPrese.pdf')
    plt.show()

    x_time = [2, 4, 8, 16]
    speakers_PESQ = [3.7, 3.4, 2.9, 2.5]
    speakers_snr = [26.58, 21.9, 13.7, 11.6]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of interfering speakers', fontsize=18)
    ax1.set_ylabel('Si-SNR', color=color, fontsize=18)
    # ax1.set_xticks(fontsize=18)
    # ax1.set_yticks(fontsize=18)
    ax1.plot(x_time, speakers_snr, color='b', label='Si-SNR', linewidth=2, marker='s')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 30])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('PESQ Score', color=color, fontsize=18)  # we already handled the x-label with ax1
    ax2.plot(x_time, speakers_PESQ, color='r', label='PESQ', linewidth=2, marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 4.5])

    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(18)

    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
                 ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(18)

    fig.tight_layout()
    plt.grid(True)
    plt.savefig('Data/Plots_Thesis/FiltersizedPrese.pdf')
    plt.show()

    return 1



if __name__ == '__main__':

    ground_path = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamples/'

    # Temp Block
    path_1TempBlock = ground_path + '1TempBlock/'
    path_3TempBlock = ground_path + '3TempBlock/'

    # Input Samples
    path_256Samples = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamples256Samples/'
    path_512Samples = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamples512Samples/'
    path_2048Samples = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamples2048Samples/'
    #path_512InputSamples = ground_path + '512InputSamples/'

    path_enhance = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamplesEnhancement'

    # Number of Filters
    path_128Filter = ground_path + '128Filter/'
    path_256Filter = ground_path + '256Filter/'

    # Loss Function
    path_MixedLoss = ground_path + 'MixedLoss/'
    path_SiSNR = ground_path + 'SiSNR/'

    # Bottleneck
    path_512 = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamplesBottleneck512/'
    path_256 = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamplesBottleneck256/'
    path_128 = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamplesBottleneck128/'


    # Architectures
    path_ResNet = os.path.join(ground_path, 'ResNet')
    path_SubMask = os.path.join(ground_path, 'SubMask')
    path_RealInverse = 'C:/Users/silas/PycharmProjects/VoiceIsolation/Data/EvalSamplesTest/'
    path_Inverse = os.path.join(ground_path, 'InverseBlock')

    # Interfering Speakers
    path_4Speakers = os.path.join(ground_path, '4Speakers')
    path_8Speakers = os.path.join(ground_path, '8Speakers')
    path_16Speakers = os.path.join(ground_path, '16Speakers')

    path_mos = 'C:/Users/silas/PycharmProjects/VoiceIsolation/mushra.csv'
    #pesq_mean, pesq_var, stoi_mean, stoi_varm, sisnr_mean, sisnr_var = calculatePESQandStoi(path_128)

    #mutual_information = calculate_MI(path_RealInverse)

    #time_series_network(path_RealInverse)

    #get_spectrogram(path_RealInverse)
    #plot_windowing_function()

    #plot_mos(path_mos)
    #plot_activationFunctions()
    #plot_preprocessing()
    #plot_database_distribution()
    #tests()

    plot_for_presi()

    test = 1