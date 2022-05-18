import os
import soundfile as sf
import voice_isolate_keras as vi
import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
import json
import glob
import random as rd
import datetime
import time

from database_helper import data_helper
from Data.DatabaseBuilder import reconstruct_speech

from tools import logger
from keras_data_generator import load_training_batch
from keras_data_generator import audiofiles_without_impulse
import matplotlib.pyplot as plt
from functools import partial


def sisnr_loss(clean, mix):

    EPS = 1e-8
    clean_zero_mean = clean - tf.reduce_mean(clean, axis=-1, keepdims=True)

    mix_zero_mean = mix - tf.reduce_mean(mix, axis=-1, keepdims=True)

    pair_wise_dot = tf.reduce_sum(clean_zero_mean * mix_zero_mean, axis=-1, keepdims=True)

    s_target_energy = tf.reduce_sum(clean_zero_mean ** 2, axis=-1, keepdims=True) + EPS
    pair_wise_proj = pair_wise_dot * clean_zero_mean / s_target_energy

    e_noise = mix_zero_mean - pair_wise_proj

    pair_wise_sdr = tf.reduce_sum(pair_wise_proj ** 2, axis=-1) / (tf.reduce_sum(e_noise ** 2, axis=-1) + EPS)
    si_snr = 10 * log10(pair_wise_sdr + EPS)

    si_snr = -1.0 * si_snr

    return si_snr


def mixed_loss(clean, mix):

    weight = 0.99
    si_loss = sisnr_loss(clean, mix)
    mse_loss = tf.keras.losses.MeanSquaredError()
    mse_loss = mse_loss(clean, mix)

    return weight * mse_loss + (1-weight) * si_loss


def log10(x):
    x1 = tf.math.log(x)
    x2 = tf.math.log(10.0)
    return x1/x2


def plotting_function(model, nearend, farend):

    # Name for Layers in neural network
    # FarMask, NearMask, Encoder, EncoderWeights

    time_steps = np.linspace(0, 500, 64)

    func_farmask = K.function([model.get_layer('input_1').input, model.get_layer('input_2').input], model.get_layer('NearMask').output)

    func_encoder = K.function([model.get_layer('input_1').input, model.get_layer('input_2').input],
                              model.get_layer('Encoder').output)

    encoder_weights = np.squeeze(model.get_layer('Encoder').weights[0])
    decoder_weights = np.squeeze(model.get_layer('Decoder').weights[0])

    #func_nearmask = K.function([model.get_layer('input_1').input, model.get_layer('input_2').input],
    #                          model.get_layer('NearMask').output)

    farend_mask = func_farmask([nearend[0], farend[0]])

    func_encoder = func_encoder([nearend[0], farend[0]])
    #func_nearmask = func_nearmask([nearend[0], farend[0]])

    farend_mask_plot = farend_mask[1]
    encoder_plot = func_encoder[1]

    #im4 = plt.imshow(encoder_weights.T, cmap="seismic", aspect=4)
    #plt.show()
    #im3 = plt.imshow(encoder_weights.T, cmap="seismic", aspect=4)
    #plt.show()
    im2 = plt.imshow(encoder_plot, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.show()

    im1 = plt.imshow(farend_mask_plot, cmap="Blues")
    plt.colorbar()
    plt.show()

    return 1


def read_tfrecord(example):

    tfrecord_format = {
            "clean": tf.io.FixedLenFeature([1024], tf.float32),
            "near": tf.io.FixedLenFeature([1024], tf.float32),
            "far": tf.io.FixedLenFeature([1024], tf.float32),
        }

    example = tf.io.parse_single_example(example, tfrecord_format)
    clean = window_audio(example["clean"])
    near = window_audio(example["near"])
    far = window_audio(example["far"])

    return [near, far], clean


def window_audio(audio):

    frame_length = 1024
    #audio = tf.signal.frame(audio, frame_length, frame_length)
    audio = tf.expand_dims(audio, axis=0)
    return audio


def load_dataset(filenames):

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(
        partial(read_tfrecord), num_parallel_calls=tf.data.AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(100)
    return dataset


if __name__ == '__main__':

    tf.keras.backend.clear_session()

    print(tf.config.list_physical_devices('GPU'))

    # META Parameter for Home or Triton
    home = True
    training = True
    load_random_speaker = False
    oneFile = False
    continue_training = True

    sampling_rate = 16000

    win_length = 2048
    hop_length = 2048

    parameters = {
        'speaker_distr': {'Speaker1': 'M', 'Speaker2': 'M'},
    }

    if home:
        path = os.path.join('C:\\', 'Users', 'silas', 'Documents', 'Python Scripts', 'LibriSpeech')

        path_train = 'C:/Users/silas/NoisyOverlappingSpeakers/Database'
        path_eval = 'C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase'
        path_eval4speakers = 'C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase_4Speakers'
        path_eval8speakers = 'C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase_8Speakers'
        path_eval16speakers = 'C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase_16Speakers'
        path_evalOnlyFarend = 'C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase_OnlyFarend'
        path_evalOnlyNearend = 'C:/Users/silas/NoisyOverlappingSpeakers/EvalDatabase_OnlyNearend'
    else:
        path = os.getcwd()
        print(path)
        parent_path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.join(parent_path, 'LibriSpeech')

        path_train = os.path.join(parent_path, 'Database')
        path_eval = os.path.join(parent_path, 'EvalDatabase')

    if load_random_speaker:
        dataset = data_helper.LibriData(parameters['speaker_distr'], path)
        audio_speaker1, audio_speaker2, fs = dataset.load_random_speaker_files()
    else:
        audio_speaker1, audio_speaker2, fs = data_helper.loader()

    #far_end_test, complete_scenario_test, clean_test, clean_farend = audiofiles_without_impulse(audio_speaker1,
    #                                                                                            audio_speaker2, win_length)

    far_end_test, complete_scenario_test, clean_test, clean_farend = load_training_batch(path_eval, [0], 2048, 2048, sampling_rate,  mode='train')

    params_network = {'dim': (1, win_length), 'batch_size': 1, 'shuffle': False, 'M': 2, 'N': 512, 'L': int(16000 * 0.032),
                      'T': 512, 'B': 128}

    # num samples
    if oneFile:
        samples = 1
    else:
        samples = 120

    if training:
        print('Found {} samples in the dataset, loading batch for training'.format(samples))
        # Reference: Target = FarEnd, SpeechMix = NearEnd, Clean = Clean

        # model
        model = vi.targetedConvTasNet(input_vecs=[(1, win_length), (1, win_length)])

        # model = tf.keras.models.load_model('MODEL', custom_objects={'sisnr_Loss': sisnr_Loss})

        # optimizer and learning rate
        learning_rate = 0.001
        optimizer_name = 'Adam'
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Optimizer
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                      patience=3, min_lr=0.001)

        def scheduler(epoch, lr):
            if epoch < 50:
                return lr
            else:
                return lr * tf.math.exp(-0.01)

        scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # compile model
        name_loss = 'mse'
        num_epochs = 1

        losses = 'mse'
        #losses = {"Near": 'MSE', "Far": 'MSE'}

        lossWeights = 1
        #lossWeights = {"Near": 1.0, "Far": 0.5}

        model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer, metrics=sisnr_loss)
        # model.compile(loss='MSE', optimizer='adam', metrics=sisnr_Loss, run_eagerly=True)

        model.summary()

        # convTasNET.summary()

        print('Starting Training')

        average_loss = 0
        sum_loss = []
        sum_used_samples = []

        if oneFile:
            epochs = 1
        else:
            epochs = 200

        total_number_samples = int(len(glob.glob(path_train + "/*.wav")) / 4)-1
        print('Total Number of Training Samples {}'.format(total_number_samples))

        if continue_training:
            finished_epochs = 1
        else:
            finished_epochs = 0

        # Number of Epochs
        for m in range(epochs-finished_epochs):

            print('---------------------------------')
            print('Epoch: {} of {}'.format(m, epochs-finished_epochs))
            print('---------------------------------')

            sample_list = np.linspace(0, total_number_samples, total_number_samples-1, dtype='int').tolist()
            rd.shuffle(sample_list)
            shuffled_list = np.array_split(np.asarray(sample_list), samples)

            start = time.time()
            # Number of
            for n in range(samples):
                print('---------------------------------')
                print('Batch: {} of {}'.format(n, samples))
                print('---------------------------------')
                if oneFile:
                    target_speaker = far_end_test
                    speech_mix = complete_scenario_test
                    clean_speaker = clean_test
                    clean_farend = clean_farend

                    epochs_per_file = 200
                    starting_index = 0
                else:
                    epochs_per_file = 1
                    if continue_training:

                        #with open("LoggingFile.json", "r") as read_file:
                        #    data = json.load(read_file)

                        #indeces = data['UsedSamples']
                        #starting_index = indeces[-1]

                        model = tf.keras.models.load_model(
                            'ModelForPlotting_2_InverseMasking',
                            custom_objects={'sisnr_loss': sisnr_loss})
                    else:
                        starting_index = 0

                    start_data = time.time()

                    target_speaker, speech_mix, clean_speaker, clean_farend = load_training_batch(path_train, shuffled_list[n], win_length, hop_length, sampling_rate=sampling_rate, mode='train')
                    epochs = 2
                    end_data = time.time()
                    print('Loading time for training data: {}s'.format(end_data - start_data))

                num_samples_in_batch = int(np.round(total_number_samples // samples))

                history = model.fit(
                    #train_dataset,
                    #x=[target_speaker[i], speech_mix[i]], y={'NearEnd': clean_speaker[i]},
                    x=[speech_mix, target_speaker],
                    y=clean_speaker,
                    # train_ds,
                    epochs=epochs_per_file,
                    # validation_data=[[target_speaker_eval[1], speech_mix_eval[1]], clean_speaker_eval[1]],
                    batch_size=125,
                    verbose=2,
                    # callbacks=[earlystopping_cb, mdlcheckpoint_cb]
                    callbacks=[reduce_lr, scheduler]
                )

                # Save Model
                model.save('ModelForPlotting_2_InverseMasking')
            end = time.time()
            print('Training Time per Epoch: {}'.format(end - start))
    else:

        total_number_samples = int(len(glob.glob(path_eval + "/*.wav")) / 4)
        sample_list = np.linspace(0, total_number_samples-1, total_number_samples-1, dtype='int').tolist()
        rd.shuffle(sample_list)

        shuffled_list = np.array_split(np.asarray(sample_list), samples)

        if oneFile:
            far_end_test, complete_scenario_test, clean_test, clean_farend = audiofiles_without_impulse(audio_speaker1, audio_speaker2, win_length)
        else:
            far_end_test, complete_scenario_test, clean_test, clean_farend = load_training_batch(path_eval, sample_list, win_length, hop_length, sampling_rate=sampling_rate, mode='eval')

        print('Testing the Network --- loading model')
        model = tf.keras.models.load_model('ModelForPlotting_2_InverseMasking',
                                           custom_objects={'sisnr_loss': sisnr_loss})

        model.summary()

    if oneFile:
        far_end_test = [far_end_test]
        complete_scenario_test = [complete_scenario_test]
        clean_test = [clean_test]
        clean_farend = [clean_farend]

    print('Testing - Masking Audio')

    plot_out = plotting_function(model, complete_scenario_test, far_end_test)
    # Predict Model
    # output = model.predict(complete_scenario_test)
    for g in range(len(far_end_test)):
        print('Samples: {} of {}'.format(g, len(far_end_test)))
        output = model.predict([complete_scenario_test[g], far_end_test[g]])

        reconstruct_signal = reconstruct_speech(output, "HALF", win_length, hop_length)
        reconstruct_signal = reconstruct_signal/max(abs(reconstruct_signal))

        reconstruct_clean = reconstruct_speech(clean_test[g], "HALF", win_length, hop_length)
        reconstruct_clean = reconstruct_clean / max(abs(reconstruct_clean))

        reconstruct_complete = reconstruct_speech(complete_scenario_test[g], "HALF", win_length, hop_length)
        reconstruct_complete = reconstruct_complete / max(abs(reconstruct_complete))

        reconstruct_farend = reconstruct_speech(far_end_test[g], "HALF", win_length, hop_length)
        reconstruct_farend = reconstruct_farend / max(abs(reconstruct_farend))

        # LowEndFilter Signal

        #filtered_reconstruct_signal = butter_lowpass_filter(reconstruct_signal, 8000)

        # Write Signal
        sf.write('Data/EvalSamples/TestSeparation{}.wav'.format(g),  reconstruct_signal, sampling_rate)
        sf.write('Data/EvalSamples/TestSeparation_clean{}.wav'.format(g), reconstruct_clean, sampling_rate)
        sf.write('Data/EvalSamples/TestSeparation_CompleteScenario{}.wav'.format(g), reconstruct_complete, sampling_rate)
        sf.write('Data/EvalSamples/TestSeparation_FarEnd{}.wav'.format(g), reconstruct_farend, sampling_rate)

    if training:
        logger(params_network, learning_rate, optimizer_name, name_loss, num_epochs, samples, sum_loss)

    print('Finished')

test = 1

