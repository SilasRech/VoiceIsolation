import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from main_keras import sisnr_loss
import keras.backend as K
from database_helper import data_helper
from keras_data_generator import make_frames

audio_speaker1, audio_speaker2, fs = data_helper.loader()

window = 512
hop = 256
fs = 16000

nearend = audio_speaker1[:int(0.5*fs)]
farend = audio_speaker2[:int(0.5*fs)]

clean_nearend = (nearend - np.mean(nearend)) / np.max(np.abs(nearend))
clean_farend = (farend - np.mean(farend)) / np.max(np.abs(farend))

speech_mix_near = 0.8 * clean_nearend + 0.3 * clean_farend
speech_mix_near = (speech_mix_near - np.mean(speech_mix_near)) / np.max(np.abs(speech_mix_near))

speech_mix_far = 0.1 * clean_nearend + 0.8 * clean_farend
speech_mix_far = (speech_mix_far - np.mean(speech_mix_far)) / np.max(np.abs(speech_mix_far))

nearend = make_frames(speech_mix_near, window, hop, "HALF")
farend = make_frames(speech_mix_far, window, hop, "HALF")
clean_nearend = make_frames(clean_nearend, window, hop, "HALF")
clean_farend = make_frames(clean_farend, window, hop, "HALF")

farend = np.expand_dims(np.asarray(farend), axis=1)
nearend = np.expand_dims(np.asarray(nearend), axis=1)
clean_nearend = np.expand_dims(np.asarray(clean_nearend), axis=1)
clean_farend = np.expand_dims(np.asarray(clean_farend), axis=1)

# Name for Layers in neural network
# FarMask, NearMask, Encoder, EncoderWeights

time_steps = np.linspace(0, 500, 64)

load_from_model = False

# Model Prediction
if load_from_model:
    model = tf.keras.models.load_model('ModelForPlotting_2_NoSecondMask',
                                       custom_objects={'sisnr_loss': sisnr_loss})

    func_farmask = K.function([model.get_layer('input_1').input, model.get_layer('input_2').input], model.get_layer('FarMask').output)

    func_encoder = K.function([model.get_layer('input_1').input, model.get_layer('input_2').input],
                              model.get_layer('Encoder').output)

    encoder_weights = np.squeeze(model.get_layer('Encoder').weights[0])
    decoder_weights = np.squeeze(model.get_layer('Decoder').weights[0])

    np.save('EncoderWeights.npy', encoder_weights)
    np.save('DecoderWeights.npy', decoder_weights)

    #func_nearmask = K.function([model.get_layer('input_1').input, model.get_layer('input_2').input],
    #                          model.get_layer('NearMask').output)

    farend_mask = func_farmask([nearend, farend])
    func_encoder = func_encoder([nearend, farend])

    np.save('Mask', farend_mask)
    np.save('EncoderOutput', func_encoder)

else:
    encoder_weights = np.load('EncoderWeights.npy')
    decoder_weights = np.load('DecoderWeights.npy')
    farend_mask = np.load('Mask.npy')
    func_encoder = np.load('EncoderOutput.npy')

#func_nearmask = func_nearmask([nearend[0], farend[0]])

farend_mask_plot = np.sum(farend_mask, axis=2)

encoder_plot = np.sum(func_encoder, axis=2)

im2 = plt.imshow(encoder_plot.T, cmap="Blues", aspect='auto')
plt.show()

im4 = plt.imshow(encoder_weights.T, cmap="seismic", aspect=4)
plt.show()
im3 = plt.imshow(decoder_weights.T, cmap="seismic", aspect=4)
plt.show()

im1 = plt.imshow(farend_mask_plot, cmap="Blues")
plt.show()

test = 1