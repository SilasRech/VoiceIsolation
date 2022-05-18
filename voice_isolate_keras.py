import tensorflow as tf


def encoder(input, N, L):
    """

    :param input:  input shape (1,512)
    :param N: Number of Filters in Encoder
    :param L: Size of Kernel in Convolution
    :return: Output Shape (1, N, K) with K = 2N/L - 1
    """
    #kernel_size = 32ms
    #stride = 16ms

    x = tf.keras.layers.Conv1D(N, kernel_size=L, strides=L//2, data_format='channels_first', name='EncoderWeights', padding='same')(input)
    # data_format='channels_first'
    #outputs = tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, padding="same")(x)
    #'

    return tf.keras.Model(input, x, name='Encoder')


def decoder(input, N, L):

    x = tf.keras.layers.Conv1DTranspose(1, kernel_size=L, strides=L//2, name='DecoderWeights', data_format='channels_first', padding='same')(input)
    # N, 1, kernel_size = L, stride = L // 2
    #x = tf.keras.layers.Permute((2, 1))(x)
    return tf.keras.Model(input, x, name='Decoder')


def targetedConvTasNet(input_vecs):

    L = 64 # Length of the filters (in samples)
    N = 512 # Number of Filters in Autoencoder

    B = 256 # Number of Channels in the bottleneck and the resiudal 1x1conv blocks
    Sc = 256 #Number of Channels in Skip Connection Path 1x1 conv
    H = 512 # Number of Channels in Convolutional Blocks
    P = 3 # Kernel Size in Conv Block

    #input_mix = input_vecs
    input_mix, input_target = input_vecs

    K = int((2 * input_target[1] / L))

    # Necessary Shapes
    mixture_in = tf.keras.Input(shape=input_mix, dtype="float32")
    target_in = tf.keras.Input(shape=input_target, dtype="float32")

    temp_in = tf.keras.Input(shape=(B, K), dtype="float32")
    dec_in = tf.keras.Input(shape=(N, K), dtype="float32")

    # Encoder to higher to lower dimension
    encoder_net = encoder(mixture_in, N, L)
    decoder_net = decoder(dec_in, N, L)

    # FIRST BLOCK
    temporal_net_far = temporal_block(temp_in, H, P, Sc, B, 'Block1', K)
    temporal_net_near1 = temporal_block(temp_in, H, P, Sc, B, 'Block2', K)
    #temporal_net_near2 = temporal_block(dec_in, H, P, Sc, B, 'Block3', K)
    #temporal_net_near3 = temporal_block(temp_in, H, P, Sc, B, 'Block4', K)

    mixture_enc = encoder_net(mixture_in)
    far_enc = encoder_net(target_in)

    # Pre Feed for TasNet
    mixture_w = tf.keras.layers.LayerNormalization()(mixture_enc)
    mixture_w = tf.keras.layers.Conv1D(B, 1, data_format='channels_first', use_bias=False)(mixture_w)

    far_end = tf.keras.layers.LayerNormalization()(far_enc)
    far_end = tf.keras.layers.Conv1D(B, 1, data_format='channels_first', use_bias=False)(far_end)

    # Build FarEnd Mask and Invert
    far_end_mask = temporal_net_far(far_end)
    far_end_mask1 = tf.keras.layers.Conv1D(N, 1, strides=1, data_format='channels_first', activation='sigmoid', name='FarMask', use_bias=False)(far_end_mask)

    far_end_mask2 = tf.subtract(tf.ones((N, K)), far_end_mask1)

    inputs_farend = tf.math.multiply(mixture_enc, far_end_mask2)
    inputs_farend = tf.keras.layers.Conv1D(B, 1, data_format='channels_first', use_bias=False)(inputs_farend)
    near_end_mask = temporal_net_near1(inputs_farend)
    #near_end_mask1 = temporal_net_near2(near_end_mask)

    #sums = tf.keras.layers.Add()([near_end_mask, near_end_mask1])
    #sums_all = tf.keras.layers.PReLU(shared_axes=[1, 2])(sums)

   # near_end_mask3 = tf.keras.layers.Conv1D(N, 1, strides=1, data_format='channels_first', activation='sigmoid', name='NearMask', use_bias=False)(near_end_mask1)

    #near_end_mask3 = tf.math.multiply(mixture_enc, near_end_mask3)
    near_end_mask = tf.keras.layers.Conv1D(N, 1, strides=1, data_format='channels_first', activation='sigmoid',
                                          use_bias=False)(near_end_mask)
    dec_est_source = decoder_net(near_end_mask)

    model = tf.keras.Model(inputs=[mixture_in, target_in], outputs=[dec_est_source], name='TargetConvNet')
    return model


def res_block_first(input, dilation_factor, P, B):

    residual = input
    x = conv_block(input, dilation_factor=dilation_factor, P=P, B=B)
    x = conv_block(x, dilation_factor=dilation_factor, P=P, B=B)

    x = tf.keras.layers.Add()([x, residual])
    outputs = tf.keras.layers.Activation('relu')(x)

    return outputs


def res_block_second(input, dilation_factor, P, B):

    residual = input
    x = conv_block(input, dilation_factor=dilation_factor, P=P, B=B)
    x = conv_block(x, dilation_factor=dilation_factor, P=P, B=B)

    x = tf.keras.layers.Add()([x, residual])
    outputs = tf.keras.layers.Activation('relu')(x)

    return outputs


def conv_block(input, dilation_factor, B=512, P=3):

    x = tf.keras.layers.SeparableConv1D(B, P, padding="same", dilation_rate=dilation_factor, data_format='channels_first')(input)
    #x = tf.keras.layers.Activation(tf.nn.leaky_relu(alpha=0.3))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    #outputs = tf.keras.layers.BatchNormalization()(x)
    return x


def temporal_block(input, H, P=3, Sc=128, B=128, name='Net', K=16):

    skip1, x = conv_block_tas(input, 1, H, P, Sc=Sc, B=B, K=K)
    skip2, x = conv_block_tas(x, 2, H, P, Sc=Sc, B=B, K=K)
    skip3, x = conv_block_tas(x, 4, H, P, Sc=Sc, B=B, K=K)
    skip4, x = conv_block_tas(x, 8, H, P, Sc=Sc, B=B, K=K)
    skip5, x = conv_block_tas(x, 16, H, P, Sc=Sc, B=B, K=K)
    skip6, x = conv_block_tas(x, 32, H, P, Sc=Sc, B=B, K=K)
    skip7, x = conv_block_tas(x, 64, H, P, Sc=Sc, B=B, K=K)
    skip8, x = conv_block_tas(x, 128, H, P, Sc=Sc, B=B, K=K)
    #skip9, x = conv_block_tas(x, 256, H, P, Sc=Sc, B=B, L=L)
    #skip10, x = conv_block_tas(x, 512, H, P, Sc=Sc, B=B, L=L)

    sums = tf.keras.layers.Add()([x, skip1, skip2, skip3, skip4, skip5, skip6, skip7, skip8])
    sums_all = tf.keras.layers.PReLU(shared_axes=[1, 2])(sums)

    return tf.keras.Model(input, sums_all, name=name)


def conv_block_tas(input, dilation_factor, H, P=3, Sc=128, B=128, K=32):

    x = tf.keras.layers.Conv1D(B, 1, padding="same", dilation_rate=dilation_factor, data_format='channels_first', use_bias=False)(input)

    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.SeparableConv1D(H, P, padding="same", dilation_rate=dilation_factor, data_format='channels_first', use_bias=False)(x)

    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    x = tf.keras.layers.LayerNormalization()(x)

    skip = tf.keras.layers.Conv1D(Sc, 1, padding="same", dilation_rate=dilation_factor, data_format='channels_first', use_bias=False)(x)
    bottleneck = tf.keras.layers.Conv1D(B, 1, padding="same", dilation_rate=dilation_factor, data_format='channels_first', use_bias=False)(x)

    output = tf.keras.layers.Add()([input, bottleneck])

    return skip, output



