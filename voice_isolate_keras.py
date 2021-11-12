import keras
import tensorflow as tf



def encoder(input_shape, output_shape):
    # TODO  Implementieren Sie hier Aufgabe 7.3



    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    # add batch normalization
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, activation='relu', name='dense_3'))
    # add dropout layer
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(output_shape, name='dense_5'))
    # change optimizer to 'Nadam'

    model.compile(optimizer='Nadam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()
    return model








def TargetedConvTasNet():




if __name__ == '__main__':


