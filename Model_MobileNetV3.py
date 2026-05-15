import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


def Model_MobileNetV3(Image, Target):
    IMG_SIZE = 32
    Train = np.zeros((Image.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Image.shape[0]):
        temp = np.resize(Image[i], (IMG_SIZE * IMG_SIZE, 3))
        Train[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    pretrained_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False
    inputs = pretrained_model.input
    x = Dense(256, activation='relu')(pretrained_model.output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(Target.shape[1], activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(Train, Target, epochs=1, steps_per_epoch=100)
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([inp], [out]) for out in outputs]
    layerNo = -1
    Feats = []
    for i in range(Train.shape[0]):
        test = Train[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)
    return np.asarray(Feats)
