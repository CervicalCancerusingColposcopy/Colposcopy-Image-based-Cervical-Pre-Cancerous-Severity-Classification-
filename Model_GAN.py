import numpy as np
from keras import layers, models
from Evaluation import evaluation


def build_discriminator(inputs, num_classes):
    input_layer = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(input_layer)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    class_output = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=input_layer, outputs=class_output)
    return model


def Model_GAN(Train_Data, Train_Target, Test_Data, Test_Target, Epoch):
    IMG_SIZE = int(Train_Data.shape[1] / 2)
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    inputs = (Train_X.shape[1], Train_X.shape[2], Train_X.shape[3])
    classifier = build_discriminator(inputs, num_classes=Train_Target.shape[1])
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()
    classifier.fit(Train_X, Train_Target, epochs=Epoch, steps_per_epoch=100)
    pred = classifier.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)
    return Eval, pred

