import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from Evaluation import evaluation


def Model(X, Y, test_x, test_y, Epoch):
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(20, 20, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(Y.shape[0], activation='relu'),
        Dropout(0.5),
        Dense(Y.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Summary of the model
    model.summary()

    # Train the model
    model.fit(X, Y, epochs=Epoch, batch_size=4)
    pred = model.predict(test_x)

    return pred


def Model_CNN(train_data, train_target, test_data, test_target, Epoch):
    IMG_SIZE = 20
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    pred = Model(Train_X, train_target, Test_X, test_target, Epoch)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(test_target, pred)

    return Eval, pred

