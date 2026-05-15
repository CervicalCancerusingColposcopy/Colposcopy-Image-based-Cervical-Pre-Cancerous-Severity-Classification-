import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from keras_unet_collection.models import unet_3plus_2d
from Data import trainGenerator, testGenerator, saveResult
from Evaluation import net_evaluation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def build_unet3plus(input_size=(512, 512, 3),
                    n_labels=1,
                    filter_num_down=None,
                    stack_num_down=2,
                    stack_num_up=2,
                    activation='ReLU',
                    output_activation='Sigmoid',
                    batch_norm=True,
                    deep_supervision=False,
                    lr=1e-4):
    # Builds the UNet 3+ model using the keras-unet-collection package
    if filter_num_down is None:
        filter_num_down = [64, 128, 256, 512, 1024]
    model = unet_3plus_2d(input_size,
                          filter_num_down=filter_num_down,
                          n_labels=n_labels,
                          stack_num_down=stack_num_down,
                          stack_num_up=stack_num_up,
                          activation=activation,
                          output_activation=output_activation,
                          batch_norm=batch_norm,
                          deep_supervision=deep_supervision)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=BinaryCrossentropy() if n_labels == 1 else 'categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def Model_Unet3plus(Unet_Path, Image_Path, Mask_Path, Predict_Path,
                    model_name='UNet3Plus.h5', epochs=100, batch_size=8,
                    steps_per_epoch=200, input_size=(256, 256, 1), **model_kwargs):
    model = build_unet3plus(input_size=input_size, **model_kwargs)
    model.summary()

    ckpt = ModelCheckpoint(os.path.join(Unet_Path, model_name),
                           monitor='loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
    rl = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1)

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    train_gen = trainGenerator(batch_size, Unet_Path, Image_Path, Unet_Path, Mask_Path, data_gen_args,
                               save_to_dir=None)
    model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[ckpt, es, rl])

    test_gen = testGenerator(Unet_Path + Image_Path + "/", num_image=len(os.listdir(Unet_Path + Image_Path)))
    results = model.predict(test_gen, verbose=1)

    os.makedirs(Unet_Path + Predict_Path, exist_ok=True)
    Images = saveResult(Unet_Path + Predict_Path + "/", results)

    Eval = net_evaluation(Images, results)
    return Eval, results
