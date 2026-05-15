from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate, Dense, Reshape, Add, \
    GlobalAveragePooling2D, Multiply, Activation, BatchNormalization, \
    Lambda
from keras.models import Model
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from Data import *
from Evaluation import net_evaluation


def conv_block(x, filters, name_prefix="conv"):
    x = Conv2D(filters, 3, activation="relu", padding="same",
               kernel_initializer="he_normal", name=f"{name_prefix}_1")(x)
    x = Conv2D(filters, 3, activation="relu", padding="same",
               kernel_initializer="he_normal", name=f"{name_prefix}_2")(x)
    return x


def channel_attention(x, reduction=16, name="ch_att"):
    ch = x.shape[-1]
    se = GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = Dense(ch // reduction, activation="relu", name=f"{name}_fc1")(se)
    se = Dense(ch, activation="sigmoid", name=f"{name}_fc2")(se)
    se = Reshape((1, 1, ch))(se)
    return Multiply(name=f"{name}_scale")([x, se])


def spatial_attention(x, name="sp_att"):
    avg_pool = Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool = Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)
    concat = concatenate([avg_pool, max_pool], axis=-1)
    sa = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid", name=f"{name}_conv")(concat)
    return Multiply(name=f"{name}_sp_scale")([x, sa])


def dilated_residual_attention_block(x, filters, dilation_rates=(1, 2, 4), dropout=0.0, name_prefix="dra"):
    shortcut = x

    y = x
    for i, d in enumerate(dilation_rates):
        y = Conv2D(filters, 3, dilation_rate=d, padding="same",
                   kernel_initializer="he_normal", name=f"{name_prefix}_dconv{i + 1}")(y)
        y = BatchNormalization(name=f"{name_prefix}_bn{i + 1}")(y)
        y = Activation("relu", name=f"{name_prefix}_act{i + 1}")(y)

    if int(shortcut.shape[-1]) != filters:
        shortcut = Conv2D(filters, 1, padding="same", name=f"{name_prefix}_proj")(shortcut)

    y = Add(name=f"{name_prefix}_res_add")([shortcut, y])
    if dropout > 0.0:
        y = Dropout(dropout, name=f"{name_prefix}_drop")(y)

    # Attention
    y = channel_attention(y, reduction=16, name=f"{name_prefix}_ch")
    y = spatial_attention(y, name=f"{name_prefix}_sp")
    return y


def nested_unet_dra(sol, input_size=(256, 256, 1), num_classes=1, deep_supervision=False,
                    use_dra_in_encoder=True, use_dra_in_decoder=True, dra_filters_multiplier=1):
    inputs = Input(input_size, name="input")
    base_filters = sol[0]
    # Encoder
    x0_0 = dilated_residual_attention_block(inputs, base_filters * dra_filters_multiplier, name_prefix="x0_0_dra") \
        if use_dra_in_encoder else conv_block(inputs, base_filters, name_prefix="x0_0")
    p0_0 = MaxPooling2D(pool_size=(2, 2))(x0_0)

    x1_0 = dilated_residual_attention_block(p0_0, base_filters * 2 * dra_filters_multiplier, name_prefix="x1_0_dra") \
        if use_dra_in_encoder else conv_block(p0_0, base_filters * 2, name_prefix="x1_0")
    p1_0 = MaxPooling2D(pool_size=(2, 2))(x1_0)

    x2_0 = dilated_residual_attention_block(p1_0, base_filters * 4 * dra_filters_multiplier, name_prefix="x2_0_dra") \
        if use_dra_in_encoder else conv_block(p1_0, base_filters * 4, name_prefix="x2_0")
    p2_0 = MaxPooling2D(pool_size=(2, 2))(x2_0)

    x3_0 = dilated_residual_attention_block(p2_0, base_filters * 8 * dra_filters_multiplier, name_prefix="x3_0_dra") \
        if use_dra_in_encoder else conv_block(p2_0, base_filters * 8, name_prefix="x3_0")
    p3_0 = MaxPooling2D(pool_size=(2, 2))(x3_0)

    # Bottleneck
    x4_0 = dilated_residual_attention_block(p3_0, base_filters * 16 * dra_filters_multiplier,
                                            name_prefix="x4_0_dra") if use_dra_in_encoder \
        else conv_block(p3_0, base_filters * 16, name_prefix="x4_0")
    x4_0 = Dropout(0.5)(x4_0)

    # Decoder (nested UNet++)
    up3_1 = UpSampling2D(size=(2, 2))(x4_0)
    x3_1 = (dilated_residual_attention_block(concatenate([x3_0, up3_1], axis=3),
                                             base_filters * 8 * dra_filters_multiplier, name_prefix="x3_1_dra")
            if use_dra_in_decoder else conv_block(concatenate([x3_0, up3_1], axis=3),
                                                  base_filters * 8, name_prefix="x3_1"))

    up2_1 = UpSampling2D(size=(2, 2))(x3_1)
    x2_1 = (dilated_residual_attention_block(concatenate([x2_0, up2_1], axis=3),
                                             base_filters * 4 * dra_filters_multiplier, name_prefix="x2_1_dra")
            if use_dra_in_decoder else conv_block(concatenate([x2_0, up2_1], axis=3),
                                                  base_filters * 4, name_prefix="x2_1"))

    up2_2 = UpSampling2D(size=(2, 2))(x3_1)
    x2_2 = (dilated_residual_attention_block(concatenate([x2_0, x2_1, up2_2], axis=3),
                                             base_filters * 4 * dra_filters_multiplier, name_prefix="x2_2_dra")
            if use_dra_in_decoder else conv_block(concatenate([x2_0, x2_1, up2_2], axis=3),
                                                  base_filters * 4, name_prefix="x2_2"))

    up1_1 = UpSampling2D(size=(2, 2))(x2_1)
    x1_1 = (dilated_residual_attention_block(concatenate([x1_0, up1_1], axis=3),
                                             base_filters * 2 * dra_filters_multiplier, name_prefix="x1_1_dra")
            if use_dra_in_decoder else conv_block(concatenate([x1_0, up1_1], axis=3),
                                                  base_filters * 2, name_prefix="x1_1"))

    up1_2 = UpSampling2D(size=(2, 2))(x2_2)
    x1_2 = (dilated_residual_attention_block(concatenate([x1_0, x1_1, up1_2], axis=3),
                                             base_filters * 2 * dra_filters_multiplier, name_prefix="x1_2_dra")
            if use_dra_in_decoder else conv_block(concatenate([x1_0, x1_1, up1_2], axis=3),
                                                  base_filters * 2, name_prefix="x1_2"))

    up1_3 = UpSampling2D(size=(2, 2))(x2_2)
    x1_3 = (dilated_residual_attention_block(concatenate([x1_0, x1_1, x1_2, up1_3], axis=3),
                                             base_filters * 2 * dra_filters_multiplier, name_prefix="x1_3_dra")
            if use_dra_in_decoder else conv_block(concatenate([x1_0, x1_1, x1_2, up1_3], axis=3),
                                                  base_filters * 2, name_prefix="x1_3"))

    up0_1 = UpSampling2D(size=(2, 2))(x1_1)
    x0_1 = (dilated_residual_attention_block(concatenate([x0_0, up0_1], axis=3),
                                             base_filters * dra_filters_multiplier, name_prefix="x0_1_dra")
            if use_dra_in_decoder else conv_block(concatenate([x0_0, up0_1], axis=3),
                                                  base_filters, name_prefix="x0_1"))

    up0_2 = UpSampling2D(size=(2, 2))(x1_2)
    x0_2 = (dilated_residual_attention_block(concatenate([x0_0, x0_1, up0_2], axis=3),
                                             base_filters * dra_filters_multiplier, name_prefix="x0_2_dra")
            if use_dra_in_decoder else conv_block(concatenate([x0_0, x0_1, up0_2], axis=3),
                                                  base_filters, name_prefix="x0_2"))

    up0_3 = UpSampling2D(size=(2, 2))(x1_3)
    x0_3 = (dilated_residual_attention_block(concatenate([x0_0, x0_1, x0_2, up0_3], axis=3),
                                             base_filters * dra_filters_multiplier, name_prefix="x0_3_dra")
            if use_dra_in_decoder else conv_block(concatenate([x0_0, x0_1, x0_2, up0_3], axis=3),
                                                  base_filters, name_prefix="x0_3"))

    up0_4 = UpSampling2D(size=(2, 2))(x1_3)
    x0_4 = (dilated_residual_attention_block(concatenate([x0_0, x0_1, x0_2, x0_3, up0_4], axis=3),
                                             base_filters * dra_filters_multiplier, name_prefix="x0_4_dra")
            if use_dra_in_decoder else conv_block(concatenate([x0_0, x0_1, x0_2, x0_3, up0_4], axis=3),
                                                  base_filters, name_prefix="x0_4"))

    # Output
    activation = "sigmoid" if num_classes == 1 else "softmax"
    if deep_supervision:
        outs = [Conv2D(num_classes, 1, activation=activation, name=f"out_{i}")(o)
                for i, o in enumerate([x0_1, x0_2, x0_3, x0_4], 1)]
        model = Model(inputs=inputs, outputs=outs, name="HybridDRA_UNetPP")
    else:
        out = Conv2D(num_classes, 1, activation=activation, name="final_out")(x0_4)
        model = Model(inputs=inputs, outputs=out, name="HybridDRA_UNetPP")

    model.compile(optimizer=Adam(learning_rate=sol[1]),
                  loss="binary_crossentropy" if num_classes == 1 else "categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def Model_UNetplusplus(Unet_Path, Image_Path, Mask_Path, Predict_Path, sol=None, model_name='UNETPP_DRA.h5', input_size=(512, 512, 3),
                 **model_kwargs):
    if sol is None:
        sol = [5, 0.01, 100]
    model = nested_unet_dra(sol, input_size=input_size, **model_kwargs)

    ckpt_path = os.path.join(Unet_Path, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = trainGenerator(4, Unet_Path, Image_Path, Mask_Path, data_gen_args, save_to_dir=None)
    model_checkpoint = ModelCheckpoint('unet_membrane_4000.hdf5', monitor='loss', verbose=1, save_best_only=True)
    image_list = os.listdir(Unet_Path + Image_Path)
    image_count = len(image_list)
    Res = []
    if sol:
        model.fit_generator(myGene, hidden_neuron=round(sol[2]), steps_per_epoch=round(sol[1]),
                            epochs=100, callbacks=[model_checkpoint])
    else:
        model.fit_generator(myGene, steps_per_epoch=500, epochs=1, callbacks=[model_checkpoint])

    testGene = testGenerator(Unet_Path + Image_Path + "/", num_image=image_count)
    results = model.predict_generator(testGene, image_count, verbose=1)
    model.save(model_name)
    if not os.path.exists(Unet_Path + Predict_Path):
        os.mkdir(Unet_Path + Predict_Path)
    Images = saveResult(Unet_Path + Predict_Path + "/", results)
    Images.append(Images)
    Res.append(results)
    Eval = net_evaluation(Images, Res)
    return Eval, Res