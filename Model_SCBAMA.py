from keras import Model
from keras.applications import EfficientNetB2
import numpy as np
import tensorflow as tf
from keras.src.optimizers.adam import Adam
from Evaluation import evaluation
from tensorflow.keras.layers import Input
from keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from Model_MobileNetV3 import Model_MobileNetV3
from Model_Twin_Trans import Model_Twin_Trans


def Model_EfficientNet(Train_Data, Train_Target):
    IMG_SIZE = 32

    Train_x = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_x[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = EfficientNetB2(
        weights='imagenet',
        include_top=False,
        pooling='max',
        input_tensor=inputs
    )
    outputs = Dense(units=Train_Target.shape[1], activation='sigmoid')(base_model.output)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Train_x, Train_Target, epochs=1)
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([inp], [out]) for out in outputs]
    layerNo = -1
    Feats = []
    for i in range(Train_x.shape[0]):
        test = Train_x[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)
    return np.asarray(Feats)


def transformer_fusion_block_3(F1, F2, F3, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1, training=False):
    """
    Transformer Fusion Block (TFB) for 3 features:
      F1 -> Local features (EfficientNet-B2)
      F2 -> Global features (Transformer/Swin)
      F3 -> Attention-guided features (MobileNetV3)
    """

    # Step 1: F1 attends to F2
    attn12 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(F1, F2, F2)
    attn12 = Dropout(dropout)(attn12, training=training)
    out12 = LayerNormalization(epsilon=1e-6)(Add()([F1, attn12]))

    # Step 2: out12 attends to F3
    attn13 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(out12, F3, F3)
    attn13 = Dropout(dropout)(attn13, training=training)
    out13 = LayerNormalization(epsilon=1e-6)(Add()([out12, attn13]))

    # Step 3: Feed-forward network
    ffn = Dense(ff_dim, activation="relu")(out13)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(dropout)(ffn, training=training)
    fused = LayerNormalization(epsilon=1e-6)(Add()([out13, ffn]))

    return fused


def Model_SCBAMA(Image, Target, Epoch=None, sol=None):
    if Epoch is None:
        Epoch = 100
    if sol is None:
        sol = [5, 0.01, 100]
    Feat_1 = Model_EfficientNet(Image, Target)
    Feat_2 = Model_Twin_Trans(Image, Target)
    Feat_3 = Model_MobileNetV3(Image, Target)

    learnperc = round(Image.shape[0] * 0.75)
    F1_train, F1_test = Feat_1[:learnperc, :], Feat_1[learnperc:, :]
    F2_train, F2_test = Feat_2[:learnperc, :], Feat_2[learnperc:, :]
    F3_train, F3_test = Feat_3[:learnperc, :], Feat_3[learnperc:, :]
    y_train, y_test = Target[:learnperc, :], Target[learnperc:, :]

    F1 = F1_train.reshape((F1_train.shape[0], 1, F1_train.shape[1]))
    F2 = F2_train.reshape((F2_train.shape[0], 1, F2_train.shape[1]))
    F3 = F3_train.reshape((F3_train.shape[0], 1, F3_train.shape[1]))

    F1_test = F1_test.reshape((F1_test.shape[0], 1, F1_test.shape[1]))
    F2_test = F2_test.reshape((F2_test.shape[0], 1, F2_test.shape[1]))
    F3_test = F3_test.reshape((F3_test.shape[0], 1, F3_test.shape[1]))

    inp1 = Input(shape=(1, F1.shape[-1]))
    inp2 = Input(shape=(1, F2.shape[-1]))
    inp3 = Input(shape=(1, F3.shape[-1]))
    # Transformer Fusion Block
    fused = transformer_fusion_block_3(inp1, inp2, inp3, embed_dim=1, num_heads=3)
    # Multi-head Self-Attention on fused representation
    attn_out = MultiHeadAttention(num_heads=3, key_dim=128, name="self_attention")(fused, fused)
    attn_out = LayerNormalization(epsilon=1e-6, name="attn_norm")(fused + attn_out)
    # Flatten sequence dimension
    x = tf.squeeze(attn_out, axis=1)
    # Fully connected + Dropout + Output
    x = Dense(sol[0], activation="relu", name="fc1")(x)
    x = Dropout(0.3, name="dropout1")(x)
    output = Dense(y_train.shape[1], activation="softmax", name="output")(x)
    model = Model(inputs=[inp1, inp2, inp3], outputs=output, name="MultiPath_TFB_Model")
    model.compile(
        optimizer=Adam(sol[1]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    model.fit([F1, F2, F2], y_train, epochs=Epoch, steps_per_epoch=sol[2], batch_size=4)
    pred = model.predict([F1_test, F2_test, F3_test])
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(y_test, pred)
    return Eval, pred


