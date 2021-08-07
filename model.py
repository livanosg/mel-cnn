import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, LayerNormalization, Dropout, GlobalAvgPool2D, Dense, Concatenate, LSTM,\
    Reshape, AveragePooling2D, ReLU
from tensorflow.keras.activations import swish
from tensorflow.keras.regularizers import l1_l2
from tensorflow import dtypes


def model_fn(args):
    main_conf = np.asarray([8, 16])
    layers = {1: main_conf,
              2: main_conf * 2,
              3: main_conf * 4}

    activ = ReLU(negative_slope=args["relu_grad"])
    rglzr = l1_l2(l1=0., l2=0.0002)
    normalization = LayerNormalization
    # -------------------------------================= Image data =================----------------------------------- #
    base_model = args['model'](include_top=False, input_shape=args['input_shape'])
    base_model.trainable = False
    image_input = Input(shape=args['input_shape'], name='image')
    base_model = base_model(image_input, training=False)
    # -----------------================= Inception module C used in Inception v4 =================-------------------- #
    conv1x1ap = AveragePooling2D(padding='same', strides=1)(base_model)
    conv1x1ap = Conv2D(layers[args['layers']][1], activation=activ, kernel_size=1, padding='same')(conv1x1ap)
    conv1x1ap = normalization()(conv1x1ap)
    conv1x1ap = Dropout(rate=args['dropout_ratio'])(conv1x1ap)
    conv1x1 = Conv2D(layers[args['layers']][1], activation=activ, kernel_size=1, padding='same')(base_model)
    conv1x1 = normalization()(conv1x1)
    conv1x1 = Dropout(rate=args['dropout_ratio'])(conv1x1)
    conv1x1_1x3_3x1 = Conv2D(layers[args['layers']][0], activation=activ, kernel_size=1, padding='same')(base_model)
    conv1x1_1x3 = Conv2D(layers[args['layers']][1], activation=activ, kernel_size=(1, 3), padding='same')(conv1x1_1x3_3x1)
    conv1x1_1x3 = normalization()(conv1x1_1x3)
    conv1x1_1x3 = Dropout(rate=args['dropout_ratio'])(conv1x1_1x3)
    conv1x1_3x1 = Conv2D(layers[args['layers']][1], activation=activ, kernel_size=(3, 1), padding='same')(conv1x1_1x3_3x1)
    conv1x1_3x1 = normalization()(conv1x1_3x1)
    conv1x1_3x1 = Dropout(rate=args['dropout_ratio'])(conv1x1_3x1)
    conv1x1_2 = Conv2D(layers[args['layers']][0], activation=activ, kernel_size=1, padding='same')(base_model)
    conv1x3_3x1 = Conv2D(layers[args['layers']][1], activation=activ, kernel_size=(1, 3), padding='same')(conv1x1_2)
    conv3x1_3x1_1x3 = Conv2D(layers[args['layers']][1], activation=activ, kernel_size=(3, 1), padding='same')(conv1x3_3x1)
    conv1x1_2_1x3 = Conv2D(layers[args['layers']][1], activation=activ, kernel_size=(1, 3), padding='same')(conv3x1_3x1_1x3)
    conv1x1_2_1x3 = normalization()(conv1x1_2_1x3)
    conv1x1_2_1x3 = Dropout(rate=args['dropout_ratio'])(conv1x1_2_1x3)
    conv1x1_2_3x1 = Conv2D(layers[args['layers']][1], activation=activ, kernel_size=(3, 1), padding='same')(conv3x1_3x1_1x3)
    conv1x1_2_3x1 = normalization()(conv1x1_2_3x1)
    conv1x1_2_3x1 = Dropout(rate=args['dropout_ratio'])(conv1x1_2_3x1)
    inc_mod = Concatenate()([conv1x1ap, conv1x1, conv1x1_1x3, conv1x1_3x1, conv1x1_2_1x3, conv1x1_2_3x1])
    custom_avg_pool = GlobalAvgPool2D()(inc_mod)
    # --------------------------------================ Tabular data =================--------------------------------- #
    image_type_input = Input(shape=(2,), name='image_type', dtype=dtypes.float32)
    sex_input = Input(shape=(2,), name='sex', dtype=dtypes.float32)
    anatom_site_input = Input(shape=(6,), name='anatom_site_general', dtype=dtypes.float32)
    age_input = Input(shape=(10,), name='age_approx', dtype=dtypes.float32)
    concat_inputs = Concatenate()([image_type_input, sex_input, anatom_site_input, age_input])
    concat_inputs = Reshape(target_shape=(20, 1))(concat_inputs)
    custom_lstm = LSTM(32, activation=activ)(concat_inputs)
    custom_fc2_layers = normalization(epsilon=1e-6)(custom_lstm)
    custom_fc2_layers = Dropout(rate=args['dropout_ratio'])(custom_fc2_layers)
    custom_fc2_layers2 = Dense(16, activation=activ, kernel_regularizer=rglzr)(custom_fc2_layers)
    custom_fc2_layers2 = Dropout(rate=args['dropout_ratio'])(custom_fc2_layers2)
    # -------------------------------================== Concat part ==================---------------------------------#
    common_layers = Concatenate(axis=1)([custom_avg_pool, custom_fc2_layers2])
    common_layers = Dense(16, activation=activ, kernel_regularizer=rglzr)(common_layers)
    common_layers = normalization(epsilon=1e-6)(common_layers)
    output_layer = Dense(args['num_classes'], activation='softmax', kernel_regularizer=rglzr, name='class')(common_layers)
    return Model([image_input, image_type_input, sex_input, anatom_site_input, age_input], [output_layer])
