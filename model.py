import numpy as np
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Reshape, Concatenate, AveragePooling2D, GlobalAvgPool2D
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout, LayerNormalization
from tensorflow.keras.activations import swish, relu
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf


def model_fn(args):
    main_conf = np.asarray([8, 16]) * args['layers']
    activation = {'swish': swish, 'relu': relu}[args['activation']]
    rglzr = l1_l2(l1=0., l2=0.0002)
    normalization = LayerNormalization
    # -------------------------------================= Image data =================----------------------------------- #

    base_model = {'xept': xception.Xception,
                  'incept': inception_v3.InceptionV3,
                  'effnet0': efficientnet.EfficientNetB0,
                  'effnet1': efficientnet.EfficientNetB1}[args['model']](include_top=False, input_shape=args['input_shape'])
    base_model.trainable = False
    image_input = Input(shape=args['input_shape'], name='image')
    base_model = base_model(image_input, training=False)
    # -----------------================= Inception module C used in Inception v4 =================-------------------- #
    conv1x1ap = AveragePooling2D(padding='same', strides=1)(base_model)
    conv1x1ap = Conv2D(main_conf[1], activation=activation, kernel_size=1, padding='same')(conv1x1ap)
    conv1x1ap = normalization()(conv1x1ap)
    conv1x1ap = Dropout(rate=args['dropout'])(conv1x1ap)
    conv1x1 = Conv2D(main_conf[1], activation=activation, kernel_size=1, padding='same')(base_model)
    conv1x1 = normalization()(conv1x1)
    conv1x1 = Dropout(rate=args['dropout'])(conv1x1)
    conv1x1_1x3_3x1 = Conv2D(main_conf[0], activation=activation, kernel_size=1, padding='same')(base_model)
    conv1x1_1x3 = Conv2D(main_conf[1], activation=activation, kernel_size=(1, 3), padding='same')(conv1x1_1x3_3x1)
    conv1x1_1x3 = normalization()(conv1x1_1x3)
    conv1x1_1x3 = Dropout(rate=args['dropout'])(conv1x1_1x3)
    conv1x1_3x1 = Conv2D(main_conf[1], activation=activation, kernel_size=(3, 1), padding='same')(conv1x1_1x3_3x1)
    conv1x1_3x1 = normalization()(conv1x1_3x1)
    conv1x1_3x1 = Dropout(rate=args['dropout'])(conv1x1_3x1)
    conv1x1_2 = Conv2D(main_conf[0], activation=activation, kernel_size=1, padding='same')(base_model)
    conv1x3_3x1 = Conv2D(main_conf[1], activation=activation, kernel_size=(1, 3), padding='same')(conv1x1_2)
    conv3x1_3x1_1x3 = Conv2D(main_conf[1], activation=activation, kernel_size=(3, 1), padding='same')(conv1x3_3x1)
    conv1x1_2_1x3 = Conv2D(main_conf[1], activation=activation, kernel_size=(1, 3), padding='same')(conv3x1_3x1_1x3)
    conv1x1_2_1x3 = normalization()(conv1x1_2_1x3)
    conv1x1_2_1x3 = Dropout(rate=args['dropout'])(conv1x1_2_1x3)
    conv1x1_2_3x1 = Conv2D(main_conf[1], activation=activation, kernel_size=(3, 1), padding='same')(conv3x1_3x1_1x3)
    conv1x1_2_3x1 = normalization()(conv1x1_2_3x1)
    conv1x1_2_3x1 = Dropout(rate=args['dropout'])(conv1x1_2_3x1)
    inc_mod = Concatenate()([conv1x1ap, conv1x1, conv1x1_1x3, conv1x1_3x1, conv1x1_2_1x3, conv1x1_2_3x1])
    common_layers = GlobalAvgPool2D()(inc_mod)
    # --------------------------------================ Tabular data =================--------------------------------- #
    if not args['only_image']:
        image_type_input = Input(shape=(2,), name='image_type', dtype=tf.float32)
        sex_input = Input(shape=(2,), name='sex', dtype=tf.float32)
        anatom_site_input = Input(shape=(6,), name='anatom_site_general', dtype=tf.float32)
        age_input = Input(shape=(10,), name='age_approx', dtype=tf.float32)

        image_type = Reshape(target_shape=(2, 1))(image_type_input)
        sex = Reshape(target_shape=(2, 1))(sex_input)
        anatom_site = Reshape(target_shape=(6, 1))(anatom_site_input)
        age = Reshape(target_shape=(10, 1))(age_input)

        image_type = LSTM(4, return_sequences=True)(image_type)
        image_type = normalization()(image_type)

        sex = LSTM(4, return_sequences=True)(sex)
        sex = normalization()(sex)

        anatom_site = LSTM(4, return_sequences=True)(anatom_site)
        anatom_site = normalization()(anatom_site)

        age = LSTM(4, return_sequences=True)(age)
        age = normalization()(age)

        concat_inputs = Concatenate(-2)([image_type, sex, anatom_site, age])
        lstm = LSTM(16, dropout=args['dropout'])(concat_inputs)
        lstm = normalization()(lstm)
        # -------------------------------================== Concat part ==================---------------------------------#
        common_layers = Concatenate(axis=1)([common_layers, lstm])
    common_layers = Dense(32, activation=activation, kernel_regularizer=rglzr)(common_layers)
    common_layers = normalization()(common_layers)
    common_layers = Dropout(args['dropout'])(common_layers)
    output_layer = Dense(args['num_classes'], activation='softmax', kernel_regularizer=rglzr, name='class')(common_layers)
    if args['only_image']:
        return Model([image_input], [output_layer])
    else:
        return Model([image_input, image_type_input, sex_input, anatom_site_input, age_input], [output_layer])
