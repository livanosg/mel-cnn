import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from tensorflow.keras.layers import Reshape, Concatenate, AveragePooling2D, GlobalAvgPool2D
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout, LayerNormalization
from tensorflow.keras.activations import swish, relu
from tensorflow.keras.regularizers import l1_l2


def model_fn(args):
    main_conf = np.asarray([8, 16]) * args['layers']
    activation = {'swish': swish, 'relu': relu}[args['activation']]
    rglzr = l1_l2(l1=0., l2=0.0003)
    normalization = LayerNormalization
    inputs_list = []
    # -------------------------------================= Image data =================----------------------------------- #

    base_model = {'xept': xception.Xception,
                  'incept': inception_v3.InceptionV3,
                  'effnet0': efficientnet.EfficientNetB0,
                  'effnet1': efficientnet.EfficientNetB1}[args['pretrained']](include_top=False, input_shape=args['input_shape'])
    base_model.trainable = False
    image_input = Input(shape=args['input_shape'], name='image')
    inputs_list.append(image_input)

    base_model = base_model(image_input, training=False)
    # -----------------================= Inception module C used in Inception v4 =================-------------------- #
    inc_avrg = AveragePooling2D(padding='same', strides=1)(base_model)
    inc_avrg = Conv2D(main_conf[1], activation=activation, kernel_size=1, padding='same')(inc_avrg)
    inc_avrg = normalization()(inc_avrg)
    inc_avrg = Dropout(rate=args['dropout'])(inc_avrg)
    inc_conv1 = Conv2D(main_conf[1], activation=activation, kernel_size=1, padding='same')(base_model)
    inc_conv1 = normalization()(inc_conv1)
    inc_conv1 = Dropout(rate=args['dropout'])(inc_conv1)
    inc_conv2 = Conv2D(main_conf[0], activation=activation, kernel_size=1, padding='same')(base_model)
    inc_conv2_1 = Conv2D(main_conf[1], activation=activation, kernel_size=(1, 3), padding='same')(inc_conv2)
    inc_conv2_1 = normalization()(inc_conv2_1)
    inc_conv2_1 = Dropout(rate=args['dropout'])(inc_conv2_1)
    inc_conv2_2 = Conv2D(main_conf[1], activation=activation, kernel_size=(3, 1), padding='same')(inc_conv2)
    inc_conv2_2 = normalization()(inc_conv2_2)
    inc_conv2_2 = Dropout(rate=args['dropout'])(inc_conv2_2)
    inc_conv3 = Conv2D(main_conf[0], activation=activation, kernel_size=1, padding='same')(base_model)
    inc_conv3 = Conv2D(main_conf[1], activation=activation, kernel_size=(1, 3), padding='same')(inc_conv3)
    inc_conv3 = Conv2D(main_conf[1], activation=activation, kernel_size=(3, 1), padding='same')(inc_conv3)
    inc_conv3_1 = Conv2D(main_conf[1], activation=activation, kernel_size=(1, 3), padding='same')(inc_conv3)
    inc_conv3_1 = normalization()(inc_conv3_1)
    inc_conv3_1 = Dropout(rate=args['dropout'])(inc_conv3_1)
    inc_conv3_2 = Conv2D(main_conf[1], activation=activation, kernel_size=(3, 1), padding='same')(inc_conv3)
    inc_conv3_2 = normalization()(inc_conv3_2)
    inc_conv3_2 = Dropout(rate=args['dropout'])(inc_conv3_2)
    inc_mod = Concatenate()([inc_avrg, inc_conv1, inc_conv2_1, inc_conv2_2, inc_conv3_1, inc_conv3_2])
    common = GlobalAvgPool2D()(inc_mod)
    # --------------------------------================ Tabular data =================--------------------------------- #
    concat_list = []
    if not args['only_image']:
        if not args['no_image_type']:
            image_type_input = Input(shape=(2,), name='image_type', dtype=tf.float32)
            image_type = Reshape(target_shape=(2, 1))(image_type_input)
            image_type = LSTM(4, activation=activation, return_sequences=True)(image_type)
            image_type = normalization()(image_type)

            inputs_list.append(image_type_input)
            concat_list.append(image_type)

        sex_input = Input(shape=(2,), name='sex', dtype=tf.float32)
        sex = Reshape(target_shape=(2, 1))(sex_input)
        sex = LSTM(4, activation=activation, return_sequences=True)(sex)
        sex = normalization()(sex)

        anatom_site_input = Input(shape=(6,), name='location', dtype=tf.float32)
        anatom_site = Reshape(target_shape=(6, 1))(anatom_site_input)
        anatom_site = LSTM(4, activation=activation, return_sequences=True)(anatom_site)
        anatom_site = normalization()(anatom_site)

        age_input = Input(shape=(10,), name='age_approx', dtype=tf.float32)
        age = Reshape(target_shape=(10, 1))(age_input)
        age = LSTM(4, activation=activation, return_sequences=True)(age)
        age = normalization()(age)

        inputs_list.append([sex_input, anatom_site_input, age_input])
        concat_list.append([sex, anatom_site, age])
        concat_inputs = Concatenate(-2)(concat_list)
        lstm = LSTM(16, activation=activation, dropout=args['dropout'])(concat_inputs)
        common_2 = normalization()(lstm)
        # -------------------------------================== Concat part ==================---------------------------------#
        common = Concatenate(axis=1)([common, common_2])
    common = Dense(32, activation=activation, kernel_regularizer=rglzr)(common)
    common = normalization()(common)
    common = Dropout(args['dropout'])(common)
    common = Dense(32, activation=activation, kernel_regularizer=rglzr)(common)
    common = normalization()(common)
    common = Dropout(args['dropout'])(common)
    output = Dense(args['num_classes'], activation='softmax', kernel_regularizer=rglzr, name='class')(common)
    return Model(inputs_list, [output])
