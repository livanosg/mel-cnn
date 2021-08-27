import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from tensorflow.keras.layers import Concatenate, AveragePooling2D, GlobalAvgPool2D
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout, LayerNormalization
from tensorflow.keras.activations import swish, relu
from tensorflow.keras.regularizers import l1_l2


def model_fn(args):
    main_conf = np.asarray([8, 16]) * args['layers']
    activation = {'swish': swish, 'relu': relu}[args['activation']]
    rglzr = l1_l2(l1=0., l2=0.0001)
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
    inc_avrg = AveragePooling2D(padding='same', strides=1)(base_model)  # Inception module C used in Inception v4
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
    if not args['only_image']:
        shape = (20, 1)
        if args['no_image_type']:
            shape = (18, 1)
        clinical_data_input = Input(shape=shape, name='clinical_data', dtype=tf.float32)
        lstm_1 = LSTM(128, return_sequences=True)(clinical_data_input)
        lstm_1 = normalization()(lstm_1)
        inputs_list.append(clinical_data_input)
        lstm_2 = LSTM(64)(lstm_1)
        lstm_2 = normalization()(lstm_2)
        common = Concatenate(axis=-1)([common, lstm_2])
        # -------------------------------================== Concat part ==================---------------------------------#
    common = Dense(32, activation=activation, kernel_regularizer=rglzr)(common)
    common = normalization()(common)
    common = Dense(32, activation=activation, kernel_regularizer=rglzr)(common)
    output = Dense(args['num_classes'], activation='softmax', kernel_regularizer=rglzr, name='class')(common)
    return Model(inputs_list, [output])
