import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Concatenate, Flatten, Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.activations import swish, relu
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from features_def import TASK_CLASSES


def model_struct(args):
    conv_nodes = np.multiply([2, 3, 3.5, 4], args['conv_layers']).astype(np.int)
    dense_nodes = np.multiply([1, 2], args['dense_layers']).astype(np.int)
    merge_nodes = np.multiply([1, 0.5, 0.25], args['merge_layers']) .astype(np.int)
    act = {'swish': swish, 'relu': relu}[args['activation']]
    l1_l2 = tf.keras.regularizers.l1_l2(l1=args['l1_reg'], l2=args['l2_reg'])
    seed = None
    inputs_list = []
    init = tf.keras.initializers.HeNormal()
    input_shape = (args['image_size'], args['image_size'], 3)
    # -------------------------------================= Image data =================----------------------------------- #

    base_model = {'xept': xception.Xception,
                  'incept': inception_v3.InceptionV3,
                  'effnet0': efficientnet.EfficientNetB0,
                  'effnet1': efficientnet.EfficientNetB1,
                  'effnet6': efficientnet.EfficientNetB6}[args['pretrained']](include_top=False, input_shape=input_shape)
    base_model.trainable = False
    image_input = Input(shape=input_shape, name='image')
    inputs_list.append(image_input)

    base_model = base_model(image_input, training=False)
    # Inception module C used in Inception v4
    inc_avrg = AveragePooling2D(padding='same', strides=1)(base_model)
    inc_avrg = Conv2D(conv_nodes[0], padding='same', activation=act, kernel_size=1, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(inc_avrg)
    inc_avrg = LayerNormalization()(inc_avrg)
    inc_avrg = Dropout(rate=args['dropout'], seed=seed)(inc_avrg)

    inc_c1 = Conv2D(conv_nodes[0], padding='same', activation=act, kernel_size=1, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(base_model)
    inc_c1 = LayerNormalization()(inc_c1)
    inc_c1 = Dropout(rate=args['dropout'], seed=seed)(inc_c1)

    inc_c2 = Conv2D(conv_nodes[1], padding='same', activation=act, kernel_size=1, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(base_model)
    inc_c2 = LayerNormalization()(inc_c2)
    inc_c2 = Dropout(rate=args['dropout'], seed=seed)(inc_c2)

    inc_c2_1 = Conv2D(conv_nodes[0], padding='same', activation=act, kernel_size=(1, 3), kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(inc_c2)
    inc_c2_1 = LayerNormalization()(inc_c2_1)
    inc_c2_1 = Dropout(rate=args['dropout'], seed=seed)(inc_c2_1)

    inc_c2_2 = Conv2D(conv_nodes[0], padding='same', activation=act, kernel_size=(3, 1), kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(inc_c2)
    inc_c2_2 = LayerNormalization()(inc_c2_2)
    inc_c2_2 = Dropout(rate=args['dropout'], seed=seed)(inc_c2_2)

    inc_c3 = Conv2D(conv_nodes[1], padding='same', activation=act, kernel_size=1, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(base_model)
    inc_c3 = LayerNormalization()(inc_c3)
    inc_c3 = Dropout(rate=args['dropout'], seed=seed)(inc_c3)

    inc_c3 = Conv2D(conv_nodes[2], padding='same', activation=act, kernel_size=(1, 3), kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(inc_c3)
    inc_c3 = LayerNormalization()(inc_c3)
    inc_c3 = Dropout(rate=args['dropout'], seed=seed)(inc_c3)

    inc_c3 = Conv2D(conv_nodes[3], padding='same', activation=act, kernel_size=(3, 1), kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(inc_c3)
    inc_c3 = LayerNormalization()(inc_c3)
    inc_c3 = Dropout(rate=args['dropout'], seed=seed)(inc_c3)

    inc_c3_1 = Conv2D(conv_nodes[0], padding='same', activation=act, kernel_size=(1, 3), kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(inc_c3)
    inc_c3_1 = LayerNormalization()(inc_c3_1)
    inc_c3_1 = Dropout(rate=args['dropout'], seed=seed)(inc_c3_1)

    inc_c3_2 = Conv2D(conv_nodes[0], padding='same', activation=act, kernel_size=(3, 1), kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(inc_c3)
    inc_c3_2 = LayerNormalization()(inc_c3_2)
    inc_c3_2 = Dropout(rate=args['dropout'], seed=seed)(inc_c3_2)

    common = Concatenate()([base_model, inc_avrg, inc_c1, inc_c2_1, inc_c2_2, inc_c3_1, inc_c3_2])
    common = Flatten()(common)
# --------------------------------================ Tabular data =================--------------------------------- #
    if not args['no_clinical_data']:
        shape = (20,)
        if args['no_image_type']:
            shape = (18,)
        clinical_data_input = Input(shape=shape, name='clinical_data', dtype=tf.float32)
        inputs_list.append(clinical_data_input)
        clinical_data_1 = Dense(dense_nodes[1], activation=act, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(clinical_data_input)
        clinical_data_1 = LayerNormalization()(clinical_data_1)
        clinical_data_1 = Dropout(rate=args['dropout'], seed=seed)(clinical_data_1)
        clinical_data_2 = Dense(dense_nodes[0], activation=act, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(clinical_data_1)
        clinical_data_2 = LayerNormalization()(clinical_data_2)
        clinical_data_2 = Dropout(rate=args['dropout'], seed=seed)(clinical_data_2)
        clinical_data_con = Concatenate(axis=-1)([clinical_data_2, clinical_data_1])
        clinical_data_3 = Dense(dense_nodes[0], activation=act, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(clinical_data_con)
        clinical_data_3 = LayerNormalization()(clinical_data_3)
        clinical_data_3 = Dropout(rate=args['dropout'], seed=seed)(clinical_data_3)
        common = Concatenate(axis=-1)([common, clinical_data_1, clinical_data_2, clinical_data_3])
    # -------------------------------================== Concat part ==================---------------------------------#
    # common = GlobalAvgPool2D()(common)
    common = Dense(merge_nodes[0], activation=act, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(common)
    common = LayerNormalization()(common)
    # common = Dropout(rate=args['dropout'], seed=seed)(common)
    common = Dense(merge_nodes[1], activation=act, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(common)
    common = LayerNormalization()(common)
    # common = Dropout(rate=args['dropout'], seed=seed)(common)
    common = Dense(merge_nodes[2], activation=act, kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init)(common)
    # common = LayerNormalization()(common)
    # common = Dense(16, activation=act, kernel_regularizer=rglzr)(common)
    output = Dense(len(TASK_CLASSES[args['task']]), activation='softmax', kernel_regularizer=l1_l2, kernel_initializer=init, bias_initializer=init, name='class')(common)
    return tf.keras.Model(inputs_list, [output])
