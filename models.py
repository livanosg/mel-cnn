import numpy as np
import tensorflow as tf


def model_struct(args):
    dense_nodes = np.asarray([1, 2]) * args['dense_layers']
    incept_nodes = np.asarray([1, 2]) * args['conv_layers']
    merge_nodes = np.asarray([1, 2]) * args['merge_layers']

    activation = {'swish': tf.keras.activations.swish,
                  'relu': tf.keras.activations.relu}[args['activation']]
    rglzr = tf.keras.regularizers.l1_l2(l1=0., l2=1e-6)
    seed = None
    inputs_list = []
    initializer = tf.keras.initializers.HeNormal()
    # -------------------------------================= Image data =================----------------------------------- #

    base_model = {'xept': tf.keras.applications.xception.Xception,
                  'incept': tf.keras.applications.inception_v3.InceptionV3,
                  'effnet0': tf.keras.applications.efficientnet.EfficientNetB0,
                  'effnet1': tf.keras.applications.efficientnet.EfficientNetB1,
                  'effnet6': tf.keras.applications.efficientnet.EfficientNetB6}[args['pretrained']](include_top=False, input_shape=args['input_shape'])
    base_model.trainable = False
    image_input = tf.keras.layers.Input(shape=args['input_shape'], name='image')
    inputs_list.append(image_input)

    base_model = base_model(image_input, training=False)
    inc_avrg = tf.keras.layers.AveragePooling2D(padding='same', strides=1)(base_model)  # Inception module C used in Inception v4
    inc_avrg = tf.keras.layers.Conv2D(incept_nodes[1], activation=activation, kernel_size=1, padding='same', kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(inc_avrg)
    inc_avrg = tf.keras.layers.LayerNormalization()(inc_avrg)
    inc_avrg = tf.keras.layers.Dropout(rate=args['dropout'], seed=seed)(inc_avrg)
    inc_conv1 = tf.keras.layers.Conv2D(incept_nodes[1], activation=activation, kernel_size=1, padding='same', kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(base_model)
    inc_conv1 = tf.keras.layers.LayerNormalization()(inc_conv1)
    inc_conv1 = tf.keras.layers.Dropout(rate=args['dropout'], seed=seed)(inc_conv1)
    inc_conv2 = tf.keras.layers.Conv2D(incept_nodes[0], activation=activation, kernel_size=1, padding='same', kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(base_model)
    inc_conv2 = tf.keras.layers.Dropout(rate=args['dropout'], seed=seed)(inc_conv2)
    inc_conv2_1 = tf.keras.layers.Conv2D(incept_nodes[1], activation=activation, kernel_size=(1, 3), padding='same', kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(inc_conv2)
    inc_conv2_1 = tf.keras.layers.LayerNormalization()(inc_conv2_1)
    inc_conv2_1 = tf.keras.layers.Dropout(rate=args['dropout'], seed=seed)(inc_conv2_1)
    inc_conv2_2 = tf.keras.layers.Conv2D(incept_nodes[1], activation=activation, kernel_size=(3, 1), padding='same', kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(inc_conv2)
    inc_conv2_2 = tf.keras.layers.LayerNormalization()(inc_conv2_2)
    inc_conv2_2 = tf.keras.layers.Dropout(rate=args['dropout'], seed=seed)(inc_conv2_2)
    inc_mod = tf.keras.layers.Concatenate()([inc_avrg, inc_conv1, inc_conv2_1, inc_conv2_2, base_model])
    common = tf.keras.layers.GlobalAvgPool2D()(inc_mod)

# --------------------------------================ Tabular data =================--------------------------------- #
    if not args['no_clinical_data']:
        shape = (20,)
        if args['no_image_type']:
            shape = (18,)
        clinical_data_input = tf.keras.layers.Input(shape=shape, name='clinical_data', dtype=tf.float32)
        inputs_list.append(clinical_data_input)
        clinical_data_1 = tf.keras.layers.Dense(dense_nodes[1], activation=activation, kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(clinical_data_input)
        clinical_data_1 = tf.keras.layers.LayerNormalization()(clinical_data_1)
        clinical_data_1 = tf.keras.layers.Dropout(rate=args['dropout'], seed=seed)(clinical_data_1)
        clinical_data_2 = tf.keras.layers.Dense(dense_nodes[0], activation=activation, kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(clinical_data_1)
        clinical_data_2 = tf.keras.layers.LayerNormalization()(clinical_data_2)
        clinical_data_2 = tf.keras.layers.Dropout(rate=args['dropout'], seed=seed)(clinical_data_2)
        clinical_data_con = tf.keras.layers.Concatenate(axis=-1)([clinical_data_2, clinical_data_1])
        clinical_data_3 = tf.keras.layers.Dense(dense_nodes[0], activation=activation, kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(clinical_data_con)
        clinical_data_3 = tf.keras.layers.LayerNormalization()(clinical_data_3)
        clinical_data_3 = tf.keras.layers.Dropout(rate=args['dropout'], seed=seed)(clinical_data_3)
        common = tf.keras.layers.Concatenate(axis=-1)([common, clinical_data_1, clinical_data_2, clinical_data_3])
    # -------------------------------================== Concat part ==================---------------------------------#
    common = tf.keras.layers.Dense(merge_nodes[1], activation=activation, kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(common)
    # common = tf.keras.layers.LayerNormalization()(common)
    # common = tf.keras.layers.Dropout(rate=args['dropout'], seed=seed)(common)
    common = tf.keras.layers.Dense(merge_nodes[0], activation=activation, kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer)(common)
    # common = tf.keras.layers.LayerNormalization()(common)
    # common = tf.keras.layers.Dense(16, activation=activation, kernel_regularizer=rglzr)(common)
    output = tf.keras.layers.Dense(args['num_classes'], activation='softmax', kernel_regularizer=rglzr, kernel_initializer=initializer, bias_initializer=initializer, name='class')(common)
    return tf.keras.Model(inputs_list, [output])
