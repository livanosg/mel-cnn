import tensorflow as tf
from tensorflow.keras import applications
from tensorflow import keras
from tensorflow import dtypes
from tensorflow.keras import regularizers


def model_fn(model, input_shape, dropout_rate, alpha, classes):
    init = tf.keras.initializers.GlorotNormal()
    models = {'xept': (applications.xception.Xception, applications.xception.preprocess_input),
              'incept': (applications.inception_v3.InceptionV3, applications.inception_v3.preprocess_input),
              'effnet0': (applications.efficientnet.EfficientNetB0, applications.efficientnet.preprocess_input),
              'effnet1': (applications.efficientnet.EfficientNetB1, applications.efficientnet.preprocess_input)}

    model_preproc = keras.Sequential([tf.keras.layers.Lambda(function=models[model][1]), ], name="model_preproc")
    image_input = keras.Input(shape=input_shape, name='image')
    preprocessed_input = model_preproc(image_input)
    base_model = models[model][0](include_top=False, input_shape=input_shape)
    base_model.trainable = False
    base_model = base_model(preprocessed_input, training=False)
    custom_conv_layers = keras.layers.Conv2D(128, kernel_size=3, padding='same', kernel_initializer=init, kernel_regularizer=regularizers.l2(0.0001))(base_model)
    custom_conv_layers = keras.layers.BatchNormalization()(custom_conv_layers)
    custom_conv_layers = keras.layers.Dropout(rate=dropout_rate)(custom_conv_layers)
    custom_conv_layers = keras.layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer=init, kernel_regularizer=regularizers.l2(0.0001))(custom_conv_layers)
    custom_conv_layers = keras.layers.BatchNormalization()(custom_conv_layers)
    custom_conv_layers = keras.layers.Dropout(rate=dropout_rate)(custom_conv_layers)
    custom_fc_layers = keras.layers.Flatten()(custom_conv_layers)
    custom_fc_layers = keras.layers.Dense(128, activation=keras.layers.LeakyReLU(alpha=alpha), kernel_initializer=init, kernel_regularizer=regularizers.l2(0.0001))(custom_fc_layers)
    custom_fc_layers = keras.layers.BatchNormalization()(custom_fc_layers)
    custom_fc_layers = keras.layers.Dropout(rate=dropout_rate)(custom_fc_layers)
    custom_fc_layers = keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=alpha), kernel_initializer=init, kernel_regularizer=regularizers.l2(0.0001))(custom_fc_layers)
    custom_fc_layers = keras.layers.BatchNormalization()(custom_fc_layers)
    custom_fc_layers = keras.layers.Dropout(rate=dropout_rate)(custom_fc_layers)

    # -----------------------------================ Values part =================--------------------------------- #
    image_type_input = keras.Input(shape=(2,), name='image_type', dtype=dtypes.float32)
    sex_input = keras.Input(shape=(2,), name='sex', dtype=dtypes.float32)
    anatom_site_input = keras.Input(shape=(6,), name='anatom_site_general', dtype=dtypes.float32)
    age_input = keras.Input(shape=(10,), name='age_approx', dtype=dtypes.float32)
    concat_inputs = keras.layers.Concatenate()([image_type_input, sex_input, anatom_site_input, age_input])
    concat_inputs = keras.layers.Dropout(rate=dropout_rate)(concat_inputs)
    custom_fc2_layers = keras.layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=alpha), kernel_initializer=init, kernel_regularizer=regularizers.l2(0.0001))(concat_inputs)
    custom_fc2_layers = keras.layers.Dropout(rate=dropout_rate)(custom_fc2_layers)
    custom_fc2_layers = keras.layers.Dense(128, activation=keras.layers.LeakyReLU(alpha=alpha), kernel_initializer=init, kernel_regularizer=regularizers.l2(0.0001))(custom_fc2_layers)
    custom_fc2_layers = keras.layers.Dropout(rate=dropout_rate)(custom_fc2_layers)
    custom_fc2_layers = keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=alpha), kernel_initializer=init, kernel_regularizer=regularizers.l2(0.0001))(custom_fc2_layers)
    custom_fc2_layers = keras.layers.Dropout(rate=dropout_rate)(custom_fc2_layers)

    # -----------------------------================= Concat part =================---------------------------------#
    common_layers = keras.layers.Concatenate()([custom_fc2_layers, custom_fc_layers])
    common_layers = keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=alpha), kernel_initializer=init, kernel_regularizer=regularizers.l2(0.0001))(common_layers)
    common_layers = keras.layers.BatchNormalization()(common_layers)
    common_layers = keras.layers.Dropout(rate=dropout_rate)(common_layers)
    common_layers = keras.layers.Dense(32, activation=keras.layers.LeakyReLU(alpha=alpha), kernel_initializer=init, kernel_regularizer=regularizers.l2(0.0001))(common_layers)
    common_layers = keras.layers.BatchNormalization()(common_layers)
    output_layer = keras.layers.Dense(classes, activation='softmax', kernel_initializer=init, name='class')(common_layers)
    return keras.Model([image_input, image_type_input, sex_input, anatom_site_input, age_input], [output_layer])

