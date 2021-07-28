import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.applications import xception, inception_v3, efficientnet
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, Concatenate, Lambda
from tensorflow.keras.activations import swish
from tensorflow.keras.regularizers import l1_l2
from tensorflow import dtypes


def model_fn(model, input_shape, dropout_rate, alpha, classes):
    init = tf.keras.initializers.GlorotNormal()
    activ = swish# LeakyReLU(alpha=alpha)
    rglzr = l1_l2(l1=0., l2=0.0001)
    normalization = BatchNormalization
    models = {'xept': (xception.Xception, xception.preprocess_input),
              'incept': (inception_v3.InceptionV3, inception_v3.preprocess_input),
              'effnet0': (efficientnet.EfficientNetB0, efficientnet.preprocess_input),
              'effnet1': (efficientnet.EfficientNetB1, efficientnet.preprocess_input)}

    # -------------------------------================= Image data =================----------------------------------- #
    model_preproc = Sequential([Lambda(function=models[model][1])], name="model_preproc")
    base_model = models[model][0](include_top=False, input_shape=input_shape)
    base_model.trainable = False
    image_input = Input(shape=input_shape, name='image')
    preprocessed_input = model_preproc(image_input)
    base_model = base_model(preprocessed_input, training=False)
    custom_conv_layers = Conv2D(256, activation=activ, kernel_size=3, padding='same', kernel_initializer=init, activity_regularizer=rglzr)(base_model)
    custom_conv_layers = normalization()(custom_conv_layers)
    custom_conv_layers = Dropout(rate=dropout_rate)(custom_conv_layers)
    custom_conv_layers = Conv2D(256, activation=activ, kernel_size=3, padding='same', kernel_initializer=init, activity_regularizer=rglzr)(custom_conv_layers)
    custom_conv_layers = normalization()(custom_conv_layers)
    custom_conv_layers = Dropout(rate=dropout_rate)(custom_conv_layers)
    custom_fc_layers = Flatten()(custom_conv_layers)
    custom_fc_layers = Dense(128, activation=activ, kernel_initializer=init, activity_regularizer=rglzr)(custom_fc_layers)
    custom_fc_layers = normalization()(custom_fc_layers)
    custom_fc_layers = Dropout(rate=dropout_rate)(custom_fc_layers)
    custom_fc_layers = Dense(128, activation=activ, kernel_initializer=init, activity_regularizer=rglzr)(custom_fc_layers)
    custom_fc_layers = normalization()(custom_fc_layers)
    custom_fc_layers = Dropout(rate=dropout_rate)(custom_fc_layers)

    # --------------------------------================ Tabular data =================--------------------------------- #
    image_type_input = Input(shape=(2,), name='image_type', dtype=dtypes.float32)
    sex_input = Input(shape=(2,), name='sex', dtype=dtypes.float32)
    anatom_site_input = Input(shape=(6,), name='anatom_site_general', dtype=dtypes.float32)
    age_input = Input(shape=(10,), name='age_approx', dtype=dtypes.float32)
    concat_inputs = Concatenate()([image_type_input, sex_input, anatom_site_input, age_input])
    concat_inputs = Dropout(rate=dropout_rate)(concat_inputs)
    custom_fc2_layers = Dense(512, activation=activ, kernel_initializer=init, activity_regularizer=rglzr)(concat_inputs)
    custom_fc2_layers = normalization()(custom_fc2_layers)
    custom_fc2_layers = Dropout(rate=dropout_rate)(custom_fc2_layers)
    custom_fc2_layers = Dense(256, activation=activ, kernel_initializer=init, activity_regularizer=rglzr)(custom_fc2_layers)
    custom_fc2_layers = normalization()(custom_fc2_layers)
    custom_fc2_layers = Dropout(rate=dropout_rate)(custom_fc2_layers)
    custom_fc2_layers = Dense(128, activation=activ, kernel_initializer=init, activity_regularizer=rglzr)(custom_fc2_layers)
    custom_fc2_layers = normalization()(custom_fc2_layers)
    custom_fc2_layers = Dropout(rate=dropout_rate)(custom_fc2_layers)

    # -------------------------------================== Concat part ==================---------------------------------#
    common_layers = Concatenate()([custom_fc2_layers, custom_fc_layers])
    common_layers = Dense(128, activation=activ, kernel_initializer=init, activity_regularizer=rglzr)(common_layers)
    common_layers = normalization()(common_layers)
    common_layers = Dropout(rate=dropout_rate)(common_layers)
    common_layers = Dense(128, activation=activ, kernel_initializer=init, activity_regularizer=rglzr)(common_layers)
    common_layers = normalization()(common_layers)
    output_layer = Dense(classes, activation='softmax', kernel_initializer=init, activity_regularizer=rglzr, name='class')(common_layers)
    return Model([image_input, image_type_input, sex_input, anatom_site_input, age_input], [output_layer])
