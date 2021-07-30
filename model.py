import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, GlobalAvgPool2D, LeakyReLU, Dense, Concatenate, LayerNormalization
from tensorflow.keras.activations import swish
from tensorflow.keras.regularizers import l1_l2
from tensorflow import dtypes


def model_fn(args):
    init = tf.keras.initializers.lecun_normal()
    activ = LeakyReLU(alpha=args["relu_grad"])  # swish
    rglzr = None  # l1_l2(l1=0., l2=0.0001)
    normalization = LayerNormalization
    # -------------------------------================= Image data =================----------------------------------- #
    # model_preproc = Sequential([Lambda(function=models[model][1])], name="model_preproc")
    base_model = args['model'](include_top=False, input_shape=args['input_shape'])
    base_model.trainable = False
    image_input = Input(shape=args['input_shape'], name='image')
    base_model = base_model(image_input, training=False)
    custom_conv_layers = Conv2D(128, activation=activ, kernel_size=3, padding='same', kernel_initializer=init,
                                bias_initializer=init, kernel_regularizer=rglzr, bias_regularizer=rglzr)(base_model)
    custom_conv_layers = normalization(gamma_initializer=init, beta_initializer=init  # moving_mean_initializer=init, moving_variance_initializer=init
                                       )(custom_conv_layers)
    custom_conv_layers = Dropout(rate=args['dropout'])(custom_conv_layers)
    custom_conv_layers = Conv2D(64, activation=activ, kernel_size=3, padding='same', kernel_initializer=init,
                                bias_initializer=init, kernel_regularizer=rglzr, bias_regularizer=rglzr)(custom_conv_layers)
    custom_conv_layers = normalization(gamma_initializer=init, beta_initializer=init  # moving_mean_initializer=init, moving_variance_initializer=init
                                       )(custom_conv_layers)
    custom_conv_layers = Dropout(rate=args['dropout'])(custom_conv_layers)
    custom_avg_pool = GlobalAvgPool2D()(custom_conv_layers)
    custom_fc_layers = Dense(32, activation=activ, kernel_initializer=init, bias_initializer=init,
                             kernel_regularizer=rglzr, bias_regularizer=rglzr)(custom_avg_pool)
    custom_fc_layers = normalization(gamma_initializer=init, beta_initializer=init  # moving_mean_initializer=init, moving_variance_initializer=init
                                     )(custom_fc_layers)
    # custom_fc_layers = Dropout(rate=args['dropout'])(custom_fc_layers)
    custom_fc_layers = Dense(16, activation=activ, kernel_initializer=init, bias_initializer=init,
                             kernel_regularizer=rglzr, bias_regularizer=rglzr)(custom_fc_layers)
    custom_fc_layers = normalization(gamma_initializer=init, beta_initializer=init  # moving_mean_initializer=init,moving_variance_initializer=init
                                     )(custom_fc_layers)
    # custom_fc_layers = Dropout(rate=args['dropout'])(custom_fc_layers)

    # --------------------------------================ Tabular data =================--------------------------------- #
    image_type_input = Input(shape=(2,), name='image_type', dtype=dtypes.float32)
    sex_input = Input(shape=(2,), name='sex', dtype=dtypes.float32)
    anatom_site_input = Input(shape=(6,), name='anatom_site_general', dtype=dtypes.float32)
    age_input = Input(shape=(10,), name='age_approx', dtype=dtypes.float32)
    concat_inputs = Concatenate()([image_type_input, sex_input, anatom_site_input, age_input])
    concat_inputs = Dropout(rate=args['dropout'])(concat_inputs)
    custom_fc2_layers = Dense(64, activation=activ, kernel_initializer=init,
                              bias_initializer=init, kernel_regularizer=rglzr, bias_regularizer=rglzr)(concat_inputs)
    custom_fc2_layers = normalization(gamma_initializer=init, beta_initializer=init # moving_mean_initializer=init, moving_variance_initializer=init
                                      )(custom_fc2_layers)
    custom_fc2_layers = Dropout(rate=args['dropout'])(custom_fc2_layers)
    custom_fc2_layers = Dense(32, activation=activ, kernel_initializer=init,
                              bias_initializer=init, kernel_regularizer=rglzr, bias_regularizer=rglzr)(
        custom_fc2_layers)
    custom_fc2_layers = normalization(gamma_initializer=init, beta_initializer=init # moving_mean_initializer=init, moving_variance_initializer=init
                                      )(custom_fc2_layers)
    # custom_fc2_layers = Dropout(rate=args['dropout'])(custom_fc2_layers)
    custom_fc2_layers = Dense(16, activation=activ, kernel_initializer=init,
                              bias_initializer=init, kernel_regularizer=rglzr, bias_regularizer=rglzr)(
        custom_fc2_layers)
    custom_fc2_layers = normalization(gamma_initializer=init, beta_initializer=init # moving_mean_initializer=init, moving_variance_initializer=init
                                      )(custom_fc2_layers)
    # custom_fc2_layers = Dropout(rate=args['dropout'])(custom_fc2_layers)

    # -------------------------------================== Concat part ==================---------------------------------#
    common_layers = Concatenate(axis=1)([custom_fc_layers, custom_fc2_layers])
    common_layers = Dense(16, activation=activ, kernel_initializer=init,
                          bias_initializer=init, kernel_regularizer=rglzr, bias_regularizer=rglzr)(common_layers)
    common_layers = normalization(gamma_initializer=init, beta_initializer=init # moving_mean_initializer=init, moving_variance_initializer=init
                                  )(common_layers)
    common_layers = Dropout(rate=args['dropout'])(common_layers)
    common_layers = Dense(16, activation=activ, kernel_initializer=init,
                          bias_initializer=init, kernel_regularizer=rglzr, bias_regularizer=rglzr)(common_layers)
    output_layer = Dense(args['num_classes'], activation='softmax', kernel_initializer=init,
                         bias_initializer=init, kernel_regularizer=rglzr, bias_regularizer=rglzr,
                         name='class')(common_layers)
    return Model([image_input, image_type_input, sex_input, anatom_site_input, age_input], [output_layer])
