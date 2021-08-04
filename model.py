import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, Dropout, GlobalAvgPool1D, LeakyReLU,\
    Dense, Concatenate, LSTM, Reshape, Embedding, MultiHeadAttention, Add, AveragePooling2D, ReLU
from tensorflow.keras.activations import swish
from tensorflow.keras.regularizers import l1_l2
from tensorflow import dtypes


class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def model_fn(args):
    main_conf = np.asarray([8, 16])
    layers = {1: main_conf,
              2: main_conf * 2,
              3: main_conf * 4}

    patch_size = 1
    projection_dim = 16
    num_heads = 16

    activ = ReLU
    rglzr = l1_l2(l1=0., l2=0.0002)
    normalization = LayerNormalization
    # -------------------------------================= Image data =================----------------------------------- #
    base_model = args['model'](include_top=False, input_shape=args['input_shape'])
    base_model.trainable = False
    image_input = Input(shape=args['input_shape'], name='image')
    base_model = base_model(image_input, training=False)
    conv1x1ap = AveragePooling2D(padding='same', strides=1)(base_model)
    conv1x1ap = Conv2D(layers[args['layers']][1], kernel_size=1, padding='same')(conv1x1ap)
    conv1x1ap = normalization()(conv1x1ap)
    conv1x1ap = Dropout(rate=args['dropout_ratio'])(conv1x1ap)
    conv1x1 = Conv2D(layers[args['layers']][1], kernel_size=1, padding='same')(base_model)
    conv1x1 = normalization()(conv1x1)
    conv1x1 = Dropout(rate=args['dropout_ratio'])(conv1x1)
    conv1x1_1x3_3x1 = Conv2D(layers[args['layers']][0], kernel_size=1, padding='same')(base_model)
    conv1x1_1x3 = Conv2D(layers[args['layers']][1], kernel_size=(1, 3), padding='same')(conv1x1_1x3_3x1)
    conv1x1_1x3 = normalization()(conv1x1_1x3)
    conv1x1_1x3 = Dropout(rate=args['dropout_ratio'])(conv1x1_1x3)
    conv1x1_3x1 = Conv2D(layers[args['layers']][1], kernel_size=(3, 1), padding='same')(conv1x1_1x3_3x1)
    conv1x1_3x1 = normalization()(conv1x1_3x1)
    conv1x1_3x1 = Dropout(rate=args['dropout_ratio'])(conv1x1_3x1)
    conv1x1_2 = Conv2D(layers[args['layers']][0], kernel_size=1, padding='same')(base_model)
    conv1x3_3x1 = Conv2D(layers[args['layers']][1], kernel_size=(1, 3), padding='same')(conv1x1_2)
    conv3x1_3x1_1x3 = Conv2D(layers[args['layers']][1], kernel_size=(3, 1), padding='same')(conv1x3_3x1)
    conv1x1_2_1x3 = Conv2D(layers[args['layers']][1], kernel_size=(1, 3), padding='same')(conv3x1_3x1_1x3)
    conv1x1_2_1x3 = normalization()(conv1x1_2_1x3)
    conv1x1_2_1x3 = Dropout(rate=args['dropout_ratio'])(conv1x1_2_1x3)
    conv1x1_2_3x1 = Conv2D(layers[args['layers']][1], kernel_size=(3, 1), padding='same')(conv3x1_3x1_1x3)
    conv1x1_2_3x1 = normalization()(conv1x1_2_3x1)
    conv1x1_2_3x1 = Dropout(rate=args['dropout_ratio'])(conv1x1_2_3x1)
    inc_mod = Concatenate()([conv1x1ap, conv1x1, conv1x1_1x3, conv1x1_3x1, conv1x1_2_1x3, conv1x1_2_3x1])

    patches = Patches(patch_size=patch_size)(inc_mod)
    num_patches = (inc_mod.shape[1] // patch_size) ** 2

    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)(patch=patches)
    x1 = normalization(epsilon=1e-6)(encoded_patches)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=args['dropout_ratio'])(x1, x1)
    x2 = Add()([attention_output, encoded_patches])
    x3 = normalization(epsilon=1e-6)(x2)
    custom_avg_pool = GlobalAvgPool1D()(x3)
    custom_fc_layers = normalization(epsilon=1e-6)(custom_avg_pool)
    # --------------------------------================ Tabular data =================--------------------------------- #
    image_type_input = Input(shape=(2,), name='image_type', dtype=dtypes.float32)
    sex_input = Input(shape=(2,), name='sex', dtype=dtypes.float32)
    anatom_site_input = Input(shape=(6,), name='anatom_site_general', dtype=dtypes.float32)
    age_input = Input(shape=(10,), name='age_approx', dtype=dtypes.float32)
    concat_inputs = Concatenate()([image_type_input, sex_input, anatom_site_input, age_input])
    concat_inputs = Reshape(target_shape=(20, 1))(concat_inputs)
    custom_lstm = LSTM(32)(concat_inputs)
    custom_fc2_layers = normalization(epsilon=1e-6)(custom_lstm)
    custom_fc2_layers = Dropout(rate=args['dropout_ratio'])(custom_fc2_layers)
    custom_fc2_layers2 = Dense(16, activation=activ, kernel_regularizer=rglzr)(custom_fc2_layers)
    custom_fc2_layers2 = normalization(epsilon=1e-6)(custom_fc2_layers2)
    custom_fc2_layers2 = Dropout(rate=args['dropout_ratio'])(custom_fc2_layers2)
    # -------------------------------================== Concat part ==================---------------------------------#
    common_layers = Concatenate(axis=1)([custom_fc_layers, custom_fc2_layers2])
    common_layers = Dense(16, activation=activ, kernel_regularizer=rglzr)(common_layers)
    common_layers = normalization(epsilon=1e-6)(common_layers)
    common_layers = Dense(16, activation=activ, kernel_regularizer=rglzr)(common_layers)
    output_layer = Dense(args['num_classes'], activation='softmax', kernel_regularizer=rglzr, name='class')(common_layers)
    return Model([image_input, image_type_input, sex_input, anatom_site_input, age_input], [output_layer])
