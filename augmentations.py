import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class TFAugmentations:
    def __init__(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def augm(self, input_image):
        translation = tf.random.uniform(shape=[2], seed=self.seed, minval=-10, maxval=10, dtype=tf.float32)
        random_degrees = tf.random.uniform(shape=[1], minval=0, seed=self.seed, maxval=360, dtype=tf.float32)
        rand = tf.random.uniform(shape=[1], minval=0, seed=self.seed, maxval=1., dtype=tf.float32)
        # sharp = tf.random.uniform(shape=[1], minval=0, seed=self.seed, maxval=.1, dtype=tf.float32)

        input_image = tf.image.random_flip_left_right(image=input_image, seed=self.seed)
        input_image = tf.image.random_flip_up_down(image=input_image, seed=self.seed)
        input_image = tf.image.random_brightness(image=input_image, max_delta=0.3, seed=self.seed)
        input_image = tf.image.random_contrast(image=input_image, lower=0.5, upper=1.5, seed=self.seed)
        input_image = tf.image.random_saturation(image=input_image, lower=0.5, upper=1.5, seed=self.seed)
        input_image = tfa.image.translate(input_image, translations=translation, name="Translation")
        input_image = tfa.image.rotate(images=input_image, angles=random_degrees, name="Rotation")
        # input_image = tfa.image.sharpness(image=tf.cast(input_image, dtype=tf.float32), factor=sharp, name="Sharpness")
        input_image = tf.cond(tf.math.greater(0.5, rand), lambda: tfa.image.gaussian_filter2d(image=input_image, sigma=2., name="Gaussian_filter"), lambda: input_image)
        return input_image
