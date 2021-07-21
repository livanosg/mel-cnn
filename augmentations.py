import numpy as np
from cv2 import GaussianBlur, filter2D, warpAffine, getRotationMatrix2D, flip


class Augmentations:
    def __call__(self, input_image):
        self.input_image = input_image
        # Random choice of augmentation method
        all_processes = [self.rotate, self.flips, self.random_translation, self.sharp, self.contrast]
        augm = np.random.choice(all_processes)
        self.input_image = augm()
        if np.random.random() < 0.5:  # Gaussian blurring
            self.input_image = self.gaussian_blur()
        if np.random.random() < 0.5:  # 2nd augmentation:
            all_processes.pop(all_processes.index(augm))
            augm = np.random.choice(all_processes)
            self.input_image = augm()
        return self.input_image

    def rotate(self):
        angle = np.random.randint(-25, 25)
        rows, cols, channels = self.input_image.shape
        m = getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
        return warpAffine(self.input_image, m, (cols, rows))

    def flips(self, ):
        flip_flag = np.random.randint(-1, 2)
        return flip(self.input_image, flip_flag)

    def sharp(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return filter2D(self.input_image, -1, kernel)

    def gaussian_blur(self):
        return GaussianBlur(self.input_image, (3, 3), sigmaX=1.5, sigmaY=1.5)

    def contrast(self):
        contrast_factor = np.random.rand() * 2.
        image_mean = np.mean(self.input_image)
        image_contr = (self.input_image - image_mean) * contrast_factor + image_mean
        return image_contr

    def random_translation(self):
        x = np.random.random_integers(-80, 80)
        y = np.random.random_integers(-80, 80)
        m = np.float32([[1, 0, x], [0, 1, y]])
        rows, cols, channels = self.input_image.shape
        return warpAffine(self.input_image, m, (cols, rows))
