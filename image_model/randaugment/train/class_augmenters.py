import random
from .augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from .augmenters.color.hedcoloraugmenter import HedColorAugmenter
from .augmenters.noise.gaussianbluraugmenter import GaussianBlurAugmenter
from .augmenters.noise.additiveguassiannoiseaugmenter import AdditiveGaussianNoiseAugmenter
from .augmenters.spatial.scalingaugmenter import ScalingAugmenter
from albumentations.augmentations.geometric.functional import elastic_transform
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from torchvision.transforms import v2


_REPLACE = 128

class Scaling:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        if random.random() > 0.5:
            factor = self.factor / 60
            augmentor = ScalingAugmenter(scaling_range=(1 - factor, 3), interpolation_order=1)
        else:
            factor = self.factor / 30
            augmentor = ScalingAugmenter(scaling_range=(1 + factor, 3), interpolation_order=1)
        image = augmentor.transform(image)
        return image

    def __repr__(self):
        return self.__class__.__name__

class HsvH:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        factor = self.factor / 30
        if random.random() > 0.5:
            factor = -factor
        augmentor = HsbColorAugmenter(hue_sigma_range=factor, saturation_sigma_range=0, brightness_sigma_range=0)
        image = augmentor.transform(image)
        return image

    def __repr__(self):
        return self.__class__.__name__

class HsvS:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        factor = self.factor / 30
        if random.random() > 0.5:
            factor = -factor
        augmentor = HsbColorAugmenter(hue_sigma_range=0, saturation_sigma_range=factor,
                                      brightness_sigma_range=0)
        image = augmentor.transform(image)
        return image

    def __repr__(self):
        return self.__class__.__name__

class HsvV:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        factor = self.factor / 30
        if random.random() > 0.5:
            factor = -factor
        augmentor = HsbColorAugmenter(hue_sigma_range=0, saturation_sigma_range=0,
                                      brightness_sigma_range=factor)
        image = augmentor.transform(image)
        return image

    def __repr__(self):
        return self.__class__.__name__

class Hsv:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        factor = self.factor / 30
        if random.random() > 0.5:
            factor = -factor
        augmentor = HsbColorAugmenter(hue_sigma_range=factor, saturation_sigma_range=factor,
                                      brightness_sigma_range=factor)
        augmentor.randomize()
        image = augmentor.transform(image)
        return image

    def __repr__(self):
        return self.__class__.__name__

class Hed:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        factor = self.factor / 30
        if random.random() > 0.5:
            factor = -factor
        augmentor = HedColorAugmenter(haematoxylin_sigma_range=factor, haematoxylin_bias_range=factor,
                                      eosin_sigma_range=factor, eosin_bias_range=factor,
                                      dab_sigma_range=factor, dab_bias_range=factor,
                                      cutoff_range=(0.15, 0.85))
        augmentor.randomize()
        image = augmentor.transform(image)
        return image

    def __repr__(self):
        return self.__class__.__name__

class HedH:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        factor = self.factor / 30
        if random.random() > 0.5:
            factor = -factor
        augmentor = HedColorAugmenter(haematoxylin_sigma_range=factor, haematoxylin_bias_range=factor,
                                      eosin_sigma_range=0, eosin_bias_range=0,
                                      dab_sigma_range=0, dab_bias_range=0,
                                      cutoff_range=(0.15, 0.85))
        image = augmentor.transform(image)
        return image

    def __repr__(self):
        return self.__class__.__name__

class HedE:
    def __init__(self, factor, eosin_sigma_range, eosin_bias_range, cutoff_range):
        self.factor = factor
        self.eosin_sigma_range = eosin_sigma_range
        self.eosin_bias_range = eosin_bias_range
        self.cutoff_range = cutoff_range

    def __call__(self, image):
        factor = self.factor / 30
        if random.random() > 0.5:
            factor = -factor
        augmentor = HedColorAugmenter(haematoxylin_sigma_range=0, haematoxylin_bias_range=0,
                                      eosin_sigma_range=factor, eosin_bias_range=factor,
                                      dab_sigma_range=0, dab_bias_range=0,
                                      cutoff_range=self.cutoff_range)
        image = augmentor.transform(image)
        return image

    def __repr__(self):
        return self.__class__.__name__

class HedD:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        factor = self.factor / 30
        if random.random() > 0.5:
            factor = -factor
        augmentor = HedColorAugmenter(haematoxylin_sigma_range=0, haematoxylin_bias_range=0,
                                      eosin_sigma_range=0, eosin_bias_range=0,
                                      dab_sigma_range=factor, dab_bias_range=factor,
                                      cutoff_range=(0.15, 0.85))
        image = augmentor.transform(image)
        return image

    def __repr__(self):
        return self.__class__.__name__

class GaussBlur:
    def __init__(self, factor):
        self.sigma_range = (factor / 5, factor / 5 * 10)

    def __call__(self, image):
        augmentor = GaussianBlurAugmenter(sigma_range=self.sigma_range)
        return augmentor.transform(image)

    def __repr__(self):
        return self.__class__.__name__

class GaussNoise:
    def __init__(self, factor):
        self.sigma_range = (0.1 * factor / 2, factor / 2)

    def __call__(self, image):
        augmentor = AdditiveGaussianNoiseAugmenter(sigma_range=self.sigma_range)
        return augmentor.transform(image)

    def __repr__(self):
        return self.__class__.__name__

class Elastic:
    def __init__(self, factor):
        self.factor = factor * 7

    def __call__(self, image):
        return elastic_transform(image, alpha=self.factor, sigma=self.factor, alpha_affine=self.factor)

    def __repr__(self):
        return self.__class__.__name__

class Color:
    def __init__(self, factor):
        self.factor = factor / 5 + 1

    def __call__(self, image):
        image = Image.fromarray(image)
        image = ImageEnhance.Color(image).enhance(self.factor)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

class Contrast:
    def __init__(self, factor):
        self.factor = factor / 5 + 1

    def __call__(self, image):
        image = Image.fromarray(image)
        image = ImageEnhance.Contrast(image).enhance(self.factor)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

class Brightness:
    def __init__(self, factor):
        self.factor = factor / 10 + 1

    def __call__(self, image):
        image = Image.fromarray(image)
        image = ImageEnhance.Brightness(image).enhance(self.factor)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

class Rotate:
    def __init__(self, degrees, replace=(_REPLACE, _REPLACE, _REPLACE)):
        self.degrees = degrees * 10
        self.replace = replace

    def __call__(self, image):
        degrees = -self.degrees if random.random() > 0.5 else self.degrees
        image = Image.fromarray(image)
        image = image.rotate(angle=degrees, fillcolor=self.replace)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

class TranslateX:
    def __init__(self, pixels, replace=(_REPLACE, _REPLACE, _REPLACE)):
        self.pixels = pixels * 3
        self.replace = replace

    def __call__(self, image):
        pixels = -self.pixels if random.random() > 0.5 else self.pixels
        image = Image.fromarray(image)
        image = image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=self.replace)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

class TranslateY:
    def __init__(self, pixels, replace=(_REPLACE, _REPLACE, _REPLACE)):
        self.pixels = pixels * 3
        self.replace = replace

    def __call__(self, image):
        pixels = -self.pixels if random.random() > 0.5 else self.pixels
        image = Image.fromarray(image)
        image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=self.replace)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

class ShearX:
    def __init__(self, level, replace=(_REPLACE, _REPLACE, _REPLACE)):
        self.level = level / 20
        self.replace = replace

    def __call__(self, image):
        
        level = -self.level if random.random() > 0.5 else self.level
        image = Image.fromarray(image)
        image = image.transform(image.size, Image.AFFINE, (1, self.level, 0, 0, 1, 0), Image.BICUBIC, fillcolor=level)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

class ShearY:
    def __init__(self, level, replace=(_REPLACE, _REPLACE, _REPLACE)):
        self.level = level / 20
        self.replace = replace

    def __call__(self, image):
        level = -self.level if random.random() > 0.5 else self.level
        image = Image.fromarray(image)
        image = image.transform(image.size, Image.AFFINE, (1, 0, 0, level, 1, 0), Image.BICUBIC, fillcolor=self.replace)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

class Autocontrast:
    def __call__(self, image):
        image = Image.fromarray(image)
        image = ImageOps.autocontrast(image)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

class Identity:
    def __call__(self, image):
        return image

    def __repr__(self):
        return self.__class__.__name__

class Sharpness:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image):
        image = Image.fromarray(image)
        image = ImageEnhance.Sharpness(image).enhance(self.factor)
        return np.asarray(image)

    def __repr__(self):
        return self.__class__.__name__

FACTOR = 4

morphology = v2.Compose([
    # Scaling(FACTOR),
    Elastic(FACTOR),
    GaussNoise(FACTOR),
    GaussBlur(FACTOR),
])

hsv = v2.Compose([
    # Scaling(FACTOR),
    Elastic(FACTOR),
    Hsv(FACTOR),
    Brightness(FACTOR),
    Contrast(FACTOR),
    GaussNoise(FACTOR),
    GaussBlur(FACTOR),
])

hed = v2.Compose([
    # Scaling(FACTOR),
    Elastic(FACTOR),
    Hed(FACTOR),
    Brightness(FACTOR),
    Contrast(FACTOR),
    GaussNoise(FACTOR),
    GaussBlur(FACTOR),
])

bc = v2.Compose([
    # Scaling(FACTOR),
    Elastic(FACTOR),
    Brightness(FACTOR),
    Contrast(FACTOR),
    GaussNoise(FACTOR),
    GaussBlur(FACTOR),
])
