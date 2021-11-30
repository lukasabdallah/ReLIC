from PIL import ImageFilter
from torchvision import transforms
import numpy as np
import torch
from torch import nn


def ColourDistortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.0)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def BlurOrSharpen(radius=2.):
    blur = GaussianBlur(radius=radius)
    full_transform = transforms.RandomApply([blur], p=.5)
    return full_transform


class ImageFilterTransform(object):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, img):
        return img.filter(self.filter)


# class GaussianBlur(ImageFilterTransform):
#
#     def __init__(self, radius=2.):
#         self.filter = ImageFilter.GaussianBlur(radius=radius)
#

class Sharpen(ImageFilterTransform):

    def __init__(self):
        self.filter = ImageFilter.SHARPEN





class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        np.random.seed(0)
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img