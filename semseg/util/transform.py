import random
import math
import numpy as np
import numbers
from collections.abc import Iterable
import torchvision.transforms.functional as F


import torch

class Compose(object):
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label):
#       # [OLD] If 'image' or 'label' come in as torch.Tensor, convert them to NumPy
#        #  so that all the following transforms can use cv2...
#        if isinstance(image, torch.Tensor):
#            image = image.numpy()
#        if isinstance(label, torch.Tensor):
#            label = label.numpy()
        
        for t in self.segtransform:
            image, label = t(image, label)

        # [OLD] convert back to Tensor 
        # ...
        return image, label
"""
 Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label):
        
        # 1) If 'image' or 'label' come in as torch.Tensor, convert them to NumPy
        #    so that all the following transforms (Crop, etc.) can use cv2 safely.
        if isinstance(image, torch.Tensor):
            # move to CPU if needed, convert to numpy
            image = image.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        
        for t in self.segtransform:
            image, label = t(image, label)


         # 3) At the end, convert back to Tensor for PyTorch training
        #    e.g. shape [C,H,W] for image, shape [H,W] for label
        if not isinstance(image, torch.Tensor):
            # If image is H×W×C, transpose to C×H×W
            if image.ndim == 3:
                image = torch.from_numpy(image.transpose((2,0,1))).float()
            elif image.ndim == 2:
        class        # If your image is single-channel, expand dims
                image = torch.from_numpy(image[None,...]).float()
            else:
                raise RuntimeError(f"Unexpected image shape={image.shape}, cannot convert to Tensor")

        if not isinstance(label, torch.Tensor):
            # Usually label is H×W, so keep it that way but cast to Long
            label = torch.from_numpy(label).long()

        return image, label
"""      


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        
        label = torch.from_numpy(label).long()

        #label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return image, label


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label



import torchvision.transforms.functional as F

class Resize(object):
    def __init__(self, size):
        self.size = size  # (h,w) or (w,h)

    def __call__(self, image, label):
        # image shape: (C,H,W), label shape: (H,W)

        # 1) F.resize for image
        image = F.resize(image, self.size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

        # 2) resize label too, but label is (H,W), so we add a channel
        label = label.unsqueeze(0)  # shape: (1,H,W)
        label = F.resize(label, self.size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        label = label.squeeze(0)    # shape back to (H,W)

        return image, label

'''
class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        assert (isinstance(size, Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, label):
        image = cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return image, label
'''

class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, Iterable) and len(scale) == 2)
        if isinstance(scale, Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop(object):
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        self.crop_h, self.crop_w = size if isinstance(size, (tuple,list)) else (size,size)
        self.crop_type   = crop_type
        self.padding     = padding
        self.ignore_label= ignore_label

    def __call__(self, image, label):
        print("DEBUG: image.shape =", image.shape)  # e.g. [C,H,W]
        print("DEBUG: label.shape =", label.shape)  # e.g. [H,W]

        # 1) Get shape from label or from image
        #    If label is [H,W], do:
        h, w = label.shape

        # 2) If we need to pad
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            top  = pad_h // 2
            bot  = pad_h - top
            left = pad_w // 2
            right= pad_w - left

            fill_val_img = 0.0
            if isinstance(self.padding, (tuple,list)) and len(self.padding)>0:
                fill_val_img = float(self.padding[0])  # or just 0

            # Pad image
            image = F.pad(
                image, 
                (left, right, top, bot),  # pad = (left, right, top, bottom)
                fill=fill_val_img,
                padding_mode='constant'
            )
            # Pad label
            label = label.unsqueeze(0) # shape [1,H,W]
            label = F.pad(
                label,
                (left, right, top, bot),
                fill=self.ignore_label,
                padding_mode='constant'
            )
            label = label.squeeze(0)   # shape back to [H,W]

            # Update h,w after padding
            h, w = label.shape

        # 3) Now do random or center crop
        if self.crop_type == 'rand':
            # random
            if h>self.crop_h:
                top  = random.randint(0, h - self.crop_h)
            else:
                top  = 0
            if w>self.crop_w:
                left = random.randint(0, w - self.crop_w)
            else:
                left = 0

            image = F.crop(image, top, left, self.crop_h, self.crop_w)
            label = label.unsqueeze(0)
            label = F.crop(label, top, left, self.crop_h, self.crop_w)
            label = label.squeeze(0)
        else:
            # center
            image = F.center_crop(image, (self.crop_h, self.crop_w))
            label = label.unsqueeze(0)
            label = F.center_crop(label, (self.crop_h, self.crop_w))
            label = label.squeeze(0)

        return image, label



'''
class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        print("DEBUG: image.shape =", image.shape)
        print("DEBUG: label.shape =", label.shape)
        h, w = label.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            # Suppose 'image' has shape (1, 40, 65, 65).
            # Remove the leading dimension => shape (40, 65, 65).
            image = image.squeeze(0)
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        return image, label
'''

class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            #image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            #label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
            # just F.rotate
            image = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            label = label.unsqueeze(0)
            label = F.rotate(label, angle, interpolation=InterpolationMode.NEAREST, fill=self.ignore_label)
            label = label.squeeze(0)
        
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            import torchvision.transforms.functional as F

            image = F.hflip(image)
            label = F.hflip(label.unsqueeze(0)).squeeze(0)

            #image = cv2.flip(image, 1)
            #label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            # Apply Gaussian blur with a kernel size of (radius, radius)
            image = F.gaussian_blur(image, kernel_size=(self.radius, self.radius))  
        return image, label


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label
