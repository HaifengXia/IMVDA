import cv2
import numpy as np
import torch

KEYS_TO_DTYPES = {
    'rgb': torch.float,
    'depth': torch.float,
}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        crop_size (int): Desired output size.

    """
    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image = sample['rgb']
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        for key in sample['inputs']:
            sample[key] = self.transform_input(sample[key], top, new_h, left, new_w)
        return sample

    def transform_input(self, input, top, new_h, left, new_w):
        input = input[top : top + new_h, left : left + new_w]
        return input

class ResizeInputs(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        if self.size is None:
            return sample
        size_h = sample['rgb'].shape[0]
        size_w = sample['rgb'].shape[1]
        scale_h = self.size / size_h
        scale_w = self.size / size_w
        inters = {'rgb': cv2.INTER_CUBIC, 'depth': cv2.INTER_NEAREST}
        for key in sample['inputs']:
            inter = inters[key] if key in inters else cv2.INTER_CUBIC
            sample[key] = self.transform_input(sample[key], scale_h, scale_w, inter)
        return sample

    def transform_input(self, input, scale_h, scale_w, inter):
        input = cv2.resize(input, None, fx=scale_w, fy=scale_h, interpolation=inter)
        return input

class Normalise(object):
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (scale * channel - mean) / std

    Args:
        scale (float): Scaling constant.
        mean (sequence): Sequence of means for R,G,B channels respecitvely.
        std (sequence): Sequence of standard deviations for R,G,B channels
            respecitvely.
        depth_scale (float): Depth divisor for depth annotations.

    """
    def __init__(self, scale, mean, std, depth_scale=1.):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.depth_scale = depth_scale

    def __call__(self, sample):
        for key in sample['inputs']:
            if key == 'depth':
                continue
            sample[key] = (self.scale * sample[key] - self.mean) / self.std
        if 'depth' in sample:
            # sample['depth'] = self.scale * sample['depth']
            # sample['depth'] = (self.scale * sample['depth'] - self.mean) / self.std
            if self.depth_scale > 0:
                # sample['depth'] = self.depth_scale * sample['depth']
                sample['depth'] = (self.scale * sample['depth'] - self.mean) / self.std
            elif self.depth_scale == -1:  # taskonomy
                # sample['depth'] = np.log(1 + sample['depth']) / np.log(2.** 16.0)
                sample['depth'] = np.log(1 + sample['depth'])
            elif self.depth_scale == -2:  # sunrgbd
                depth = sample['depth']
                sample['depth'] = (depth - depth.min()) * 255.0 / (depth.max() - depth.min())
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for key in sample['inputs']:
            sample[key] = torch.from_numpy(
                sample[key].transpose((2, 0, 1))
            ).to(KEYS_TO_DTYPES[key] if key in KEYS_TO_DTYPES else KEYS_TO_DTYPES['rgb'])
        # sample['mask'] = torch.from_numpy(sample['mask']).to(KEYS_TO_DTYPES['mask'])
        return sample


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]