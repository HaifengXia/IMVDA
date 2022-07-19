import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 3:
            images = [(val.split()[0], val.split()[1], np.array(int(la) for la in val.split()[2:])) for val in image_list]
        else:
            images = [(val.split()[0], val.split()[1], int(val.split()[2])) for val in image_list]
    return images

def image_loader(path):
    img_arr = np.array(Image.open(path))
    if len(img_arr.shape) == 2:
        img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
    return img_arr


class MultiImagelist(Dataset):
    def __init__(self, image_list, transform=None, labels=None):
        imgs = make_dataset(image_list, labels)
        self.imgs = imgs
        self.transform = transform
        self.loader = image_loader

    def __getitem__(self, index):
        rgb_path, depth_path, target = self.imgs[index]
        sample = {}
        sample['inputs'] = ['rgb', 'depth']
        sample['rgb'] = self.loader(rgb_path)
        sample['depth'] = self.loader(depth_path)
        sample['target'] = target
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.imgs)

def make_dataset_(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array(int(la) for la in val.split()[1:])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def image_loader_(path):
    from PIL import Image
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Imagelist(Dataset):
    def __init__(self, image_list, transform=None, labels=None):
        imgs = make_dataset_(image_list, labels)
        self.imgs = imgs
        self.transform = transform
        self.loader = image_loader_

    def __getitem__(self, index):
        rgb_path, target = self.imgs[index]
        rgb_img = self.loader(rgb_path)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        return rgb_img, target, index

    def __len__(self):
        return len(self.imgs)



def create_loaders(rgbd_path, rgb_path, train_bs, test_bs):
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    from utils.datasets import MultiImagelist, Imagelist
    from utils.transform import Normalise, RandomCrop, ToTensor, ResizeInputs
    from utils.pre_process import image_train

    input_size, crop_size = 256, 224
    normalise_params = [1. / 255,  # Image SCALE
                        np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # Image MEAN
                        np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),  # Image STD
                        1. / 5000]  # Depth SCALE

    multi_transform = transforms.Compose([
        ResizeInputs(input_size),
        RandomCrop(crop_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    # RGB-D sets
    source_set = MultiImagelist(open(rgbd_path).readlines(), transform=multi_transform)
    source_loader = DataLoader(source_set, batch_size=train_bs, shuffle=True, num_workers=0, drop_last=True)

    # RGB sets
    uni_transform = image_train(resize_size=input_size, crop_size=crop_size)
    target_set = Imagelist(open(rgb_path).readlines(), uni_transform)
    target_loader = DataLoader(target_set, batch_size=train_bs, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(target_set, batch_size=test_bs, shuffle=False, num_workers=0, drop_last=False)

    return source_loader, target_loader, test_loader