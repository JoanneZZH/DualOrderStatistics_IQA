import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


class KonIQ10KDataset(Dataset):
    """
    load KonIQ-10K dataset
    """

    def __init__(self, mos_df, images_folder, training=True, dist=False):
        """
        Args:
            mos_df (DataFrame): mos detail about KonIQ-10k
            images_folder (str): path to real image
            training (bool, optional): whether in training process. Defaults to True.
            dist (bool, optional): to predict distribution of mos. Defaults to True.
        """
        self.mos_df = mos_df
        self.len = len(self.mos_df)
        self.distribution = dist

        if training:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(3, expand=True),
                transforms.CenterCrop((768, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((768, 1024)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.images_folder = os.path.join(images_folder, '1024x768')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, mos_detail[0])
        image = self.transforms(Image.open(image_path))

        if self.distribution:
            mos_distribution = (mos_detail.c1, mos_detail.c2,
                                mos_detail.c3, mos_detail.c4, mos_detail.c5)
            label = tuple([m/mos_detail.c_total for m in mos_distribution])
        else:
            label = [mos_detail[1] / 5]
        return image, torch.Tensor(label)


class LiveCDataSet(Dataset):
    def __init__(self, mos_df, images_folder, training=True):
        self.mos_df = mos_df
        self.len = len(self.mos_df)

        if training:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(3, expand=True),
                transforms.CenterCrop((500, 500)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((500, 500)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


        self.images_folder = images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, mos_detail[0])
        image = self.transforms(Image.open(image_path))
        label = [mos_detail[1] / 20]
        return image, torch.Tensor(label)

class Tid2013DataSet(Dataset):
    def __init__(self, mos_df, images_folder, training=True):
        self.mos_df = mos_df
        self.len = len(self.mos_df)

        if training:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(3, expand=True),
                transforms.CenterCrop((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((512, 384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.images_folder = images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, mos_detail[1])
        image = self.transforms(Image.open(image_path))
        label = [mos_detail[0]]
        return image, torch.Tensor(label)

class Kadid10KDataSet(Dataset):
    def __init__(self, mos_df, images_folder, training=True):
        self.mos_df = mos_df
        self.len = len(self.mos_df)

        if training:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(3, expand=True),
                transforms.CenterCrop((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.images_folder = images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, mos_detail[0])
        image = self.transforms(Image.open(image_path))
        label = [mos_detail[1] / 5]
        return image, torch.Tensor(label)

class BidDataSet(Dataset):
    def __init__(self, mos_df, images_folder, training=True):
        self.mos_df = mos_df
        self.len = len(self.mos_df)

        if training:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(3, expand=True),
                transforms.CenterCrop((200, 200)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((200, 200)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.images_folder = images_folder

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        mos_detail = self.mos_df.iloc[index]
        image_path = os.path.join(self.images_folder, 'DatabaseImage{}.JPG'.format(str(int(mos_detail[0])).zfill(4)))
        image = self.transforms(Image.open(image_path))
        label = [mos_detail[1] / 5]
        return image, torch.Tensor(label)