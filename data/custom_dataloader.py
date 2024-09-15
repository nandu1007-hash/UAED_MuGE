import torch
from torch.utils import data
import os
from os.path import join
import imageio
import torchvision.transforms as transforms

class CustomImageDataset(data.Dataset):
    """
    Custom Dataset for images with 512x512 size.
    """
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        if self.split == 'train':
            self.filelist = [join(self.root, 'train', file) for file in os.listdir(join(self.root, 'train')) if file.endswith(('.png', '.jpg', '.jpeg'))]
        elif self.split == 'test':
            self.filelist = [join(self.root, 'test', file) for file in os.listdir(join(self.root, 'test')) if file.endswith(('.png', '.jpg', '.jpeg'))]
        else:
            raise ValueError("Invalid split type! Use 'train' or 'test'.")
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        img_file = self.filelist[index]
        img = imageio.imread(img_file)

        if img.shape[:2] != (512, 512):
            raise ValueError(f"Image size mismatch! Expected 512x512, but got {img.shape[:2]}")

        if self.transform:
            img = self.transform(img)

        # For the sake of simplicity, let's assume you don't have labels for now
        return img
