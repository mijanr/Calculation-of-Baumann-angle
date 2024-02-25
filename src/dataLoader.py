from torch.utils.data import Dataset
import albumentations as A
import torch
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.images[idx].squeeze().numpy()
            label = list(tuple(self.labels[idx].numpy()))
            transformed = self.transform(image=img, keypoints=label)
            transformed_img = torch.Tensor(transformed['image']).unsqueeze(0)
            transformed_label = torch.Tensor(transformed['keypoints'])
            return transformed_img, transformed_label
        return self.images[idx], self.labels[idx]


