import random

import PIL.Image
import torch, os, sys
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class GenerateCGMHDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.image_path = os.path.join(self.root_path, "Image/")
        self.label_path = os.path.join(self.root_path, "Label/")
        self.path_set = []
        for path in os.listdir(self.image_path):
            if path.endswith(".png"):
                self.path_set.append(os.path.join(self.image_path,path))
        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

    def __len__(self):
        return len(self.path_set)

    def __getitem__(self, item):
        path = self.path_set[item]
        image_path = path
        label_path = path.replace("Image/", "Label/")
        image = PIL.Image.open(image_path).convert("L")
        label = PIL.Image.open(label_path).convert("L")
        image = self.transform(image).float()
        label = (self.transform(label) > 0.5).float()
        if random.random() > 0.5:
            return label, 1, label
        else:
            return image, 0, label

class CGMHDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.image_path = os.path.join(self.root_path, "Image/")
        self.label_path = os.path.join(self.root_path, "Label/")
        self.path_set = []
        for path in os.listdir(self.image_path):
            if path.endswith(".png"):
                self.path_set.append(os.path.join(self.image_path,path))
        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

        from utils.stnaugment import STNAugment
        self.data_aug = STNAugment()

    def apply_transforms(self, image,label, transform, seed=None):
        if seed is None:
            MAX_RAND_VAL = 2147483647
            seed = np.random.randint(MAX_RAND_VAL)

        if transform is not None:
            random.seed(seed)
            torch.random.manual_seed(seed)
            turn_list = []
            turn_list.append(image)
            turn_list.append(label)
            turn_list = self.data_aug(turn_list)
            return turn_list[0],turn_list[1]

    def __len__(self):
        return len(self.path_set)

    def __getitem__(self, item):
        path = self.path_set[item]
        image_path = path
        label_path = path.replace("Image/", "Label/")
        image = PIL.Image.open(image_path).convert("L")
        label = PIL.Image.open(label_path).convert("L")
        image = self.transform(image).float()
        label = (self.transform(label) > 0.5).float()
        image,label = self.apply_transforms(image,label,transforms)
        if_label = random.random() > 0.5
        if if_label:
            return (label) * 2 - 1, 1, label
        else:
            return (image) * 2 - 1, 0, label


def split_train_and_val(dataset,split_ratio = 0.9):
    from sklearn.model_selection import StratifiedShuffleSplit
    labels = [0 for i in range(len(dataset))]
    ss = StratifiedShuffleSplit(n_splits=1, test_size=1 - split_ratio, random_state=0)
    train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    dst_train = torch.utils.data.Subset(dataset, train_indices)
    dst_test = torch.utils.data.Subset(dataset, valid_indices)
    return dst_train,dst_test
