import torchxrayvision as xrv
from .stnaugment import STNAugment
import os, sys, math, random, torch
import zipfile
import imageio
from PIL import Image
from torchvision import transforms
import numpy as np
from torchxrayvision.datasets import apply_transforms
from torch.utils.data import DataLoader, Dataset


def normalize(img, reshape=False, z_norm=False):
    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")
        # add color channel
        img = img[None, :, :]
    img = torch.from_numpy(img.astype(np.float32) / 255)
    if z_norm:
        img = 2 * img - 1.
    return img


class COVID19Dataset(xrv.datasets.COVID19_Dataset):
    def __init__(self,
                 imgpath,
                 csvpath,
                 views=["PA", "AP"],
                 transform=None,
                 semantic_masks=False
                 ):
        super(COVID19Dataset, self).__init__(
            imgpath=imgpath,
            csvpath=csvpath,
            views=views,
            transform=transform,
            semantic_masks=semantic_masks
        )
        self.data_aug = STNAugment()
        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

    def apply_transforms(self, sample, transform, seed=None):
        if seed is None:
            MAX_RAND_VAL = 2147483647
            seed = np.random.randint(MAX_RAND_VAL)

        if transform is not None:
            random.seed(seed)
            torch.random.manual_seed(seed)
            turn_list = []
            turn_list.append(sample["img"])
            if "semantic_masks" in sample:
                for i in sample["semantic_masks"].keys():
                    turn_list.append(sample["semantic_masks"][i])
            turn_list = self.data_aug(turn_list)
            sample["img"] = turn_list[0]
            for i, name in enumerate(sample["semantic_masks"].keys()):
                sample["semantic_masks"][name] = turn_list[i + 1]
        return sample

    def get_semantic_mask_dict(self, image_name):

        archive_path = "semantic_masks_v7labs_lungs/" + image_name
        semantic_masks = {}
        if archive_path in self.semantic_masks_v7labs_lungs_namelist:
            with zipfile.ZipFile(self.semantic_masks_v7labs_lungs_path).open(archive_path) as file:
                mask = imageio.imread(file.read())
                mask = Image.fromarray(mask).convert("L")
                semantic_masks["Lungs"] = mask

        return semantic_masks

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        imgid = self.csv['filename'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = Image.open(img_path).convert('L')
        sample["img"] = img
        if self.semantic_masks:
            sample["semantic_masks"] = self.get_semantic_mask_dict(imgid)
        sample = apply_transforms(sample, self.transform)
        mask = (sample["semantic_masks"]["Lungs"] == 1.).float()
        sample["semantic_masks"]["Lungs"] = mask

        sample = self.apply_transforms(sample, self.data_aug)
        return sample


class CleanCOVID19Dataset(Dataset):
    def __init__(self, samples, dataset):
        self.samples = samples
        self.dataset = dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        idx = self.samples[item]
        sample = self.dataset[idx]
        return sample["img"].float(), sample["semantic_masks"]["Lungs"].float()


def clean_dataset(dataset):
    assert dataset.semantic_masks, "only turn segmentation task"
    samples = []
    for idx in range(len(dataset)):
        imgid = dataset.csv['filename'].iloc[idx]
        archive_path = "semantic_masks_v7labs_lungs/" + imgid
        if archive_path in dataset.semantic_masks_v7labs_lungs_namelist:
            samples.append(idx)
    return CleanCOVID19Dataset(samples, dataset)


class GenerateCOVID19Dataset(Dataset):
    def __init__(self, samples, dataset):
        self.samples = samples
        self.dataset = dataset

    def __len__(self):
        return 2 * len(self.samples)

    def __getitem__(self, item) -> [torch.Tensor,int,int]:
        if_label = (int(item // len(self.samples)) == 0)
        item = item % len(self.samples)
        idx = self.samples[item]
        sample = self.dataset[idx]
        if if_label:
            return sample["semantic_masks"]["Lungs"].float(),1,item
        else:
            return sample["img"].float(),0,item

def generate_clean_dataset(dataset):
    assert dataset.semantic_masks, "only turn segmentation task"
    samples = []
    for idx in range(len(dataset)):
        imgid = dataset.csv['filename'].iloc[idx]
        archive_path = "semantic_masks_v7labs_lungs/" + imgid
        if archive_path in dataset.semantic_masks_v7labs_lungs_namelist:
            samples.append(idx)
    return GenerateCOVID19Dataset(samples, dataset)
