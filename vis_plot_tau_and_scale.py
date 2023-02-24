import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser(description='Finetune Diffusion Model')
parser.add_argument('--dataset', type=str, default='COVID19', help='dataset')
parser.add_argument('--loss_type', type=str, default='mse', help='loss type')
parser.add_argument('--learn_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training networks')
parser.add_argument('--data_path', type=str,
                    default='./covid-chestxray-dataset/images/',
                    help='dataset path')
parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
parser.add_argument('--csv_path', type=str,
                    default="./covid-chestxray-dataset/metadata.csv")
parser.add_argument('--save_path', type=str, default="/home/Bigdata/mtt_distillation_ckpt/stage2")
parser.add_argument('--unet_ckpt_path', type=str,
                    default="/home/sst/product/diffusion-model-learning/demo/256x256_diffusion.pt")
parser.add_argument('--class_cond', type=bool, default=True)
parser.add_argument('--num_classes_1', type=int, default=2)
parser.add_argument('--num_classes_2', type=int, default=-1)
parser.add_argument('--cuda_devices', type=str, default="0", help="data parallel training")


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    intersection = torch.sum(torch.abs(y_pred * y_true))
    mask_sum = torch.sum(torch.abs(y_true)) + torch.sum(torch.abs(y_pred))
    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice



class PairDatset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = []
        self.masks = []
        self.turn = torchvision.transforms.ToTensor()
        for root, dirs, files in os.walk(data_path):
            for file in files:
                path = str(os.path.join(self.data_path, file))

                if file.startswith("image_"):
                    self.images.append(path)
                elif file.startswith("mask_"):
                    self.masks.append(path)
                else:
                    continue
        import re
        self.indexs = [re.findall(r"\d+",str(self.images[i]))[-1] for i in range(len(self.images))]
        # print(self.indexs)
        # exit(-1)
    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, item):
        image_path = os.path.join(self.data_path, "image_" + str(self.indexs[item]) + ".png")
        mask_path = os.path.join(self.data_path, "mask_" + str(self.indexs[item]) + ".png")
        image, mask = Image.open(image_path).convert("L"), Image.open(mask_path).convert("L")
        image, mask = self.turn(image), self.turn(mask)
        mask = (mask > 0.5).float()
        return image, mask, self.indexs[item]


def classifier(model_path="/home/Bigdata/mtt_distillation_ckpt/COVID19/stage4_tau_0.5/stage3_model_5000.pt"):
    from backbone import UNet
    classifier_fn = UNet(n_classes=1,n_channels=1)
    classifier_fn.load_state_dict(torch.load(model_path,map_location="cpu"))
    classifier_fn = classifier_fn.cuda()
    return classifier_fn


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        pre = predict.view(num, -1)
        tar = target.view(num, -1)
        intersection = (pre * tar).sum(-1)  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1)
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score


def choose(model_path, data_path):
    classifier_fn = classifier(model_path)
    dataset = PairDatset(data_path)
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False)

    dice_loss = mean_dice_np
    dice_list = []
    with torch.no_grad():
        classifier_fn.eval()
        for i, (image, label, indexs) in enumerate(dataloader):
            image, label = image.cuda(), label.cuda()
            pred = (classifier_fn(image).sigmoid()>0.5).float()
            label = (label > 0.5).float()
            for j in range(pred.shape[0]):
                dice = dice_loss(pred[j],label[j])
                dice_list.append(dice)
    dice_list = torch.stack(dice_list)
    nums , _ = torch.histogram(dice_list.clone().cpu(),10,range = (0,1))
    nums = nums/torch.sum(nums)
    nums = nums.tolist()
    print(nums, _ )

if __name__ == "__main__":
    choose("./buffers/CGMH/imagenette/CGMH/unet_for_cgmh_fid.pt", "/home/Bigdata/medical_dataset/output/CGMH/tau_0.2_scale_1.0")
