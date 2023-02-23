import torchvision
import torch, os, sys
import numpy as np
from PIL import Image
import argparse
from torch.utils.data import DataLoader, Dataset
from utils.vis_utils import vis_trun
import torch.nn as nn

parser = argparse.ArgumentParser(description='Stage IV')
parser.add_argument('--unet-checkpoint', type=str)
parser.add_argument('--stage3-output', type=str)
parser.add_argument('--stage4-output', type=str)

args = parser.parse_args()

class TestDiceLoss(nn.Module):
    def __init__(self):
        super(TestDiceLoss, self).__init__()

    def forward(self, y_true, y_pred, **kwargs):
        """
        compute mean dice for binary segmentation map via numpy
        """
        intersection = torch.sum(torch.abs(y_pred * y_true), [1, 2, 3])
        mask_sum = torch.sum(torch.abs(y_true), [1, 2, 3]) + torch.sum(torch.abs(y_pred), [1, 2, 3])
        smooth = .000001
        dice = 1 - 2 * (intersection + smooth) / (mask_sum + smooth)
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
        self.indexs = [re.findall(r"\d+", str(self.images[i]))[-1] for i in range(len(self.images))]
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
        return image, mask #, self.indexs[item]


def classifier(model_path):
    from backbone import UNet
    classifier_fn = UNet(n_classes=1, n_channels=1)
    classifier_fn.load_state_dict(torch.load(model_path, map_location="cpu"))
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


def choose(model_path, data_path, save_path, tau=0.2):
    classifier_fn = classifier(model_path)
    dataset = PairDatset(data_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=1)
    pass_list = []
    no_pass_list = []
    dice_loss = TestDiceLoss()
    with torch.no_grad():
        classifier_fn.eval()
        for i, (image, label, indexs) in enumerate(dataloader):
            image, label = image.cuda(), label.cuda()
            pred = (classifier_fn(image).sigmoid() > 0.5).float()
            label = label.float()
            dices = dice_loss(pred, label).tolist()
            print(dices)
            j = 0
            for (dice, index) in zip(dices, indexs):
                if_good = dice < tau
                if if_good:
                    pass_list.append([image[j], label[j]])
                elif dice < tau * 2:
                    pass_list.append([image[j], pred[j]])
                else:
                    no_pass_list.append([image[j], label[j]])
                j += 1
    turn = torchvision.transforms.ToPILImage()
    print(f"{len(pass_list) / (len(pass_list) + len(no_pass_list))}")
    # tau = 1/1 0.7%  4.234%
    # tau = 1/2 0.7%  4.567%
    # tau = 1/3  --   4.725%
    for i in range(len(pass_list)):
        image = pass_list[i][0].cpu()
        mask = pass_list[i][1].cpu()
        image = turn(image)
        mask = turn(mask)
        image.save(f"{save_path}/image_{i}.png")
        mask.save(f"{save_path}/mask_{i}.png")


if __name__ == "__main__":

    choose(args.unet_checkpoint,
           args.stage3_output,
           args.stage4_output,
           0.065)
