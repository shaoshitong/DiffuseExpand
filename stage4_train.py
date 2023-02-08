import torchvision
import torch, os, sys
import numpy as np
from PIL import Image
import argparse
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

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


def choose(model_path, data_path, tau=0.2):
    classifier_fn = classifier(model_path)
    dataset = PairDatset(data_path)
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False)
    pass_list = []
    no_pass_list = []
    dice_loss = DiceLoss()
    with torch.no_grad():
        classifier_fn.eval()
        for i, (image, label, indexs) in enumerate(dataloader):
            image, label = image.cuda(), label.cuda()
            pred = (classifier_fn(image).sigmoid()>0.5).float()
            label = label.float()
            dices = dice_loss(pred, label).tolist()
            print(dices)
            j = 0
            for (dice, index) in zip(dices, indexs):
                if_good = dice < tau
                if if_good:
                    pass_list.append([image[j], label[j]])
                else:
                    no_pass_list.append([image[j], label[j]])
                j += 1
    turn = torchvision.transforms.ToPILImage()
    print(f"{len(pass_list) / (len(pass_list) + len(no_pass_list))}")
    for i in range(len(pass_list)):
        if i < 500:
            image = pass_list[i][0].cpu()
            mask = pass_list[i][1].cpu()
            image = turn(image)
            mask = turn(mask)
            image.save(f"./stage5_tau_0.333/image_{i}.png")
            mask.save(f"./stage5_tau_0.333/mask_{i}.png")


if __name__ == "__main__":
    choose("/home/Bigdata/mtt_distillation_ckpt/COVID19/imagenette/covid19_NO_ZCA/Unet/unet_for_fid.pt", "./a100/stage3_tau_0.333_tmp", 0.065)
