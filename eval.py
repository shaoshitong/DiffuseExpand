import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from timm.scheduler import CosineLRScheduler
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from util import (ParamDiffAug, TensorDataset, epoch2, get_daparam,
                  get_dataset, get_network)
from utils import DiceLoss
from utils.covid19_dataset import STNAugment

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestDiceLoss(nn.Module):
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
        super(TestDiceLoss, self).__init__()
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
        score = 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score.mean()

class PairDatset(Dataset):
    def __init__(self, data_path,if_randaugment=False):
        self.data_path = data_path
        self.images = []
        self.masks = []
        self.if_randaugment = if_randaugment
        self.turn = torchvision.transforms.ToTensor()
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if len(self.images)>500:
                    break
                path = str(os.path.join(self.data_path, file))
                if file.startswith("image_"):
                    self.images.append(path)
                elif file.startswith("mask_"):
                    self.masks.append(path)
                else:
                    continue
        self.indexs = [i for i in range(len(self.images))]
        self.data_aug = STNAugment()

    def apply_transforms(self, image, mask, transform, seed=None):
        if transform is not None:
            turn_list = [image, mask]
            turn_list = self.data_aug(turn_list)
            return turn_list[0], turn_list[1]

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, item):
        image_path = os.path.join(self.data_path, "image_" + str(self.indexs[item]) + ".png")
        mask_path = os.path.join(self.data_path, "mask_" + str(self.indexs[item]) + ".png")
        image, mask = Image.open(image_path).convert("L"), Image.open(mask_path).convert("L")
        image, mask = self.turn(image), self.turn(mask)
        mask = (mask > 0.5).float()
        if self.if_randaugment:
            return self.apply_transforms(image, mask, self.data_aug)
        return image, mask


def main(args):
    with  open("./outputs/" + f"{args.generate_data_path.split('/')[-1]}" + f"_model_{args.model}_no_{random.random()}.txt", "w") as ff: #  f"{args.generate_data_path.split('/')[-1]}"
        args.dsa = True if args.dsa == 'True' else False
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.dsa_param = ParamDiffAug()
        from utils.cgmh_dataset import split_train_and_val
        if args.dataset == "COVID19":
            from utils.covid19_dataset import COVID19Dataset, clean_dataset
            assert args.csv_path != "no", "COVID-19 Segmentation task need csv metadata!"
            dst = COVID19Dataset(imgpath=args.data_path, csvpath=args.csv_path, semantic_masks=True)
            dst = clean_dataset(dst)
            dst_train, dst_test = split_train_and_val(dst,split_ratio=args.ratio)
            dst_train_2 = PairDatset(args.generate_data_path)
            dst_train =  ConcatDataset([dst_train_2,dst_train])
        elif args.dataset == "CGMH":
            from utils.cgmh_dataset import CGMHDataset
            dst_train = CGMHDataset(args.data_path,if_val=True)
            dst_train, dst_test = split_train_and_val(dst_train)
            dst_train_2 = PairDatset(args.generate_data_path)
            dst_train =  ConcatDataset([dst_train_2,dst_train])

        else:
            raise NotImplementedError

        # print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)

        save_dir = os.path.join("./checkpoint", args.dataset)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ''' organize the real dataset '''
        print("BUILDING DATASET")
        print("total train images %d" % (len(dst_train)))
        if args.loss_type == "cross":
            criterion = nn.CrossEntropyLoss().to(args.device)
        elif args.loss_type == "l1":
            criterion = nn.L1Loss().to(args.device)
        elif args.loss_type == "sigmoid_l1":
            c_loss = nn.L1Loss().to(args.device)
            criterion = lambda x, y, tau=1: c_loss(torch.sigmoid(x / tau), y)
        elif args.loss_type == "bce":
            c_loss = nn.BCELoss().to(args.device)
            criterion = lambda x, y: c_loss(torch.sigmoid(x), y)
        else:
            raise NotImplementedError
        criterion_dice = DiceLoss()
        criterion_test_dice = TestDiceLoss()
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_train, shuffle=True, num_workers=4)

        ''' Train synthetic data '''
        teacher_net = get_network(args.model, 1, 1, 256).to(args.device)  # get a random model
        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr,
                                         weight_decay=args.l2)  # optimizer_img for synthetic data
        teacher_optim.zero_grad()
        scheduler = CosineLRScheduler(teacher_optim, args.train_epochs * len(trainloader), lr_min=1e-7,
                                      warmup_lr_init=lr * 0.01,
                                      warmup_t=5 * len(trainloader), t_in_epochs=False)
        scaler = torch.cuda.amp.GradScaler()
        iter = 0
        for e in range(args.train_epochs):
            train_loss, train_dice, train_psnr = epoch2("train", dataloader=trainloader, net=teacher_net,
                                                        optimizer=teacher_optim, scheduler=scheduler, iter=iter,
                                                        scaler=scaler,
                                                        criticion=criterion, criticion_dice=criterion_dice, args=args)

            test_loss, test_dice, test_psnr = epoch2("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                                     scheduler=scheduler, iter=iter, scaler=scaler,
                                                     criticion=criterion, criticion_dice=criterion_test_dice, args=args)
            iter += len(trainloader)
            log = "Epoch: {}\tIter: {}\tLr: {}\tTrain PSNR: {}\tTrain DICE: {}\tTest PSNR: {}\tTest DICE: {}".format(e,
                                                                                                                     iter,
                                                                                                                     scheduler._get_lr(
                                                                                                                         iter)[
                                                                                                                         0],
                                                                                                                     train_psnr,
                                                                                                                     train_dice,
                                                                                                                     test_psnr,
                                                                                                                     test_dice)
            print(log)
            ff.write(log + "\n")
    #
    # print("Saving {}".format(os.path.join(save_dir, "unet_for_cgmh_fid.pt")))
    # torch.save(teacher_net.state_dict(), os.path.join(save_dir, "unet_for_cgmh_fid.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CGMH', help='dataset')
    parser.add_argument('--model', type=str, default='Unet', help='model')
    parser.add_argument('--loss_type', type=str, default='sigmoid_l1', help='loss type')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=16, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=16, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--data_path', type=str, default='./CGMH_PelvisSegment', help='dataset path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument("--generate_data_path", type=str, default="./output/CGMH/tau_0.5_scale_1.0")
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--ratio', type=float,default=0.9)
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--csv_path', type=str, default="no")
    args = parser.parse_args()
    main(args)

"""
python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 \
--data_path=./CGMH_PelvisSegment \
--csv_path=./covid-chestxray-dataset-master/metadata.csv 
"""
