import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from util import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
import copy
from timm.scheduler import CosineLRScheduler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    assert args.dataset in ["COVID19",],"The target of segmentation dataset distillation must be segmentation dataset!"
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset  == "COVID19":
        save_dir = os.path.join(save_dir, args.subset, "covid19")
        save_dir += "_NO_ZCA"
    else:
        raise NotImplementedError

    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ''' organize the real dataset '''
    print("BUILDING DATASET")
    print("total train images %d"%(len(dst_train)))
    if args.loss_type == "cross":
        criterion = nn.CrossEntropyLoss().to(args.device)
    elif args.loss_type == "l1":
        criterion = nn.L1Loss().to(args.device)
    elif args.loss_type == "sigmoid_l1":
        c_loss = nn.L1Loss().to(args.device)
        criterion = lambda x,y,tau=0.05:c_loss(torch.sigmoid(x/tau),y)
    elif args.loss_type == "bce":
        c_loss = nn.BCELoss().to(args.device)
        criterion = lambda x,y:c_loss(torch.sigmoid(x),y)
    else:
        raise NotImplementedError

    trajectories = []
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=4)

    for it in range(0, args.num_experts):

        ''' Train synthetic data '''
        print(f"Begin training expert {it}")
        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device) #  get a random model
        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr, weight_decay=args.l2)  # optimizer_img for synthetic data
        teacher_optim.zero_grad()
        timestamps = []
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
        scheduler = CosineLRScheduler(teacher_optim, args.train_epochs * len(trainloader), lr_min=1e-7, warmup_lr_init=lr * 0.01,
                                      warmup_t=5 * len(trainloader), t_in_epochs=False)
        scaler = torch.cuda.amp.GradScaler()

        iter = 0
        for e in range(args.train_epochs):

            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,scheduler = scheduler, iter = iter, scaler = scaler,
                                        criterion=criterion, args=args)

            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,scheduler = scheduler, iter = iter, scaler = scaler,
                                        criterion=criterion, args=args)
            iter += args.batch_train
            print("Itr: {}\tEpoch: {}\tIter: {}\tLr: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, iter,scheduler._get_lr(iter)[0], train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])


        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='COVID19', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='Unet', help='model')
    parser.add_argument('--loss_type', type=str, default='sigmoid_l1', help='loss type')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=16, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=16, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--csv_path',type=str,default="no")
    args = parser.parse_args()
    main(args)


"""
python segmentation_buffer.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 \
--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv 
"""
