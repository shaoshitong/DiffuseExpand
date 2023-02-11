import copy
import torch.nn.functional as F
from utils import NoiseScheduleVP, model_wrapper, DPM_Solver
import argparse
from torch.utils.data import DataLoader
from utils import create_model_and_diffusion
import torch, io
import blobfile as bf

parser = argparse.ArgumentParser(description='Finetune Diffusion Model')
parser.add_argument('--dataset', type=str, default='ISIC', help='dataset')
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
parser2 = copy.deepcopy(parser)
scale_tau = 1
# TODO: V. define guidance_scale
guidance_scale = 10.  # Nothing to do with uncond scenarios


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def create_argparser():
    defaults = dict(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
        data_dir="",
        val_data_dir="",
        noised=True,
        weight_decay=0.0,
        anneal_lr=False,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint=None,
        log_interval=10,
        eval_interval=5,
        save_interval=100,
        channel_mult="",
        lr=1e-4,
        fp16_scale_growth=1e-3,
        lr_anneal_steps=300,
        isic=True,
    )

    diffusion_defaults = dict(
        learn_sigma=False,  # TODO; MUST BE FALSE
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    defaults.update(diffusion_defaults)

    # TODO: classifier is not need
    # classifier_defaults=dict(
    #         image_size=64,
    #         classifier_use_fp16=False,
    #         classifier_width=128,
    #         classifier_depth=2,
    #         classifier_attention_resolutions="32,16,8",  # 16
    #         classifier_use_scale_shift_norm=True,  # False
    #         classifier_resblock_updown=True,  # False
    #         classifier_pool="attention",
    #     )
    # defaults.update(classifier_defaults)

    add_dict_to_argparser(parser, defaults)
    return parser


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


# TODO: Definition

args = create_argparser().parse_args()
BATCHSIZE = args.batch_size
label1 = torch.ones(BATCHSIZE).long().cuda()

NAME = [
    "image_size",
    "class_cond",
    "learn_sigma",
    "num_channels",
    "num_res_blocks",
    "channel_mult",
    "num_heads",
    "num_head_channels",
    "num_heads_upsample",
    "attention_resolutions",
    "dropout",
    "diffusion_steps",
    "noise_schedule",
    "timestep_respacing",
    "use_kl",
    "predict_xstart",
    "rescale_timesteps",
    "rescale_learned_sigmas",
    "use_checkpoint",
    "use_scale_shift_norm",
    "resblock_updown",
    "use_fp16",
    "use_new_attention_order",
    "num_classes_1",
    "num_classes_2",
    "isic"
]

# TODO: I. define model
args.num_classes_2 = 1
_model_fn, diffusion = create_model_and_diffusion(
    **args_to_dict(args, NAME)
)
import os, sys

model_path = "/home/Bigdata/mtt_distillation_ckpt/model_isic_stage2_30000.pt"
if not os.path.exists(model_path):
    raise KeyError

_model_fn.load_state_dict(torch.load(model_path, map_location="cpu"))
_model_fn.cuda()


@torch.no_grad()
def model(x, t, **kwargs):
    B, C = x.shape[:2]
    model_output = _model_fn(x, t, **kwargs)
    return model_output


label2 = None
# TODO: II. define model_kwargs
model_kwargs = {"y1": label1, "y2": label2}

# TODO: III. define condition
condition = None

# TODO: IV. define unconditional_condition
unconditional_condition = None  # Nothing to do with guidance-classifier scenarios

# TODO: VI. define classifier
from utils import create_classifier_and_diffusion


def create_argparser_2():
    defaults = dict(
        iterations=5000,
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
        data_dir="",
        val_data_dir="",
        noised=True,
        weight_decay=0.0,
        anneal_lr=False,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint=None,
        log_interval=10,
        eval_interval=5,
        save_interval=1000,
        channel_mult="",
        lr=3e-4,
        fp16_scale_growth=1e-3,
        lr_anneal_steps=30000,
        isic=True,
    )

    diffusion_defaults = dict(
        learn_sigma=False,  # TODO; MUST BE FALSE
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    defaults.update(diffusion_defaults)

    # TODO: classifier is not need
    classifier_defaults = dict(
        image_size=256,
        classifier_use_fp16=True,
        classifier_width=64,
        classifier_depth=2,
        classifier_attention_resolutions="16",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )
    defaults.update(classifier_defaults)

    add_dict_to_argparser(parser2, defaults)
    return parser2


args_2 = create_argparser_2().parse_args()
NAME = [
    "image_size",
    "classifier_use_fp16",
    "classifier_width",
    "classifier_depth",
    "classifier_attention_resolutions",
    "classifier_use_scale_shift_norm",
    "classifier_resblock_updown",
    "classifier_pool",
    "learn_sigma",
    "diffusion_steps",
    "noise_schedule",
    "timestep_respacing",
    "use_kl",
    "predict_xstart",
    "rescale_timesteps",
    "rescale_learned_sigmas",
    "num_classes_1",
    "num_classes_2",
    "isic",
]
# TODO: Define UNet and diffusion scheduler
args.num_classes_2 = 1
classifier_fn, _ = create_classifier_and_diffusion(
    **args_to_dict(args_2, NAME)
)

model_path_2 = "/home/Bigdata/mtt_distillation_ckpt/stage3_isic_model_10000.pt"
if not os.path.exists(model_path_2):
    raise KeyError
model_params = torch.load(model_path_2, map_location="cpu")
classifier_fn.load_state_dict(model_params)
classifier_fn = classifier_fn.cuda()

# TODO: VII. define classifier_kwargs
classifier_kwargs = {}  # Nothing to do with uncond scenarios

# TODO: VIII. define betas
from utils import get_named_beta_schedule

betas = torch.from_numpy(get_named_beta_schedule("linear", 1000)).cuda()

noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
image_shape = (BATCHSIZE, 3, 256, 256)

for j in range(0, 20, args.batch_size):
    label2 = None

    model_kwargs = {"y1": label1, "y2": label2}


    def condition_1(x1, x2, y=(model_kwargs["y2"])):
        sig_x = torch.nn.functional.log_softmax(x2 / scale_tau)[label1]
        sig_x = sig_x.mean()
        print("stage 1:",sig_x)
        return sig_x


    classifier_1 = lambda x, t, cond: cond(*classifier_fn(torch.clamp(x/2+0.5,0,1), t))

    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="noise",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
        guidance_type="classifier",
        condition=condition_1,
        guidance_scale=guidance_scale,
        classifier_fn=classifier_1,
        classifier_kwargs=classifier_kwargs,
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                            correcting_x0_fn="dynamic_thresholding")
    x_T = torch.randn(image_shape).cuda()
    with torch.no_grad():
        y_label = dpm_solver.sample(
            x_T,
            steps=30,
            order=3,
            skip_type="time_uniform",
            method="multistep",
        )
        acc = torch.Tensor([0.2989, 0.5870, 0.1140])[None, :, None, None].cuda()
        y_label = (y_label).min(1, keepdim=True)[0].expand(-1, 3, -1, -1)

    model_kwargs = {"y1": label1 * 0, "y2": torch.sign(y_label)}
    model_kwargs["y2"][model_kwargs["y2"] <= 0] = torch.Tensor([-1]).cuda()
    model_kwargs["y2"] = model_kwargs["y2"].cuda()
    # TODO: III. define condition
    import torch.nn.functional as F


    def condition_2(x1,x2, y=(model_kwargs["y2"])):
        def dice(predict, target):
            assert predict.size() == target.size(), "the size of predict and target must be equal."
            num = predict.size(0)
            pre = predict.view(num, -1)
            tar = target.view(num, -1)
            intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
            union = (pre + tar).sum(-1).sum()
            score = 2 * (intersection + 1e-8) / (union + 1e-8)
            return score

        sig_x_2 = torch.nn.functional.log_softmax(x2 / scale_tau)[(label1*0).long()]
        sig_x_2 = sig_x_2.mean()
        sig_x_1 = F.sigmoid(x1 / scale_tau)[y.mean(1, keepdim=True) > 0]
        sig_value = torch.log(sig_x_1 + 1e-5).mean()
        dice_value = dice((x1 / scale_tau).sigmoid(), (y.mean(1, keepdim=True) > 0).float())
        print("stage 2:",dice_value,sig_x_2)
        return sig_value * 0 + dice_value + sig_x_2


    # TODO: IV define classifier
    classifier_2 = lambda x, t, cond: cond(*classifier_fn(x, t))

    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="noise",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
        guidance_type="classifier",
        condition=condition_2,
        guidance_scale=guidance_scale,
        classifier_fn=classifier_2,
        classifier_kwargs=classifier_kwargs,
    )
    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++", )
    # correcting_x0_fn="dynamic_thresholding")
    with torch.no_grad():
        x_image = dpm_solver.sample(
            torch.randn_like(x_T).cuda(),
            steps=100,
            order=3,
            skip_type="time_uniform",
            method="multistep",
        )

    from PIL import Image


    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images


    for i in range(x_image.shape[0]):
        sub_image, sub_label = x_image[i], y_label[i]
        sub_image = sub_image * torch.Tensor([0.229, 0.224, 0.225])[:, None, None].cuda() \
                    + torch.Tensor([0.485, 0.456, 0.406])[:, None, None].cuda()
        sub_image = sub_image.clamp(0, 1)
        sub_image = sub_image.cpu().permute(1, 2, 0).numpy()
        sub_label = ((sub_label / 2 + 0.5).mean(0, keepdim=True).clamp(0, 1) > 0.5).float()
        sub_label = sub_label.cpu().permute(1, 2, 0).numpy()
        numpy_to_pil(sub_image)[0].save(f"./isic_test/{j + i}_image.png")
        numpy_to_pil(sub_label)[0].save(f"./isic_test/{j + i}_mask.png")
