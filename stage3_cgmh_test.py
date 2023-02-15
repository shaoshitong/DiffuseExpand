import copy

from utils import NoiseScheduleVP, model_wrapper, DPM_Solver
import argparse
from torch.utils.data import DataLoader
from utils import create_model_and_diffusion
import torch

parser = argparse.ArgumentParser(description='Finetune Diffusion Model')
parser.add_argument('--dataset', type=str, default='CGMH', help='dataset')
parser.add_argument('--loss_type', type=str, default='mse', help='loss type')
parser.add_argument('--learn_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training networks')
parser.add_argument('--save_path', type=str, default="./stage4")
parser.add_argument('--class_cond', type=bool, default=True)
parser.add_argument('--num_classes_1', type=int, default=2)
parser.add_argument('--num_classes_2', type=int, default=-1)
parser.add_argument('--scale_tau', type=float, default=1.)
parser.add_argument('--guidance_scale', type=float, default=2.)
parser.add_argument('--cuda_devices', type=str, default="0", help="data parallel training")
parser2 = copy.deepcopy(parser)


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
        isic=False,
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

model_path = "/home/project/Medical-Seg-Dataset-Distillation/stage2_cgmh/model_stage2_cgmh_30000.pt"
if not os.path.exists(model_path):
    raise KeyError

_model_fn.load_state_dict(torch.load(model_path, map_location="cpu"))
_model_fn.cuda()


def grad_estlimate(y, tau, label):
    b = y.shape[0]
    sig_y = torch.sigmoid(y / tau).view(b, -1)
    label = label.view(b, -1).bool()
    soft_y = torch.stack([sig_y, 1 - sig_y], -1)
    soft_label = torch.stack([label, ~label], -1)
    learning_rate = (1 - soft_y[soft_label]) / tau
    return learning_rate.mean(-1)


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
        isic=False
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
    "isic"
]
# TODO: Define UNet and diffusion scheduler
args.num_classes_2 = 1
classifier_fn, _ = create_classifier_and_diffusion(
    **args_to_dict(args_2, NAME)
)

model_path_2 = "/home/project/Medical-Seg-Dataset-Distillation/stage2/stage3_cgmh_model_10000.pt"
if not os.path.exists(model_path_2):
    raise KeyError

model_params = torch.load(model_path_2, map_location="cpu")
classifier_fn.load_state_dict(model_params)
classifier_fn = classifier_fn.cuda()

# TODO: VII. define classifier_kwargs
classifier_kwargs = {}  # Nothing to do with uncond scenarios

# TODO: VIII. define betas
from utils import get_named_beta_schedule

# TODO: V. define guidance_scale
scale_tau = 1. / args.scale_tau
guidance_scale = args.guidance_scale  # Nothing to do with uncond scenarios

save_path = os.path.join(args.save_path, f"tau_{scale_tau}_scale_{guidance_scale}")
if not os.path.exists(save_path):
    os.makedirs(save_path)

betas = torch.from_numpy(get_named_beta_schedule("linear", 1000)).cuda()

noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
image_shape = (BATCHSIZE, 1, 256, 256)

for j in range(0, 50000, args.batch_size):
    label2 = None
    model_kwargs = {"y1": label1, "y2": label2}


    def condition_1(x1, x2, y=(model_kwargs["y2"])):
        sig_x = torch.nn.functional.log_softmax(x2 / scale_tau, 1)[range(BATCHSIZE), label1].sum()
        sig_x = sig_x
        return sig_x


    classifier_1 = lambda x, t, cond: cond(*classifier_fn(x, t))

    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="noise",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
        guidance_type="classifier",
        condition=condition_1,
        guidance_scale=guidance_scale,
        classifier_fn=classifier_1,
        classifier_kwargs={},
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
    model_kwargs = {"y1": torch.cat([label1,label1],0) * 0, "y2": torch.sign(y_label)}
    model_kwargs["y2"][model_kwargs["y2"] <= 0] = torch.Tensor([0]).cuda()
    model_kwargs["y2"] = torch.cat([model_kwargs["y2"].cuda(),model_kwargs["y2"].cuda()],0)
    # TODO: III. define condition
    import torch.nn.functional as F

    times = 30
    lrs = torch.linspace(1, 0.5, times)
    num_iter = 0


    def condition_2(x1, x2, y=(model_kwargs["y2"][0:model_kwargs["y2"].shape[0]//2])):
        def dice(pred, mask):
            weit = 1 + mask * 10  # torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
            wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
            wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

            pred = torch.sigmoid(pred)
            inter = ((pred * mask) * weit).sum(dim=(2, 3))
            union = ((pred + mask) * weit).sum(dim=(2, 3))
            wiou = (inter + 1) / (union - inter + 1)
            return wiou.sum(), - wbce.sum()

        global num_iter
        sig_x_2 = torch.nn.functional.log_softmax(x2 / scale_tau, 1)[range(BATCHSIZE), (label1 * 0).long()].sum()
        dice_value1, dice_value2 = dice((x1 / scale_tau), (y.mean(1, keepdim=True) > 0).float())
        print("stage 2:", dice_value1, dice_value2)
        learning_rate = lrs[num_iter]
        num_iter += 1
        return (dice_value1 + dice_value2 + sig_x_2) * learning_rate.detach()


    # TODO: IV define classifier

    classifier_2 = lambda x, t, cond: cond(*classifier_fn(x, t))

    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="noise",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
        guidance_type="classifier-free",
        condition=condition_2,
        guidance_scale=guidance_scale,
        classifier_fn=classifier_2,
        classifier_kwargs=classifier_kwargs,
    )
    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                            correcting_x0_fn="dynamic_thresholding")
    with torch.no_grad():
        x_image = dpm_solver.sample(
            torch.randn_like(x_T).cuda(),
            steps=times,
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


    from PIL import Image
    import torchvision
    import numpy as np
    from utils.vis_utils import vis_trun

    turn = torchvision.transforms.ToPILImage()
    for i in range(x_image.shape[0]):
        sub_image, sub_label = x_image[i], y_label[i]
        sub_image = (sub_image / 2 + 0.5).clamp(0, 1).float().cpu()
        sub_label = ((sub_label / 2 + 0.5).clamp(0, 1) > 0.5).float().cpu()
        # sem_image = vis_trun(sub_image.expand(3,-1,-1).numpy(),sub_label.numpy()).transpose((1,2,0))
        # sem_image = Image.fromarray(sem_image)
        sub_image = turn(sub_image)
        sub_label = turn(sub_label)
        # sem_image.save(f"./cgmh_test/sem_{j + i}.png")
        image_path = os.path.join(save_path, f"image_{j + i}.png")
        label_path = os.path.join(save_path, f"mask_{j + i}.png")
        sub_image.save(image_path)
        sub_label.save(label_path)
