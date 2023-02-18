import numpy as np
from PIL import Image


def label_to_onehot(label, colormap):
    """
    Converts a segmentation label (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in colormap:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def onehot_to_label(semantic_map, colormap):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(semantic_map, axis=-1)
    colour_codes = np.array(colormap)
    label = np.uint8(colour_codes[x.astype(np.uint8)])
    return label


def onehot2mask(semantic_map):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(semantic_map, axis=0).astype(np.uint8)
    return _mask


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    semantic_map = [mask == i for i in range(num_classes)]
    return np.array(semantic_map).astype(np.uint8)


def vis_trun(image, mask, weight=0.3):
    """
    :param image:  shape [3,H,W]
    :param mask:   shape [H,W] or [1,H,W]
    :param weight:
    :return:
    """
    assert image.ndim == 3 and image.shape[0] == 3 and (mask.ndim == 2 or mask.ndim == 3)
    if mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape[-1] == 1:
        mask = mask[..., 0]
    semantic_map = mask2onehot(mask, 2)
    color = np.array([106, 206, 235])[:, None, None]  # 3,1,1
    color_a = semantic_map[1][None, ...].astype(np.float) * color.astype(np.float)
    color_b = (image * 255).astype(np.uint8).astype(np.float)
    color_c = color_a * mask * weight + color_b * (1 - mask)
    color_c = color_c + mask * (1 - weight) * color_b
    color_c = color_c.astype(np.uint8)
    return color_c
