import os
import torch
from torchvision.utils import save_image

def save_feature_maps_as_png(features, save_dir, epoch):
    """
    features: dict, {layer_name: tensor[C,H,W]} 每层特征
    save_dir: 根目录
    epoch: 当前 epoch
    """
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    for name, feat in features.items():
        layer_dir = os.path.join(epoch_dir, name)
        os.makedirs(layer_dir, exist_ok=True)
        C = feat.shape[0]
        for c in range(C):
            channel_feat = feat[c:c+1]  # [1,H,W]
            # 归一化到0-1
            channel_feat = (channel_feat - channel_feat.min()) / (channel_feat.max() - channel_feat.min() + 1e-8)
            save_image(channel_feat, os.path.join(layer_dir, f"channel_{c}.png"))
