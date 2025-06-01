import torch
from torch import nn
from torchvision import models


def vit_model():
    model_weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    inference_model = models.vit_b_16(weights=model_weights)
    classes = model_weights.meta["categories"]
    return inference_model, classes, inference_model.image_size

