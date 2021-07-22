import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import pickle
from util import get_image_to_tensor_balanced, get_mask_to_tensor
import random
from .FluidShakeDataset import FluidShakeDataset
from .WaterPourDataset import FluidPourDataset


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, stage='train', image_size=(128, 128), img_format='png', seed=10):
        super(MixedDataset, self).__init__()
        self.shake_dataset = FluidShakeDataset(stage=stage, image_size=image_size, img_format=img_format)
        self.pour_dataset = FluidPourDataset(stage=stage, image_size=image_size, img_format=img_format)
        self.dataset_size = self.shake_dataset.__len__() + self.pour_dataset.__len__()

        random.seed(seed)

        order = [i for i in range(self.dataset_size)]
        random.shuffle(order)

        self.order = order


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        item = self.order[item]
        if item < self.shake_dataset.__len__():
            return self.shake_dataset.__getitem__(item)
        else:
            return self.pour_dataset.__getitem__(item - self.shake_dataset.__len__())


