import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale

from PIL import Image
import random
import numpy as np

from utils.image_utils import crop_img


class Degradation(object):
    """
    图像退化处理类
    用于对图像添加各种退化效果，主要用于图像去噪任务的数据增强
    """
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

    def _add_gaussian_noise(self, clean_patch, sigma):
        """
              为图像添加高斯噪声

              Args:
                  clean_patch: 干净的图像块（numpy数组格式）
                  sigma: 高斯噪声的标准差，值越大噪声越强

              Returns:
                  noisy_patch: 添加噪声后的图像块
                  clean_patch: 原始干净的图像块
        """
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        """
             根据指定的降质类型对图像进行降质处理

             Args:
                 clean_patch: 干净的图像块
                 degrade_type: 降质类型编号
                     - 0: 添加sigma=15的高斯噪声（轻度噪声）
                     - 1: 添加sigma=25的高斯噪声（中度噪声）
                     - 2: 添加sigma=50的高斯噪声（重度噪声）

             Returns:
                 degraded_patch: 降质后的图像块
                 clean_patch: 原始干净的图像块
        """
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)
        else:
            raise NotImplementedError(f"Degradation type {degrade_type} not defined.")

        return degraded_patch, clean_patch

    def single_degrade(self,clean_patch,degrade_type = None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1
