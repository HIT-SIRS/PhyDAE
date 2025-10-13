import os
import cv2
import glob
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Resize, InterpolationMode

from data.degradation_utils import Degradation
from utils.image_utils import random_augmentation, crop_img


class CDD11(Dataset):
    def __init__(self, args, split: str = "train", subset: str = "all"):
        super(CDD11, self).__init__()
        """
            特点：
            1. 处理预定义的退化图像对（清晰图像 + 对应的退化图像）
            2. 支持单一、双重、三重退化类型的组合
            3. 数据集结构固定，文件夹结构预定义
            4. 主要用于评估和测试特定的退化恢复任务
        """
        self.args = args
        self.toTensor = ToTensor()
        self.de_type = self.args.de_type
        self.dataset_split = split # 数据集划分：train/test/val
        self.subset = subset # 退化类型子集：single/double/triple/all/specific_degradation_name

        # 根据数据集划分设置patch大小
        if split == "train":
            self.patch_size = args.patch_size
        else:
            self.patch_size = 64 # 测试时使用固定大小

        self._init() # 初始化数据集路径和退化类型字典

    def __getitem__(self, index):
        # 训练时随机选择退化类型，测试时使用指定子集
        if self.dataset_split == "train":
            degradation_type = random.choice(list(self.degraded_dict.keys()))
            degraded_image_path = random.choice(self.degraded_dict[degradation_type])
        else:
            degradation_type = self.subset
            degraded_image_path = self.degraded_dict[degradation_type][index]

        degraded_name = os.path.basename(degraded_image_path)

        # 通过文件名匹配找到对应的清晰图像
        image_name = os.path.basename(degraded_image_path)
        assert degraded_name == image_name
        clean_image_path = os.path.join(os.path.dirname(self.clean[0]), image_name)

        # 加载图像
        lr = np.array(Image.open(degraded_image_path).convert('RGB')) # 低质量（退化）图像
        hr = np.array(Image.open(clean_image_path).convert('RGB')) # 高质量（清晰）图像

        # 训练时进行随机增强和裁剪
        if self.dataset_split == "train":
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))

        # 转换为张量
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        return [clean_image_path, degradation_type], lr, hr

    def __len__(self):
        return sum(len(images) for images in self.degraded_dict.values())

    def _init(self):
        """初始化数据集路径和退化类型字典"""
        data_dir = os.path.join(self.args.data_file_dir, "")
        # data_dir = os.path.join(self.args.data_file_dir, "cdd11")
        self.clean = sorted(glob.glob(os.path.join(data_dir, f"{self.dataset_split}/clear", "*.png")))

        if len(self.clean) == 0:
            raise ValueError(f"No clean images found in {os.path.join(data_dir, f'{self.dataset_split}/clear')}")

        # 构建退化类型字典
        self.degraded_dict = {}
        allowed_degradation_folders = self._filter_degradation_folders(data_dir)
        for folder in allowed_degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            degraded_images = sorted(glob.glob(os.path.join(folder, "*.png")))
            
            if len(degraded_images) == 0:
                raise ValueError(f"No images found in {folder_name}")

            # 训练时扩大数据集规模
            # scale dataset length
            if self.dataset_split == "train":
                degraded_images *= 2
            
            self.degraded_dict[folder_name] = degraded_images

    def _filter_degradation_folders(self, data_dir):
        """
        根据subset参数过滤退化文件夹
        支持：'single', 'double', 'triple', 'all' 或具体的退化类型名称
        """
        degradation_folders = sorted(glob.glob(os.path.join(data_dir, self.dataset_split, "*/")))
        filtered_folders = [] 

        for folder in degradation_folders:
            folder_name = os.path.basename(folder.strip('/'))
            if folder_name == "clear":
                continue

            # 通过下划线数量判断退化类型数量
            degradation_count = folder_name.count('_') + 1

            # 根据subset参数进行过滤
            if self.subset == "single" and degradation_count == 1:
                filtered_folders.append(folder)
            elif self.subset == "double" and degradation_count == 2:
                filtered_folders.append(folder)
            elif self.subset == "triple" and degradation_count == 3:
                filtered_folders.append(folder)
            elif self.subset == "all":
                filtered_folders.append(folder)
            # 如果subset是具体的退化文件夹名称，精确匹配
            elif self.subset not in ["single", "double", "triple", "all"]:
                if folder_name == self.subset:
                    filtered_folders.append(folder)

        print(f"Degradation type mode: {self.subset}")
        print(f"Loading degradation folders: {[os.path.basename(f.strip('/')) for f in filtered_folders]}")
        return filtered_folders

    def _crop_patch(self, img_1, img_2):
        """在相同位置裁剪两张图像的patch"""
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2
    
    
        
class AIOTrainDataset(Dataset):
    """
    AIO (All-In-One) 训练数据集类 - 多任务联合训练数据集

    特点：
    1. 整合多种图像恢复任务（去噪、去模糊、去雨、去雾、低光增强）
    2. 支持在线退化生成（特别是去噪任务）
    3. 通过重复采样平衡不同任务的数据量
    4. 每个样本都有退化类型标识，支持多任务学习
    """
    def __init__(self, args, split: str = "train"):
        super(AIOTrainDataset, self).__init__()
        self.args = args
        self.split = split
        self.de_temp = 0
        self.de_type = self.args.de_type # 支持的退化类型列表
        self.D = Degradation(args)  # 退化生成器（用于在线生成退化）

        # 退化类型到ID的映射
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        self.de_dict_reverse = {idx: dataset for idx, dataset in enumerate(self.de_type)}

        # 图像预处理变换
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

        self._init_lr() # 初始化各个任务的数据路径
        self._merge_tasks() # 合并所有任务的数据
            
    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        deg_type = self.de_dict_reverse[de_id]

        hr_sample = self.hr[idx]
        lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
        hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)
        
        lr, hr = random_augmentation(*self._crop_patch(lr, hr))
            
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        
        return [lr_sample["img"], de_id], lr, hr
        
    
    def __len__(self):
        return len(self.lr)
    
    
    def _init_lr(self):
        # synthetic datasets
        # 根据配置的退化类型初始化相应的数据集
        if 'dedark' in self.de_type:
            self._init_dedark(id=self.de_dict['dedark'])
        if 'deblur' in self.de_type:
            self._init_deblur(id=self.de_dict['deblur'])
        if 'denoise' in self.de_type:
            self._init_denoise(id=self.de_dict['denoise'])
        if 'dehaze' in self.de_type:
            self._init_dehaze(id=self.de_dict['dehaze'])

    def _merge_tasks(self):
        """合并所有任务的数据，实现多任务联合训练"""
        self.lr = []
        self.hr = []
        # synthetic datasets
        if "denoise" in self.de_type:
            self.lr += self.denoise_lr
            self.hr += self.denoise_hr
        if "deblur" in self.de_type:
            self.lr += self.deblur_lr 
            self.hr += self.deblur_hr
        if "dedark" in self.de_type:
            self.lr += self.dedark_lr
            self.hr += self.dedark_hr
        if "dehaze" in self.de_type:
            self.lr += self.dehaze_lr 
            self.hr += self.dehaze_hr

        print(len(self.lr))

    def _init_deblur(self, id):
        inputs = self.args.data_file_dir + f"/{self.split}/blur/"
        targets = self.args.data_file_dir + f"/{self.split}/clean/"
        
        self.deblur_lr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.deblur_hr = [{"img" : x, "de_type":id} for x in sorted(glob.glob(targets + "/*.png"))]

        print("Total Deblur {} pairs : {}".format(self.split, len(self.deblur_lr)))

    def _init_dedark(self, id):
        inputs = self.args.data_file_dir + f"/{self.split}/dark/"
        targets = self.args.data_file_dir + f"/{self.split}/clean/"

        self.dedark_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.dedark_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        print("Total Dedark {} pairs : {}".format(self.split, len(self.dedark_lr)))

    def _init_dehaze(self, id):
        inputs = self.args.data_file_dir + f"/{self.split}/haze/"
        targets = self.args.data_file_dir + f"/{self.split}/clean/"

        self.dehaze_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.dehaze_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        print("Total Dehaze {} pairs : {}".format(self.split, len(self.dehaze_lr)))

    def _init_denoise(self, id):
        inputs = self.args.data_file_dir + f"/{self.split}/noise/"
        targets = self.args.data_file_dir + f"/{self.split}/clean/"

        self.denoise_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.denoise_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        print("Total Denoise {} pairs : {}".format(self.split, len(self.denoise_lr)))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    
class IRBenchmarks(Dataset):
    def __init__(self, args):
        super(IRBenchmarks, self).__init__()

        """
            IR (Image Restoration) 基准测试数据集类 - 专门用于模型评估

            特点：
            1. 整合多个标准基准测试数据集
            2. 统一的测试接口和数据格式
            3. 支持在线噪声生成（去噪任务）
            4. 主要用于模型性能评估和比较
        """

        self.args = args
        self.benchmarks = args.benchmarks # 要使用的基准数据集列表
        self.de_type = self.args.de_type
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        
        self.toTensor = ToTensor()
        
        self.resize = Resize(size=(256, 256), interpolation=InterpolationMode.NEAREST)
        
        self._init_lr() # 根据指定的基准数据集进行初始化
        
    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]

        # 其他任务：直接加载预存的图像对
        hr_sample = self.hr[idx]
        lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
        hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)
            
        lr = self.toTensor(lr)
        hr = self.toTensor(hr)
        return [lr_sample["img"], de_id], lr, hr
    
    def __len__(self):
        return len(self.lr)
    
    def _init_lr(self):
        """根据指定的基准数据集进行初始化"""

        if 'denoise' in self.benchmarks:
            self._init_denoise(id=self.de_dict['denoise'])
        if 'dehaze' in self.benchmarks:
            self._init_dehaze(id=self.de_dict['dehaze'])
        if 'dedark' in self.benchmarks:
            self._init_dedark(id=self.de_dict['dedark'])
        if 'deblur' in self.benchmarks:
            self._init_deblur(id=self.de_dict['deblur'])

    ####################################################################################################
    def _init_deblur(self, id):
        inputs = self.args.data_file_dir + "/test/blur/"
        targets = self.args.data_file_dir + "/test/clean/"

        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        print("Total Deblur training pairs : {}".format(len(self.lr)))

    def _init_dedark(self, id):
        inputs = self.args.data_file_dir + "/test/dark/"
        targets = self.args.data_file_dir + "/test/clean/"

        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        print("Total Dedark training pairs : {}".format(len(self.lr)))

    def _init_dehaze(self, id):
        inputs = self.args.data_file_dir + "/test/haze/"
        targets = self.args.data_file_dir + "/test/clean/"

        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        print("Total Dehaze training pairs : {}".format(len(self.lr)))

    def _init_denoise(self, id):
        inputs = self.args.data_file_dir + "/test/noise/"
        targets = self.args.data_file_dir + "/test/clean/"

        self.lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        print("Total Denoise training pairs : {}".format(len(self.lr)))