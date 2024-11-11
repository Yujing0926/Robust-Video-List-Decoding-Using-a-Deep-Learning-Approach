import os
import torch
import numpy as np
import random
import cv2

from torchvision import transforms
from models.maniqa import MANIQA
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import ToTensor, Normalize
from tqdm import tqdm
from argparse import ArgumentParser

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Image(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, num_crops=20):
        super(Image, self).__init__()
        self.img_name = image_path.split('/')[-1]
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape
        # print(self.img.shape)
        new_h = 224
        new_w = 224

        self.img_patches = []
        for i in range(num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)

        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Super_patch(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, patch_size, sp_number):
        super(Super_patch, self).__init__()
        self.img_name = image_path.split('/')[-1]
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape
        # print(self.img.shape)
        self.patch_size = patch_size
        self.sp_number = sp_number
        new_h = self.patch_size * self.sp_number
        new_w = self.patch_size * self.sp_number

        M = int(h / self.patch_size - (self.sp_number - 1))
        N = int(w / self.patch_size - (self.sp_number - 1))

        self.img_patches = []
        for m in range(0, M):
            for n in range(0, N):
                # top = np.random.randint(0, h - new_h)
                # left = np.random.randint(0, w - new_w)
                top = m * self.patch_size
                left = n * self.patch_size
                patch = self.img[:, top: top + new_h, left: left + new_w]
                self.img_patches.append(patch)

        self.img_patches = np.array(self.img_patches)
        # print(self.img_patches.shape)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)
    parser = ArgumentParser(description='Test MANIQA')
    parser.add_argument("--img_path", type=str,
                        default='data/c38_0.png')
    args = parser.parse_args()

    # config file
    config = Config({
        # image path
        "image_path": args.img_path,

        # valid times
        "num_crops": 1404,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,

        # checkpoint path
        "ckpt_path": "./output/models/Our_data_f1f2_inter/our_data_f1f2-base_s20/epoch1.pt",
    })

    # data load
    # Img = Image(image_path=config.image_path,
    #     transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
    #     num_crops=config.num_crops)

    Img = Super_patch(image_path=config.image_path,
                      transform=transforms.Compose([ToTensor()]), patch_size=32, sp_number=7)

    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
                 patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
                 depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    net.load_state_dict(torch.load(config.ckpt_path), strict=False)
    net = net.cuda()

    avg_score = 0
    for i in tqdm(range(config.num_crops)):
    # for i in tqdm(range(1404)):
        with torch.no_grad():
            net.eval()
            patch_sample = Img.get_patch(i)
            patch = patch_sample['d_img_org'].cuda()
            patch = patch.unsqueeze(0)
            score = net(patch)
            # print("score:%s\n", score)
            avg_score += score

    # print("Image {} score: {}".format(Img.img_name, avg_score / config.num_crops))
    print("{} {}".format(Img.img_name, (avg_score / config.num_crops).cpu().detach().numpy()[0]))