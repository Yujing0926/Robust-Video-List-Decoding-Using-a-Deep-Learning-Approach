from argparse import ArgumentParser
import PIL.Image
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from IQADataset import NonOverlappingCropPatches
import numpy as np


class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        # self.conv1 = nn.Conv2d(1, n_kers, ker_size)
        self.conv1 = nn.Conv2d(3, n_kers, ker_size)
        self.fc1 = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        self.fc3 = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #

        h = self.conv1(x)

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)  # max-min pooling
        h = h.squeeze(3).squeeze(2)

        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)
        # q = h
        return q


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch CNNIQA test demo')
    parser.add_argument("--im_path", type=str, default='patch_c47_2_16_15.png',
                        help="image path")
    parser.add_argument("--model_file", type=str,
                        default='checkpoints/CNNIQA-1024pDatabase-EXP0-lr=0.001-patch64-psnr-qp37-intra-rgb-pattern-2',
                        help="model file (default: models/CNNIQA-LIVE)")
    parser.add_argument("--patch_size", type=int, default=64,
                        help="patch_size")

    args = parser.parse_args()
    img_type = 'RGB'
    patch_size = args.patch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800).to(device)

    # model.load_state_dict(torch.load(args.model_file))
    model.load_state_dict(torch.load(args.model_file, map_location="cpu"))  # use CPU to test

    if img_type == 'RGB':
        im = Image.open(args.im_path)  # .convert('L')  # L: grey image; RGB: color image in RGB
    elif img_type == 'YUV':
        f = open(args.im_path, 'rb')
        yuv_buf = f.read()
        yuv_np = np.frombuffer(yuv_buf, dtype=np.ubyte)
        # 3 dimension image
        yuv_re = yuv_np.reshape((3, patch_size, patch_size)).transpose((2, 1, 0)).transpose((1, 0, 2))
        im = PIL.Image.fromarray(yuv_re, mode="YCbCr")
    elif img_type == 'Y':
        f = open(args.im_path, 'rb')
        yuv_buf = f.read()
        yuv_np = np.frombuffer(yuv_buf, dtype=np.ubyte)
        # 1 dimension image
        yuv_re = yuv_np.reshape((patch_size, patch_size)).transpose((1, 0)).transpose((1, 0))
        im = PIL.Image.fromarray(yuv_re, mode="L")
    patches = NonOverlappingCropPatches(im, patch_size, patch_size, channel=3)

    model.eval()
    with torch.no_grad():
        patch_scores = model(torch.stack(patches).to(device))
        # print(args.im_path, patch_scores.mean().item())
        print(args.im_path, patch_scores.item())


    ## Test for loading part of model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CNNIQAnet(ker_size=7,
    #                   n_kers=50,
    #                   n1_nodes=800,
    #                   n2_nodes=800).to(device)
    # pretext_model = torch.load(args.model_file, map_location=torch.device('cpu'))
    # model2_dict = model.state_dict()
    # state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
    # model2_dict.update(state_dict)
    # model.load_state_dict(model2_dict)
    #
    # im = Image.open(args.im_path)
    # patches = NonOverlappingCropPatches(im, 64, 64)
    # model.eval()
    # with torch.no_grad():
    #     patch_features = model(torch.stack(patches).to(device))
    #
    # print(np.shape(patch_features))
