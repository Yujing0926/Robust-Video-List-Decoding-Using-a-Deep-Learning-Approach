import os
# import cv2
import PIL
# import yaml
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import scipy.io as scio
from scipy import stats
from torch.optim import Adam
from torch.cuda import device
import torch.nn.functional as F
from argparse import ArgumentParser
from scipy.signal import convolve2d
from torch.optim.lr_scheduler import StepLR
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from ignite.metrics.metric import Metric
from torchvision.transforms.functional import to_tensor
# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader, random_split, TensorDataset


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Definition of the neural network
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
        return q


class VisualScorePredictor(nn.Module):
    def __init__(self, input_size=(64, 64), conv_output_size=(50, 58, 58), pool_kernel_size=2, num_classes=1):
        super(VisualScorePredictor, self).__init__()

        # Convolution layer: 50 filters of size 7x7
        self.conv1 = nn.Conv2d(1, 50, kernel_size=7)

        # Calculate the size after the convolution operation
        self.conv_output_size = conv_output_size

        # Pooling layer: Max pooling
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)

        # Calculate the size after the pooling operation
        self.pool_output_size = (self.conv_output_size[0],
                                 self.conv_output_size[1] // pool_kernel_size,
                                 self.conv_output_size[2] // pool_kernel_size)

        # Calculate the number of features to feed into the fully connected layer
        self.num_flat_features = self.pool_output_size[0] * self.pool_output_size[1] * self.pool_output_size[2]

        # Fully connected layers: 400 neurons each
        self.fc1 = nn.Linear(self.num_flat_features, 400)
        self.fc2 = nn.Linear(400, 400)

        # Output layer for linear regression
        self.output = nn.Linear(400, num_classes)

    def forward(self, x):
        # Convolution followed by ReLU and max pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Flattening the output for fully connected layers
        x = x.view(-1, self.num_flat_features)

        # Fully connected layers followed by ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.output(x)
        return x


# Function to calculate the output size of each layer
def calculate_conv_output_size(input_size, kernel_size, stride=1, padding=0):
    output_size = (input_size - kernel_size + 2 * padding) // stride + 1
    return output_size


def extract_patches(image_path, patch_size=64):
    # Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Verify that the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load image at path: {image_path}")
        return None

    # Get image dimensions
    height, width = image.shape

    # Extract patches
    patches = []
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)

    return patches


def default_loader(path, img_type, patch_size):
    if img_type == 'RGB':
        return Image.open(path)  # .convert('RGB')  # L: grey image; RGB: color image in RGB
    elif img_type == 'Y':
        f = open(path, 'rb')
        yuv_buf = f.read()
        yuv_np = np.frombuffer(yuv_buf, dtype=np.ubyte)
        yuv_re = yuv_np.reshape((patch_size, patch_size)).transpose((1, 0)).transpose((1, 0))
        im = PIL.Image.fromarray(yuv_re, mode="L")
        return im
    elif img_type == 'YUV':
        f = open(path, 'rb')
        yuv_buf = f.read()
        yuv_np = np.frombuffer(yuv_buf, dtype=np.ubyte)
        yuv_re = yuv_np.reshape((3, patch_size, patch_size)).transpose((2, 1, 0)).transpose((1, 0, 2))
        im = PIL.Image.fromarray(yuv_re, mode="YCbCr")
        return im


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)  # Uniform kernel to average values over the window
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C  # Using E([X-E[X])^2]=E[X^2] - E[X]^2
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)  # Local normalization
    # Number of unique values is 1, meaning variance is 0
    # if len(np.unique(patch_mean)) == 1:
    #     if np.all(patch_mean == 0):
    #         # Case all values are zero, meaning erroneous patch
    #         if torch.equal(patch_ln.cpu(), torch.zeros(patch_ln.shape)):
    #             patch_ln = patch_ln
    #     elif not np.all(patch_mean == 0):
    #         # Case all values are not zero, meaning a uniform patch
    #         if torch.allclose(patch_ln.cpu(), torch.zeros(patch_ln.shape)):
    #             patch_ln = torch.add(patch_ln, -0.013)

        # tensor_max = torch.max(patch_ln)
        # tensor_eq = torch.eq(patch_ln,tensor_max)
        # if torch.equal(tensor_eq.cpu(), torch.zeros(tensor_eq.shape)):
        #     patch_ln = torch.add(patch_ln, 0.5)
        # else:
        #     patch_ln = patch_ln
    return patch_ln


def NonOverlappingCropPatches(im, channel=3):
    w, h = im.size
    patches = ()
    # patches = [(),(),()]
    # for i in range(0, h - stride, stride):
    #     for j in range(0, w - stride, stride):
    for i in range(0, 1):
        for j in range(0, 1):
            # patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patch = to_tensor(im)
            if channel == 1:
                patch = LocalNormalization(patch[0].numpy())
            elif channel == 3:
                patch1 = LocalNormalization(patch[0].numpy()).numpy()
                patch2 = LocalNormalization(patch[1].numpy()).numpy()
                patch3 = LocalNormalization(patch[2].numpy()).numpy()
                t = np.array([patch1[0], patch2[0], patch3[0]])
                # patch = torch.from_numpy(t)
                patch = t
            patches = patches + (patch,)
    return patches


def load_psnr_for_patches(patch_route, patch_score_route, patch_size,
                          reference_metric, frame_type, QP, img_type, APPEND_CHAR):
    patch_pairs = []
    psnr_values = []
    psnr_values_in = []

    psnr_route = patch_score_route + APPEND_CHAR + 'fr_scores_patch' + str(patch_size) \
                 + '_' + reference_metric + '_' + frame_type + '_' + QP + '_rgb_pattern.xlsx'
    df_psnr = pd.read_excel(psnr_route)
    im_data = pd.DataFrame(df_psnr, columns=['im_names', 'subjective_scores_nz'])
    im_name = im_data['im_names']
    im_name = im_name.values.tolist()
    psnr_score = im_data['subjective_scores_nz']
    psnr_score = psnr_score.values.tolist()

    # for i in range(0, 1):
    for i in range(0, len(im_name)):
        patch_1 = default_loader(patch_route + im_name[i], img_type, patch_size)
        psnr_values.append(psnr_score[i])

        vn = im_name[i].split("_", 1)
        vn_n = vn[1].split("_", 3)
        if 'c' in vn_n[0]:
            vnnn = vn_n[3].split(".", 1)
            vn_n_0 = vn_n[0]
            vn_n_inatct = 'i' + vn_n_0[1:]
            # print(vn_n_inatct)
            patch_2_name = vn[0] + '_' + vn_n_inatct + '_' + vn_n[2] + '_' + vnnn[0] + '.png'
            patch_2 = default_loader(patch_route + patch_2_name, img_type, patch_size)

            patch_c = np.array(patch_1)
            patch_i = np.array(patch_2)
            psnr_intact = psnr(patch_c, patch_i)
            psnr_values_in.append(psnr_intact)

            patch_1 = NonOverlappingCropPatches(patch_1, channel=3)
            patch_2 = NonOverlappingCropPatches(patch_2, channel=3)
            patch_pairs.append((patch_1, patch_2))

        elif 'i' in vn_n[0]:
            psnr_values_in.append(50)
            patch_1 = NonOverlappingCropPatches(patch_1, channel=3)
            patch_pairs.append((patch_1, patch_1))

    return patch_pairs, psnr_values, psnr_values_in


# Custom loss function
def custom_loss(y_pred, y_theoric, y_intact, y_th_in):
    # We ensure that all predictions are of float type for calculations
    y_pred = y_pred.float()  # Predicted score by CNN for reconstructed patch
    y_theoric = y_theoric.float()  # Groud-truth reference score for reconstructed patch
    y_intact = y_intact.float()  # Predicted score by CNN for corresponding intact patch
    epsilon = 0.05  # parameter to modify and see the impact
    threshold = 50
    # y_theoric = y_theoric.unsqueeze(1)

    # Loss where the prediction is greater than the theoretical value (F1)
    # b = y_pred - y_theoric
    # b_abs = torch.abs(b)
    # c = torch.sub(y_pred, y_theoric)
    # c_abs = torch.abs(c)
    loss_theoric = torch.abs(y_pred - y_theoric)
    loss_l1 = F.l1_loss(y_pred, y_theoric)

    # Loss where prediction is greater than intact value (F2)
    loss_intact = torch.max(torch.zeros(1), (y_pred - (y_intact - epsilon)))

    # Combination of the two losses with conditions
    # y_th_in is the full-reference quality measure between the reconstructed patch and corresponding intact patch
    combined_loss = torch.clone(loss_theoric)
    for i in range(0, len(y_th_in)):
        if y_th_in[i] < threshold:
            combined_loss[i] = 0*combined_loss[i] + 1*loss_intact[i]  # coefficients to modify
            # print("Position: {}, combined_loss: {}, psnr_th_in: {}".format(i, combined_loss[i], y_th_in[i]))

    # We could possibly average over the samples if necessary
    return combined_loss.mean()
    # a = loss_theoric.mean()
    # print("My Loss: {}, loss_l1: {}, loss_intact:{}".format(a, loss_l1, loss_intact.mean()))
    # return loss_theoric.mean()


def compute(output, label):
    sq = np.asarray(label)
    q = np.asarray(output)
    # print(sq)
    # print(q)

    srocc = stats.spearmanr(sq, q)[0]
    krocc = stats.kendalltau(sq, q)[0]
    plcc = stats.pearsonr(sq, q)[0]
    rmse = np.sqrt(((sq - q) ** 2).mean())
    mae = np.abs((sq - q)).mean()
    outlier_ratio = (np.abs(sq - q) > 2 * 1).mean()
    if np.isnan(srocc):
        srocc = 0
        krocc = 0
        plcc = 0

    return srocc, krocc, plcc, rmse, mae, outlier_ratio


if __name__ == "__main__":
    print(f"Torch: {torch.__version__}")

    # Chose your parameters, your route and your system
    parser = ArgumentParser(description='PyTorch CNNIQA')
    parser.add_argument('--system', type=str, default='Windows')
    parser.add_argument('--main_route', type=str, default='.',
                        help='input your main route for the .xlsx and .txt file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.00003)')
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--gamma", type=int, default=0.7)

    args = parser.parse_args()

    system = args.system
    if system == 'Windows':
        APPEND_CHAR = '\\'
    elif system == 'Linux':
        APPEND_CHAR = '/'
    elif system == 'Mac':
        APPEND_CHAR = '/'

    QP = 'qp37'
    codec = 'HEVC'
    img_type = 'RGB'
    reference_metric = 'psnr'
    error_frame = 2

    patch_size = 64
    image_width = 1920
    image_height = 1024

    if error_frame == 1:
        frame_type = 'intra'
    else:
        frame_type = 'inter'

    # main_route = APPEND_CHAR + 'home' + APPEND_CHAR + 'ar28470@ens.ad.etsmtl.ca' + APPEND_CHAR + 'Documents'
    # main_route = APPEND_CHAR + 'home' + APPEND_CHAR + 'ar28470' + APPEND_CHAR + 'Documents'

    # For Github codes
    # main_route = args.main_route + APPEND_CHAR
    # train_database_route = main_route + APPEND_CHAR + 'data'
    # patch_score_route = train_database_route

    # main_route = 'E:\doctor\codes' + APPEND_CHAR
    main_route = 'D:' + APPEND_CHAR
    train_database_route = main_route + APPEND_CHAR + '1024p_database_patchs' + str(patch_size) \
                           + '_train_' + frame_type
    patch_route = train_database_route + APPEND_CHAR + QP + '_patch' + str(patch_size) + APPEND_CHAR

    patch_score_route = train_database_route + APPEND_CHAR + 'patch' + str(patch_size) + '_' \
                        + reference_metric + '_' + QP + APPEND_CHAR
    if not (os.path.exists(patch_score_route)):
        os.mkdir(patch_score_route)

    # Extract pair patches
    patch_pairs, psnr_values, psnr_values_in = load_psnr_for_patches(patch_route, patch_score_route, patch_size,
                                                                     reference_metric, frame_type, QP, img_type,
                                                                     APPEND_CHAR)

    # Prepare your model, optimizer, etc.
    model = CNNIQAnet(ker_size=7,
                      n_kers=50,
                      n1_nodes=800,
                      n2_nodes=800)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = args.epochs

    # If your images are grayscale and your model expects a single channel
    patch_pairs = np.asarray(patch_pairs)
    patch_tensors_stacked = torch.stack([torch.tensor(pair[0]) for pair in patch_pairs])
    patch_tensors_stacked2 = torch.stack([torch.tensor(pair[1]) for pair in patch_pairs])
    psnr_tensors = torch.tensor(psnr_values, dtype=torch.float32)  # You need to set psnr_values
    psnr_tensors_in = torch.tensor(psnr_values_in, dtype=torch.float32)

    dataset = TensorDataset(patch_tensors_stacked, patch_tensors_stacked2, psnr_tensors, psnr_tensors_in)

    # Dataset sizes
    total_size = len(dataset)
    train_size = int(0.6 * total_size)  # Choose the value you want for the train/val and test
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # Division of dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # train_dataset, val_dataset, test_dataset = [], [], []
    # trainindex = list(range(0, train_size))
    # testindex = list(range((total_size - test_size), total_size))
    # for i in range(len(psnr_tensors)):
    #     train_dataset.append(dataset[i]) if (i in trainindex) else \
    #         test_dataset.append(dataset[i]) if (i in testindex) else \
    #             val_dataset.append(dataset[i])
    print("Train Images: {}".format(len(train_dataset)))
    print("Validation Images: {}".format(len(val_dataset)))
    print("Test Images: {}".format(len(test_dataset)))

    # Creation of DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # print(train_loader)

    # Training loop with validation for each epoch we will calculate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir('results')
    database_name = '1024pDatabase'
    save_result_file = 'results' + APPEND_CHAR + 'CNNIQA-2loss-' + database_name + '-patch' + str(patch_size) \
                       + '-' + reference_metric + '-' + QP + '-' + frame_type + '-rgb-1score-pattern-f1f2'
    ensure_dir('checkpoints')
    trained_model_file = 'checkpoints' + APPEND_CHAR + 'CNNIQA-2loss-' + database_name + '-patch' + str(patch_size) \
                         + '-' + reference_metric + '-' + QP + '-' + frame_type + '-rgb-1score-pattern-f1f2'
    global best_criterion
    global best_epoch
    best_criterion = 1

    for epoch in range(num_epochs):
        model.train()  # Training mode
        train_loss = 0.0
        for data in train_loader:
            patch_corrupted, patch_intact, psnr_theoric, psnr_theoric_in = data
            patch_corrupted, patch_intact, psnr_theoric, psnr_theoric_in = patch_corrupted.to(device), \
                patch_intact.to(device), psnr_theoric.to(device), psnr_theoric_in.to(device)

            optimizer.zero_grad()

            y_pred_corrupted = model(patch_corrupted)  # we calculate the prediction for the corrupt or intact patch
            y_pred_intact = model(patch_intact)  # we calculate the prediction of the intact patch

            # patch_c_np = patch_corrupted.numpy()
            # patch_i_np = patch_intact.numpy()
            # psnr_intact = psnr_theoric
            # for i in range(0, 1):
            #     psnr_intact[i] = psnr(patch_c_np[i], patch_i_np[i])
            # psnr_intact = psnr([p for p in patch_c_np], [p for p in patch_i_np])

            # we apply the loss function
            psnr_theoric = psnr_theoric.unsqueeze(1)
            psnr_theoric_in = psnr_theoric_in.unsqueeze(1)
            loss = custom_loss(y_pred_corrupted, psnr_theoric, y_pred_intact, psnr_theoric_in)
            # print("My Loss: {}, y_pred: {}\n".format(loss, torch.transpose(y_pred_corrupted, 0, 1)))
            # print("My Loss: {}, y[0]: {}\n".format(loss, torch.transpose(psnr_theoric, 0, 1)))
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()  # Accumulate the loss

        # Calculating Average Training Loss
        # train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()  # Evaluation mode
        a = []
        b = []
        with torch.no_grad():  # No gradient calculation
            val_loss = 0.0
            for data in val_loader:
                patch_corrupted, patch_intact, psnr_theoric, psnr_theoric_in = data
                patch_corrupted, patch_intact, psnr_theoric, psnr_theoric_in = patch_corrupted.to(
                    device), patch_intact.to(
                    device), psnr_theoric.to(device), psnr_theoric_in.to(device)
                y_pred_corrupted = model(patch_corrupted)
                y_pred_intact = model(patch_intact)

                psnr_theoric = psnr_theoric.unsqueeze(1)
                psnr_theoric_in = psnr_theoric_in.unsqueeze(1)
                loss = custom_loss(y_pred_corrupted, psnr_theoric, y_pred_intact, psnr_theoric_in)
                val_loss = val_loss + loss.item()
                a.extend(np.asarray(y_pred_corrupted.squeeze(1)))
                b.extend(np.asarray(psnr_theoric.squeeze(1)))

            # Calculating the average validation loss
            # val_loss = val_loss / len(val_loader.dataset)

        srocc, krocc, plcc, rmse, mae, outlier_ratio = compute(a, b)

        # Displaying results
        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print("Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} "
              "RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
              .format(epoch + 1, srocc, krocc, plcc, rmse, mae, 100 * outlier_ratio))

        if val_loss < best_criterion:
            best_criterion = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), trained_model_file)

    print("Final Results - Epoch: {} Training Loss: {:.4f} Validation Loss: {:.4f} %"
          .format(best_epoch, train_loss, val_loss))
    np.save(save_result_file, (train_loss, val_loss))
