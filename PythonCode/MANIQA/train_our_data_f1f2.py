import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
import cv2
import pandas as pd

from torchvision import transforms
from models.maniqa import MANIQA
from config import Config
from utils.process import RandCrop, ToTensor, Normalize, five_point_crop
from utils.process import split_dataset_kadid10k, split_dataset_koniq10k
from utils.process import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader, random_split, TensorDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


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
    loss_theoric = torch.pow(torch.abs(y_pred - y_theoric), 2)
    loss_mse = F.mse_loss(y_pred, y_theoric)

    # Loss where prediction is greater than intact value (F2)
    loss_intact = torch.max(torch.zeros(1).cuda(), (y_pred - (y_intact - epsilon)))

    # Combination of the two losses with conditions
    # y_th_in is the full-reference quality measure between the reconstructed patch and corresponding intact patch
    combined_loss = torch.clone(loss_theoric)
    for i in range(0, len(y_intact)):
        if y_th_in[i] < threshold:
            combined_loss[i] = combined_loss[i] + loss_intact[i]  # coefficients to modify
            # print("Position: {}, combined_loss: {}, psnr_th_in: {}".format(i, combined_loss[i], y_th_in[i]))

    # We could possibly average over the samples if necessary
    return combined_loss.mean()
    # a = loss_theoric.mean()
    # # print("My Loss: {}, loss_l1: {}, loss_intact:{}".format(a, loss_l1, loss_intact.mean()))
    # return loss_theoric.mean()


def load_psnr_for_patches(psnr_file, dis_path):
    patch_pairs = []
    psnr_values = []
    psnr_values_in = []

    dis_files_data = []
    intact_files_data, intact_score_data = [], []
    with open(psnr_file, 'r') as listFile:
        for line in listFile:
            dis, score = line.split()
            score = float(score)
            dis_files_data.append(dis)
            psnr_values.append(score)

    # for i in range(0, 1):
    for i in range(0, len(dis_files_data)):
        patch_1 = cv2.imread(os.path.join(dis_path, dis_files_data[i]), cv2.IMREAD_COLOR)
        patch_1 = cv2.cvtColor(patch_1, cv2.COLOR_BGR2RGB)
        patch_1 = np.array(patch_1).astype('float32') / 255
        patch_1 = np.transpose(patch_1, (2, 0, 1))

        vn = dis_files_data[i].split("_", 1)
        vn_n = vn[1].split("_", 3)
        if 'c' in vn_n[0]:
            vnnn = vn_n[3].split(".", 1)
            vn_n_0 = vn_n[0]
            vn_n_inatct = 'i' + vn_n_0[1:]
            # print(vn_n_inatct)
            patch_2_name = vn[0] + '_' + vn_n_inatct + '_' + vn_n[2] + '_' + vnnn[0] + '.png'
            patch_2 = cv2.imread(os.path.join(dis_path, patch_2_name), cv2.IMREAD_COLOR)
            patch_2 = cv2.cvtColor(patch_2, cv2.COLOR_BGR2RGB)
            patch_2 = np.array(patch_2).astype('float32') / 255
            patch_2 = np.transpose(patch_2, (2, 0, 1))

            psnr_intact = psnr(patch_1, patch_2)
            psnr_values_in.append(psnr_intact)

            patch_pairs.append((patch_1, patch_2))

        elif 'i' in vn_n[0]:
            psnr_values_in.append(50)
            patch_pairs.append((patch_1, patch_1))

    return patch_pairs, psnr_values, psnr_values_in


def train_epoch(epoch, net, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for data in tqdm(train_loader):
        # x_d = data['d_img_org'].cuda()
        # labels = data['score']
        # x_i = data['d_img_in'].cuda()
        # labels_intact = data['intact_score']

        patch_corrupted, patch_intact, psnr_theoric, psnr_theoric_in = data
        # patch_corrupted, patch_intact = patch_corrupted.cuda(), patch_intact.cuda()
        patch_corrupted, patch_intact, psnr_theoric, psnr_theoric_in = patch_corrupted.to(device), patch_intact.to(
            device), psnr_theoric.to(device), psnr_theoric_in.to(device)
        pred_d = net(patch_corrupted.cuda())
        pred_i = net(patch_intact.cuda())
        # labels = torch.squeeze(psnr_theoric.type(torch.FloatTensor)).cuda()
        # labels_intact = torch.squeeze(psnr_theoric_in.type(torch.FloatTensor)).cuda()
        labels = torch.squeeze(psnr_theoric.type(torch.FloatTensor)).cuda()
        labels_intact = torch.squeeze(psnr_theoric_in.type(torch.FloatTensor)).cuda()

        optimizer.zero_grad()
        loss = custom_loss(pred_d, labels, pred_i, labels_intact)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for data in tqdm(test_loader):
            pred = 0
            pred_i = 0
            for i in range(config.num_avg_val):
                # x_d = data['d_img_org'].cuda()
                # labels = data['score']
                # labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                # x_i = data['d_img_in'].cuda()
                # labels_i = data['intact_score']
                # labels_i = torch.squeeze(labels_i.type(torch.FloatTensor)).cuda()

                patch_corrupted, patch_intact, psnr_theoric, psnr_theoric_in = data
                # patch_corrupted, patch_intact = patch_corrupted.cuda(), patch_intact.cuda()
                patch_corrupted, patch_intact, psnr_theoric, psnr_theoric_in = patch_corrupted.to(device), \
                    patch_intact.to(device), psnr_theoric.to(device), psnr_theoric_in.to(device)
                # labels = torch.squeeze(psnr_theoric.type(torch.FloatTensor)).cuda()
                # labels_intact = torch.squeeze(psnr_theoric_in.type(torch.FloatTensor)).cuda()
                labels = torch.squeeze(psnr_theoric.type(torch.FloatTensor)).cuda()
                labels_intact = torch.squeeze(psnr_theoric_in.type(torch.FloatTensor)).cuda()

                patch_corrupted = five_point_crop(i, d_img=patch_corrupted.cuda(), config=config)
                pred += net(patch_corrupted.cuda())
                patch_intact = five_point_crop(i, d_img=patch_intact.cuda(), config=config)
                pred_i += net(patch_intact.cuda())

            pred /= config.num_avg_val
            pred_i /= config.num_avg_val
            # compute loss
            loss = custom_loss(pred, labels, pred_i, labels_intact)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p))
        return np.mean(losses), rho_s, rho_p


if __name__ == '__main__':
    cpu_num = 32
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1")

    # config file
    config = Config({
        # dataset path
        "dataset_name": "our_data",

        # PIPAL
        "train_dis_path": "/mnt/IQA_dataset/PIPAL22/Train_dis/",
        "val_dis_path": "/mnt/IQA_dataset/PIPAL22/Val_dis/",
        "pipal22_train_label": "./data/PIPAL22/pipal22_train.txt",
        "pipal22_val_txt_label": "./data/PIPAL22/pipal22_val.txt",

        # KADID-10K
        "kadid10k_path": "/mnt/IQA_dataset/kadid10k/images/",
        "kadid10k_label": "./data/kadid10k/kadid10k_label.txt",

        # KONIQ-10K
        "koniq10k_path": "E:\doctor\codes\KonIQ-10k\\1024x768\\",
        "koniq10k_label": "./data/koniq10k/koniq10k_label.txt",

        # Our_data
        "our_data_path": "/home/ar28470/Documents/1024p_database_patchs32_train_intra/qp37_sp224_rgb_pattern",
        # "our_data_path": "D:\\1024p_database_patchs32_train_intra\\qp37_sp224_rgb_pattern",
        # "our_data_path": "E:\doctor\codes\\1024p_database_patchs32_train_intra\\qp37_sp224_rgb_pattern",
        "our_data_label": "./data/our_data_f1f2/1024pdata_intra.txt",

        # optimization
        "batch_size": 8, #8
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 20,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 1,  # if training koniq10k, num_avg_val is set to 1
        "num_workers": 8, #8

        # data
        "split_seed": 20,
        "train_keep_ratio": 1.0,
        "val_keep_ratio": 1.0,
        "crop_size": 224,
        "prob_aug": 0.7,

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

        # load & save checkpoint
        "model_name": "our_data_f1f2-base_s20",
        "type_name": "Our_data_f1f2",
        "ckpt_path": "./output/models/",  # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/"
    })

    config.log_file = config.model_name + ".log"
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.type_name)
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.model_name)

    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name)
    config.ckpt_path = os.path.join(config.ckpt_path, config.model_name)

    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    if not os.path.exists(config.tensorboard_path):
        os.makedirs(config.tensorboard_path)

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    if config.dataset_name == 'kadid10k':
        from data.kadid10k.kadid10k import Kadid10k

        train_name, val_name = split_dataset_kadid10k(
            txt_file_name=config.kadid10k_label,
            split_seed=config.split_seed
        )
        dis_train_path = config.kadid10k_path
        dis_val_path = config.kadid10k_path
        label_train_path = config.kadid10k_label
        label_val_path = config.kadid10k_label
        Dataset = Kadid10k
    elif config.dataset_name == 'pipal':
        from data.PIPAL22.pipal import PIPAL

        dis_train_path = config.train_dis_path
        dis_val_path = config.val_dis_path
        label_train_path = config.pipal22_train_label
        label_val_path = config.pipal22_val_txt_label
        Dataset = PIPAL
    elif config.dataset_name == 'koniq10k':
        from data.koniq10k.koniq10k import Koniq10k

        train_name, val_name = split_dataset_koniq10k(
            txt_file_name=config.koniq10k_label,
            split_seed=config.split_seed
        )
        dis_train_path = config.koniq10k_path
        dis_val_path = config.koniq10k_path
        label_train_path = config.koniq10k_label
        label_val_path = config.koniq10k_label
        Dataset = Koniq10k
    elif config.dataset_name == 'our_data':
        # from data.our_data_f1f2.our_data_f1f2 import Our_data_f1f2

        patch_pairs, psnr_values, psnr_values_in = load_psnr_for_patches(config.our_data_label, config.our_data_path)
        patch_pairs = np.asarray(patch_pairs)
        patch_tensors_stacked = torch.stack([torch.tensor(pair[0]) for pair in patch_pairs])
        patch_tensors_stacked2 = torch.stack([torch.tensor(pair[1]) for pair in patch_pairs])
        psnr_tensors = torch.tensor(psnr_values, dtype=torch.float32)  # You need to set psnr_values
        psnr_tensors_in = torch.tensor(psnr_values_in, dtype=torch.float32)
        dataset = TensorDataset(patch_tensors_stacked, patch_tensors_stacked2, psnr_tensors, psnr_tensors_in)

        # train_name, val_name = split_dataset_our_data(
        #     txt_file_name=config.our_data_label,
        #     split_seed=config.split_seed
        # )
        # dis_train_path = config.our_data_path
        # dis_val_path = config.our_data_path
        # label_train_path = config.our_data_label
        # label_val_path = config.our_data_label
        # Dataset = Our_data_f1f2

        total_size = len(dataset)
        train_size = int(0.8 * total_size)  # Choose the value you want for the train/val and test
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        print("Train Images: {}".format(len(train_dataset)))
        print("Validation Images: {}".format(len(val_dataset)))
    else:
        pass

    # data load
    # train_dataset = Dataset(
    #     dis_path=dis_train_path,
    #     txt_file_name=label_train_path,
    #     list_name=train_name,
    #     transform=transforms.Compose([RandCrop(patch_size=config.crop_size),
    #                                   Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=config.prob_aug), ToTensor()]),
    #     keep_ratio=config.train_keep_ratio
    # )
    # val_dataset = Dataset(
    #     dis_path=dis_val_path,
    #     txt_file_name=label_val_path,
    #     list_name=val_name,
    #     transform=transforms.Compose([RandCrop(patch_size=config.crop_size),
    #                                   Normalize(0.5, 0.5), ToTensor()]),
    #     keep_ratio=config.val_keep_ratio
    # )

    # train_dataset = Dataset(
    #     dis_path=dis_train_path,
    #     txt_file_name=label_train_path,
    #     list_name=train_name,
    #     transform=transforms.Compose([ToTensor()]),
    #     keep_ratio=config.train_keep_ratio
    # )
    # val_dataset = Dataset(
    #     dis_path=dis_val_path,
    #     txt_file_name=label_val_path,
    #     list_name=val_name,
    #     transform=transforms.Compose([ToTensor()]),
    #     keep_ratio=config.val_keep_ratio
    # )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    # load the data
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              num_workers=config.num_workers, drop_last=True, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
                            num_workers=config.num_workers, drop_last=True, shuffle=False)

    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
                 patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
                 depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    net = nn.DataParallel(net)
    net = net.cuda()

    # loss function
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = 0
    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(epoch, net, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p = eval_epoch(config, epoch, net, val_loader)
            logging.info('Eval done...')

            if rho_s + rho_p > main_score:
                main_score = rho_s + rho_p
                best_srocc = rho_s
                best_plcc = rho_p

                logging.info('======================================================================================')
                logging.info(
                    '============================== best main score is {} ================================='.format(
                        main_score))
                logging.info('======================================================================================')

                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.ckpt_path, model_name)
                torch.save(net.module.state_dict(), model_save_path)
                logging.info(
                    'Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))

        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))