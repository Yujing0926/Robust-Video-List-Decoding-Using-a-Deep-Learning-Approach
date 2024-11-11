import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F


class Our_data_f1f2(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Our_data_f1f2, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        intact_files_data, intact_score_data = [], []
        # im_names, im_scores = [], []
        # with open(self.txt_file_name, 'r') as listFile:
        #     for line in listFile:
        #         dis, score = line.split()
        #         score = float(score)
        #         im_names.append(dis)
        #         im_scores.append(score)

        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score, intact, score_in = line.split()
                if dis in list_name:
                    score = float(score)
                    score_in = float(score_in)
                    dis_files_data.append(dis)
                    score_data.append(score)
                    intact_files_data.append(intact)
                    intact_score_data.append(score_in)
            # for line in listFile:
            #     dis, score = line.split()
            #     vn = dis.split("_", 1)
            #     vn_n = vn[1].split("_", 3)
            #     dis_files_data.append(dis)
            #     score_data.append(score)
            #     if 'c' in vn_n[0]:
            #         vnnn = vn_n[3].split(".", 1)
            #         vn_n_0 = vn_n[0]
            #         vn_n_inatct = 'i' + vn_n_0[1:]
            #         patch_2_name = vn[0] + '_' + vn_n_inatct + '_' + vn_n[2] + '_' + vnnn[0] + '.png'
            #         # print(patch_2_name)
            #         intact_files_data.append(patch_2_name)
            #         intact_index = im_names.index(patch_2_name)
            #         intact_score = im_scores[intact_index]
            #         intact_score_data.append(intact_score)
            #     elif 'i' in vn_n[0]:
            #         intact_files_data.append(dis)
            #         intact_score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        intact_score_data = np.array(intact_score_data)
        intact_score_data = self.normalization(intact_score_data)
        intact_score_data = list(intact_score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data,
                          'd_img_intact': intact_files_data, 'intact_score_list': intact_score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        # print(idx)
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        # d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.data_dict['score_list'][idx]

        d_img_intact_name = self.data_dict['d_img_intact'][idx]
        d_img_intact = cv2.imread(os.path.join(self.dis_path, d_img_intact_name), cv2.IMREAD_COLOR)
        # d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img_intact = cv2.cvtColor(d_img_intact, cv2.COLOR_BGR2RGB)
        d_img_intact = np.array(d_img_intact).astype('float32') / 255
        d_img_intact = np.transpose(d_img_intact, (2, 0, 1))
        # print(d_img_intact)
        intact_score = self.data_dict['intact_score_list'][idx]

        sample = {
            'd_img_org': d_img,
            'd_img_in': d_img_intact,
            'score': score,
            'intact_score': intact_score
        }
        if self.transform:
            sample = self.transform(sample)
        # d_img = self.transform(d_img)
        # d_img_intact = self.transform(d_img_intact)
        # score = self.transform(score)
        # intact_score = self.transform(intact_score)
        # return d_img, d_img_intact, score, intact_score
        return sample
