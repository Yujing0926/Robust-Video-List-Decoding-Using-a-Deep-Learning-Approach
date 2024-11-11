import os
import re
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patch_psnr import patch_generation_psnr
from patch_psnr import patch_generation_psnr_rgb

start = time.time()

system = 'Windows'

if system == 'Windows':
    APPEND_CHAR = '\\'
elif system == 'Linux':
    APPEND_CHAR = '/'
elif system == 'Mac':
    APPEND_CHAR = '/'

QP = 'qp37'
codec = 'HEVC'
reference_metric = 'psnr'
error_frame = 2

patch_size = 64
image_width = 1920
image_height = 1024

rgb = True
yuv = False
generate_score = True

damage_rate = ['0.3', '0.6', '0.9']
# damage_rate = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','0.99']

if error_frame == 1:
    frame_type = 'intra'
else:
    frame_type = 'inter'

main_route = 'E:\doctor\codes' + APPEND_CHAR
test_database_route = main_route + APPEND_CHAR + '1024p_database_patchs' + str(patch_size) + '_test_' + frame_type #+ '_150'
original_img_route = test_database_route + APPEND_CHAR + 'ref_img_' + QP + APPEND_CHAR
candidate_img_route = test_database_route + APPEND_CHAR + QP + '_rgb_pattern' + APPEND_CHAR # + '57videos' + APPEND_CHAR
candidate_img_name_route = test_database_route + APPEND_CHAR + QP + '_rgb_pattern' + APPEND_CHAR # + '57videos' + APPEND_CHAR

output_route = test_database_route + APPEND_CHAR + 'patch' + str(patch_size) + '_' + reference_metric \
               + '_' + QP + '_rgb_pattern_f1f2' + APPEND_CHAR
if not (os.path.exists(output_route)):
    os.mkdir(output_route)

# list_data = sorted(os.listdir(candidate_img_name_route))
list_data = os.listdir(candidate_img_name_route)
video_name = [s for s in list_data]
print(len(video_name))

# videonames_route = main_route + APPEND_CHAR + 'videonames.csv'
# df = pd.read_csv(videonames_route, encoding='utf-8')
# video_name = df['57videos']
# video_name = video_name.values.tolist()
# video_name = [s for s in video_name if str(s) != 'nan']
# print(len(video_name))

decision_list = list()
score_list = list()
good_decision = list()
bad_decision = list()

intact_score = list()
bad_intact_score = list()
bad_candidate_score = list()
bad_difference = list()

for vn in range(0, len(video_name)):
# for vn in range(0, 1):
    print(vn)
    print(video_name[vn])

    original_img_name = original_img_route + video_name[vn] + '_' + str(error_frame)

    img_route = candidate_img_route + video_name[vn] + '_' + str(error_frame)
    img_list_data = os.listdir(img_route)
    if yuv:
        original_img = original_img_name + '.yuv'
        img_name = [i for i in img_list_data if i.endswith('.yuv')]
    else:
        original_img = original_img_name + '.png'
        img_name = [i for i in img_list_data if i.endswith('.png')]
    # print(img_name)

    result_route = output_route + APPEND_CHAR + video_name[vn] + '_' + str(error_frame)
    if not (os.path.exists(result_route)):
        os.mkdir(result_route)
    scores_result = result_route + APPEND_CHAR + 'cnniqa_test_' + video_name[vn] + '.txt'
    candidate_scores = list()
    candidate_name = list()

    for im in range(0, len(img_name)):
        # for im in range(0,1):
        img_png = img_route + APPEND_CHAR + img_name[im]
        img_list = os.path.basename(img_png)
        img = img_list.split(".")[-2]
        patch_route = img_route + APPEND_CHAR + img + APPEND_CHAR
        if not (os.path.exists(patch_route)):
            os.mkdir(patch_route)
        patch_result = result_route + APPEND_CHAR + 'cnniqa_test_' + img_name[im] + '.txt'
        patch_result_name = result_route + APPEND_CHAR + 'cnniqa_test_' + str(img_name[im]) + '_name.txt'

        # if vn in except_list:
        #     print("rerun")
        # else:
        #     patch_result_new = output_route_new + '/' + video_name[vn] + '/cnniqa_test_' + img_name[im] + '.txt'
        #     shutil.copy(patch_result,patch_result_new)

        if not generate_score:

            if rgb:
                patch_generation_psnr_rgb(img_png, original_img, img, patch_route, patch_route, patch_size, image_width,
                                          image_height, APPEND_CHAR)
            else:
                patch_generation_psnr(img_png, original_img, img, patch_route, patch_route, patch_size, image_width,
                                      image_height, APPEND_CHAR)
        else:

            patch_list_data = os.listdir(patch_route)
            if yuv:
                patch_list = [i for i in patch_list_data if i.endswith('.yuv')]
            else:
                patch_list = [i for i in patch_list_data if i.endswith('.png')]
            # print(patch_list)
            # for p in range(0,len(patch_list)):
            #     patch_name = patch_route + '/' + patch_list[p]
            #     cmd = 'python test_demo.py --im_path=%s --model_file=checkpoints/' \
            #           'CNNIQA-1024pDatabase-EXP0-lr=0.001-qp37-psnr-patch64-intra >>%s' \
            #           % (patch_name, patch_result)
            #     os.system(cmd);

            # f = open(patch_result,'r')
            # l = []
            # for line in f:
            #     line = line.strip('\n')
            #     l.append(line.split(' '))
            # patch_scores = np.array(l,dtype=float)

            patch_scores = list()
            with open(patch_result, "r") as f:
                data = f.read()
                reg = re.compile('(?<= )\s*\d*\.\d*')
                patch_scores = reg.findall(data)
                patch_scores = np.array(patch_scores)
                patch_scores.tolist()

                ## for triq scores
                # file = f.readlines()
                # for line in file:
                #     line = line.strip("\n")
                #     score = float(line)
                #     patch_scores.append(score)
                # patch_scores = np.array(patch_scores)
                # # print(np.shape(patch_scores))

            patch_scores = [float(x) for x in patch_scores]
            for x in range(0,len(patch_scores)):
                if patch_scores[x] >= 1:
                    patch_scores[x] = 0.0001
            # print(len(patch_scores))

            ## for triq scores
            # if len(patch_scores) == 0:
            #     continue
            # else:
            #     img_score = np.mean(patch_scores)
            #     print(img_score)
            #     candidate_scores.append(img_score)
            #     candidate_name.append(img_name[im])
            #     if 'i' in img_name[im]:
            #         intact_score.append(img_score)
            # f.close()

            img_score = np.mean(patch_scores)
            print(img_score)
            candidate_scores.append(img_score)
            candidate_name.append(img_name[im])
            name = ['patch_name', 'patch_score']
            data = list(zip(patch_list, patch_scores))
            patch_data = pd.DataFrame(data=data, columns=name)
            patch_score_csv = result_route + APPEND_CHAR + img + '_patch_scores.csv'
            patch_data.to_csv(patch_score_csv, encoding='gbk')
            if 'i' in img_name[im]:
                intact_score.append(img_score)

    if generate_score:
        # new_list = []
        # new_name_list = []
        # for elem in candidate_scores:
        #     if not np.isnan(elem):
        #         new_list.append(elem)
        #         name_index = candidate_scores.index(elem)
        #         new_name_list.append(candidate_name[name_index])
        # max_score = np.max(new_list)
        # max_index = np.argmax(new_list)
        # decision_list.append(new_name_list[max_index])
        # score_list.append(new_list[max_index])
        #
        # if 'i' in new_name_list[max_index]:
        #     good_decision.append(new_name_list[max_index])
        # else:
        #     bad_decision.append(new_name_list[max_index])
        #     bad_candidate_score.append(new_list[max_index])
        #     for im in range(0, len(new_name_list)):
        #         # if candidate_name[im] == 'i' + str(vn) + '.png':
        #         if 'i' in new_name_list[im]:
        #             bad_intact = new_list[im]
        #             bad_intact_score.append(bad_intact)
        #             # break
        #     bad_difference.append(float(new_list[max_index]) - float(bad_intact))

        print(candidate_name)
        max_score = np.max(candidate_scores)
        max_index = np.argmax(candidate_scores)
        decision_list.append(candidate_name[max_index])
        score_list.append(candidate_scores[max_index])
        # print(candidate_scores)

        # fp = open(scores_result,'r')
        # ls = []
        # for line in fp:
        #     line = line.strip('\n')
        #     ls.append(line.split(' '))
        # fp.close()
        # test_results_list = np.array(ls,dtype=float)
        # print(test_results_list)
        # for im in range(0, len(img_name)):
        #     if img_name[im] == 'I' + str(vn) + '.png':
        #         intact_score.append(test_results_list[im])

        # test_results_list = test_results.tolist()
        # max_score = max(test_results_list,key=lambda x:float(x))
        # max_score = np.max(test_results_list)
        # print(max_score)
        # max_index = np.argmax(test_results_list)
        # print(max_index)
        # decision_list.append(img_name[max_index])
        # print(decision_list)
        # score_list.append(test_results_list[max_index])

        # if candidate_name[max_index] == 'i'+str(vn)+'.png':
        if 'i' in candidate_name[max_index]:
            good_decision.append(candidate_name[max_index])
        else:
            bad_decision.append(candidate_name[max_index])
            bad_candidate_score.append(candidate_scores[max_index])
            for im in range(0, len(candidate_name)):
                # if candidate_name[im] == 'i' + str(vn) + '.png':
                if 'i' in candidate_name[im]:
                    bad_intact = candidate_scores[im]
                    bad_intact_score.append(bad_intact)
                    # break
            bad_difference.append(float(candidate_scores[max_index]) - float(bad_intact))

if generate_score:
    good_decision_percent = len(good_decision) / len(video_name)
    print("good_decision_percent: ", good_decision_percent)
    print("bad_decision_number: ", len(bad_decision))
    print("bad_decision: ", bad_decision)
    print("bad_decision_intact: ", bad_intact_score)
    print("bad_decision_candidate: ", bad_candidate_score)

    name = ['decided_name', 'cnniqa_score']
    data = list(zip(decision_list, score_list))
    test = pd.DataFrame(data=data, columns=name)
    cnniqa_decision_csv = output_route + APPEND_CHAR + codec + '_' + QP + '_all_cnniqa_classification_result.csv'
    test.to_csv(cnniqa_decision_csv, encoding='gbk')

    plt.figure()
    plt.xlabel("CNNIQA_score_difference")
    plt.ylabel("Number of results")
    plt.title("CNNIQA_differences_histogram_of_bad_classification_results_" + QP + "_" + codec)
    plt.hist(bad_difference)
    plt.savefig(output_route + APPEND_CHAR + codec + '_' + QP + '_cnniqa_difference_hist.png')

    intact_score = [float(x) for x in intact_score]
    intact_mean = np.mean(intact_score)
    plt.figure()
    plt.xlabel("CNNIQA_score")
    plt.ylabel("Number of intact sequences")
    plt.title("CNNIQA_scores_histogram_of_intact_videos_" + QP + "_" + codec)
    plt.hist(intact_score)
    plt.savefig(output_route + APPEND_CHAR + codec + '_' + QP + '_CNNIQA_intact_hist.png')

    score_list = [float(x) for x in score_list]
    system_mean = np.mean(score_list)
    plt.figure()
    plt.xlabel("CNNIQA_score")
    plt.ylabel("Number of sequences")
    plt.title("CNNIQA_scores_histogram_of_system_videos_" + QP + "_" + codec)
    plt.hist(score_list)
    plt.savefig(output_route + APPEND_CHAR + codec + '_' + QP + '_cnniqa_system_hist.png')

    bad_candidate_score = [float(x) for x in bad_candidate_score]
    bad_candidate_mean = np.mean(bad_candidate_score)
    bad_intact_score = [float(x) for x in bad_intact_score]
    bad_intact_mean = np.mean(bad_intact_score)
    bad_difference = [float(x) for x in bad_difference]
    bad_difference_mean = np.mean(bad_difference)
    f = open(output_route + APPEND_CHAR + codec + '_' + QP + '_decision_info.txt', 'w+')
    f.writelines(['cnniqa_classification_percent: ', str(good_decision_percent)])
    f.writelines(['\nintact_image_score_mean: ', str(intact_mean)])
    f.writelines(['\nsystem_decision_image_score_mean: ', str(system_mean)])
    f.writelines(['\nbad_decision_image_score_mean: ', str(bad_candidate_mean)])
    f.writelines(['\nbad_decision_intact_image_score_mean: ', str(bad_intact_mean)])
    f.writelines(['\nbad_decision_difference_score_mean: ', str(bad_difference_mean)])
    f.writelines(['\nbad_decision_list: ', str(bad_decision)])

end = time.time()
print("running time: ", end - start)
