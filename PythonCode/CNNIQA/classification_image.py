import os
import time
import numpy
from argparse import ArgumentParser


def test_for_scores(video_name, candidate_img_route, output_route, model_used, yuv, APPEND_CHAR, patch_size):
    img_route = candidate_img_route + video_name
    img_name_route = video_name_route + video_name
    img_list_data = os.listdir(img_name_route)
    if yuv:
        img_name = [i for i in img_list_data if i.endswith('.yuv')]
    else:
        img_name = [i for i in img_list_data if i.endswith('.png')]
    print(img_name)

    result_route = output_route + APPEND_CHAR + video_name
    if not (os.path.exists(result_route)):
        os.mkdir(result_route)

    # for im in range(3, 4):
    for im in range(0, len(img_name)):
        img_png = img_route + APPEND_CHAR + img_name[im]
        img_list = os.path.basename(img_png)
        img = img_list.split(".")[-2]
        patch_route = img_route + APPEND_CHAR + img + APPEND_CHAR

        # patch_route = img_route
        if not (os.path.exists(patch_route)):
            os.mkdir(patch_route)
        patch_result = result_route + APPEND_CHAR + 'cnniqa_test_' + img_name[im] + '.txt'

        patch_list_data = os.listdir(patch_route)
        if yuv:
            patch_list = [i for i in patch_list_data if i.endswith('.yuv')]
        else:
            patch_list = [i for i in patch_list_data if i.endswith('.png')]

        for p in range(0, len(patch_list)):
            patch_name = patch_route + APPEND_CHAR + patch_list[p]
            model_path = 'checkpoints' + APPEND_CHAR
            cmd = 'python test_1patch.py --im_path=%s --model_file=%s%s --patch_size=%d >>%s' \
                  % (patch_name, model_path, model_used, patch_size, patch_result)
            os.system(cmd)


def test_for_scores_pair(video_number, video_name, candidate_img_route, output_route, model_used, yuv, APPEND_CHAR):
    img_route = candidate_img_route + video_name
    img_list_data = os.listdir(img_route)
    if yuv:
        img_name = [i for i in img_list_data if i.endswith('.yuv')]
    else:
        img_name = [i for i in img_list_data if i.endswith('.png')]
    print(img_name)

    result_route = output_route + APPEND_CHAR + video_name
    if not (os.path.exists(result_route)):
        os.mkdir(result_route)

    # for im in range(0, 1):
    for im in range(0, len(img_name)):
        img_png = img_route + APPEND_CHAR + img_name[im]
        img_list = os.path.basename(img_png)
        img = img_list.split(".")[-2]
        patch_route = img_route + APPEND_CHAR + img + APPEND_CHAR
        patch_2_route = img_route + APPEND_CHAR + 'i' + str(video_number) + APPEND_CHAR

        # patch_route = img_route
        if not (os.path.exists(patch_route)):
            os.mkdir(patch_route)
        patch_result = result_route + APPEND_CHAR + 'cnniqa_test_' + img_name[im] + '.txt'

        patch_list_data = os.listdir(patch_route)
        if yuv:
            patch_list = [i for i in patch_list_data if i.endswith('.yuv')]
        else:
            patch_list = [i for i in patch_list_data if i.endswith('.png')]

        for p in range(0, len(patch_list)):
            patch_name = patch_route + APPEND_CHAR + patch_list[p]
            vn = patch_list[p].split("_", 1)
            vn_n = vn[1].split("_", 3)
            if 'c' in vn_n[0]:
                vnnn = vn_n[3].split(".", 1)
                vn_n_0 = vn_n[0]
                vn_n_inatct = 'i' + vn_n_0[1:]
                # print(vn_n_inatct)
                patch_2_name = patch_2_route + APPEND_CHAR + 'patch_' + vn_n_inatct + '_' + vn_n[2] + '_' \
                               + vnnn[0] + '.png'
            elif 'i' in vn_n[0]:
                patch_2_name = patch_2_route + APPEND_CHAR + patch_list[p]

            model_path = 'checkpoints' + APPEND_CHAR
            cmd = 'python test_pair_patches.py --im_path_1=%s --im_path_1=%s --model_file=%s%s >>%s' \
                  % (patch_name, patch_2_name, model_path, model_used, patch_result)
            os.system(cmd)


if __name__ == "__main__":
    start = time.time()

    system = 'Windows'
    if system == 'Windows':
        APPEND_CHAR = '\\'
    elif system == 'Linux':
        APPEND_CHAR = '/'
    elif system == 'Mac':
        APPEND_CHAR = '/'

    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    parser = ArgumentParser(description='Test CNNIQA')
    # parser.add_argument("--video_number", type=int, default=8)
    parser.add_argument("--database_route", type=str,
                        default='E:\doctor\codes\\1024p_database_patchs64_test_inter')
    # default='/home/ar28470@ens.ad.etsmtl.ca/Documents/1024p_database_patchs64_test_intra'
    # default='/home/ar28470/Documents/1024p_database_patchs64_test_intra'
    # default='E:\doctor\codes\\1024p_database_patchs64_test_intra'
    parser.add_argument("--qp", type=str,
                        default='37')
    parser.add_argument("--model", type=str,
                        default='CNNIQA-2loss-1024pDatabase-patch64-psnr-qp37-inter-rgb-1score-pattern-f1f2')
    parser.add_argument("--patch_size", type=int, default=64,
                        help="patch_size")
    args = parser.parse_args()

    model_used = args.model
    patch_size = args.patch_size

    yuv = False
    pair = False

    # original_img_route = args.database_route + APPEND_CHAR + 'ref_img_qp' + args.qp + APPEND_CHAR
    candidate_img_route = args.database_route + APPEND_CHAR + 'qp' + args.qp + '_rgb_pattern' + APPEND_CHAR
    video_name_route = args.database_route + APPEND_CHAR + 'qp' + args.qp + '_rgb_pattern' + APPEND_CHAR #+ '57videos' + APPEND_CHAR
    # candidate_img_route = args.database_route + '/test_for_train/c0_0/'

    output_route = args.database_route + APPEND_CHAR + 'patch' + str(patch_size) + '_psnr_qp' + args.qp \
                   + '_rgb_pattern_f2' + APPEND_CHAR
    # output_route = args.database_route + '/test_train_c0_0_Y/'
    if not (os.path.exists(output_route)):
        os.mkdir(output_route)

    list_data = sorted(os.listdir(video_name_route))
    video_name = [s for s in list_data]
    # print(video_name)

    if pair:
        test_for_scores_pair(args.video_number, video_name[args.video_number], candidate_img_route, output_route,
                             model_used, yuv, APPEND_CHAR)
    else:
        for i in range(0, len(video_name)):
            test_for_scores(video_name[i], candidate_img_route, output_route, model_used, yuv, APPEND_CHAR, patch_size)

    end = time.time()
    print("running time: ", end - start)

