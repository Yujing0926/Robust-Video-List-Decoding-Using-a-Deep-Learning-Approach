# Improved MANIQA
This repository is the improved PyTorch implementation of the following paper: [Yang S, Wu T, Shi S, et al. Maniqa: Multi-dimension attention network for no-reference image quality assessment[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 1191-1200.](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.html), with the modification of the following paper: [Zhang, Y., Coulombe, S., Coudoux, F.-X., Guichemerre, A. & Corlay, P. (2024). Robust Video List Decoding in Error-prone Transmission Systems Using a Deep Learning Approach. Accepted by IEEE Access.]

## Dataset
The database is created with the Matlab codes in **Matlabcode**, and you can create your route for your database (for example `1024p_database_patchs64_test_intra/qp37_patch64`), remember to change the database route when you test with this code (in `train_our_data.py` and `train_our_data_f1f2.py`).

**Attention:**
- Put the reference score file and the data python files into **data** folder. 

## Checkpoints

Click into the website and download the pretrained model checkpoints, ignoring the source files.

## Usage
### Training MANIQA model
- Modify "dataset_name" in config
- Modify train dataset path: "train_dis_path"
- Modify validation dataset path: "val_dis_path"
```
python train_maniqa.py
```
### Predicting one image quality score
- Modify the path of image "image_path"
- Modify the path of checkpoint "ckpt_path"
```
python predict_one_image_maniqa.py 
```
### Inference for testing the data with super-patches
Generating the ouput file:
- Modify the path of image "image_path"
- Modify the path of checkpoint "ckpt_path"
```
python predict_super_patches.py
```

## Environments
- Platform: PyTorch 1.8.0
- Language: Python 3.7.9
- Ubuntu 18.04.6 LTS (GNU/Linux 5.4.0-104-generic x86\_64)
- CUDA Version 11.2
- GPU: NVIDIA GeForce RTX 3090 with 24GB memory

## Requirements
 Python requirements can installed by:
```
pip install -r requirements.txt
```

## Citation
```
@inproceedings{yang2022maniqa,
  title={MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment},
  author={Yang, Sidi and Wu, Tianhe and Shi, Shuwei and Lao, Shanshan and Gong, Yuan and Cao, Mingdeng and Wang, Jiahao and Yang, Yujiu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1191--1200},
  year={2022}
}
```
```
@article{yujing2024,
    title   = {Robust Video List Decoding in Error-prone Transmission Systems Using a Deep Learning Approach},
    author  = {Zhang, Yujing and Coulombe, Stéphane and Coudoux, François-Xavier and Guichemerre, Alexis and Corlay, Patrick},
    journal = {Accepted by IEEE Access},
    year    = {2024}
}
```

