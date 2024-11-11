# CNNIQA
PyTorch implementation of the following paper:
[Kang L, Ye P, Li Y, et al. Convolutional neural networks for no-reference image quality assessment[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1733-1740.](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf) <br>
with the modification of the following paper:
[Yujing ZHANG, Stéphane Coulombe, François-Xavier Coudoux, Anthony Trioux, Patrick Corlay. Optimisation du décodage par liste de vidéos corrompues basée sur une architecture CNN. 22ème édition de la conférence COmpression et REprésentation des Signaux Audiovisuels, CORESA, Jun 2023, Lille, France. 4 p. ⟨hal-04246635⟩](https://hal.science/hal-04246635)

### Note
- The database is created with the Matlab codes in **Matlabcode-new**, and you can create your route for your database (for example `1024p_database_patchs64_test_intra/qp37_patch64`), remember to change the database route when you test with this code (in `main_new.py`, `classification_image.py` and `classification_main.py`).

## Training
With the `main_new.py`, we implement the training of the CNN model with improved local normalization algorithm.<br>

```bash
python main_new.py 
```
or
```bash
python3 main_new.py
```
We can use `python` or `python3`. With the `main_new.py`, we implement the training of the CNN model with 2 improved loss functions.
Note that for training the `main_new.py`, you should also correctly generate the reference scores of the data.

## Evaluation
Test with only 1 patch 
```bash
python test_1patch.py --im_path=data/your_patch
```
You can use the file `classification_image.py` to do the test with the image in our database.
```bash
python classification_image.py --video_number=0 --qp=37 --model=your_model_name
```

## Requirements
```bash
conda create -n CNNIQA pip python=3.8
source activate CNNIQA
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
source deactive
```
- Python 3.8.0
- PyTorch 1.11.0
- TensorboardX 2.6, TensorFlow 2.2.0
- [pytorch/ignite 0.2.1](https://github.com/pytorch/ignite)

Note: You need to install the right CUDA version.


