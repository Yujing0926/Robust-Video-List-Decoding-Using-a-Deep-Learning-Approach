# Robust Video List Decoding in Error-prone Transmission Systems Using a Deep Learning Approach
by Yujing ZHANG<sup>1,2,3</sup>, Stéphane COULOMBE<sup>1,3</sup>, François-Xavier COUDOUX<sup>2</sup>, Alexis GUICHEMERRE1,3, and Patrick CORLAY<sup>2</sup>.

<sup>1</sup> Department of Software and IT Engineering, École de technologie supérieure, Université du Québec, Montréal, QC H3C 1K3, Canada

<sup>2</sup> CNRS, UMR 8520, DOAE—Département d’Opto-Acousto-Électronique, IEMN—Institut d’Électronique de Microélectronique et de Nanotechnologie, Université Polytechnique Hauts-de-France, 59313 Valenciennes, France

<sup>3</sup> International Laboratory on Learning Systems (ILLS), McGill - ÉTS - Mila - CNRS - Université Paris Saclay - CentraleSupelec, Montréal, QC, H3H 2T2, Canada

This work was supported in part by the Conseil de Recherches en Sciences Naturelles et en Génie du Canada (CRSNG), and in part by the Université Polytechnique Hauts-de-France (UPHF).

## Abstract
This paper introduces a novel deep-learning assisted video list decoding method for error-prone video transmission systems. Unlike traditional list decoding techniques, our proposed system uses a Transformer-based no-reference image quality assessment method to select the highest-scoring reconstructed video candidate after reception. Three new components are defined and used in the Transformer-assisted image quality evaluation metric: neighborhood-based patch fidelity aggregation, discriminant color texture transformation and ranking-constrained penalty loss function. We have also created our own database of non-uniformly distorted images, similar to those that might result from transmission errors, in a HEVC context. In our specific testing context, our improved Transformer-assisted method has a decision accuracy of 100% for intra-coded image, while, for errors occurring in an inter image, it is 96%. Notably, in the few cases where a wrong choice is made, the selected candidate’s quality remains similar to the intact frame.


## MatlabCode-new
This part of code is to introduce the generation of our database which will be input into CNN model and Transformer model later. Please read the **README** file in this folder to find how to use our proposed codes. This part is used to generate the patch-based database with full-reference (FR) patch-level quality measures for training, and the image-based database with ground-truth FR scores for testing.

## PythonCode-new
This part of code implements our proposed CNNIQA-based image quality assessment (IQA) method and Transformer-based IQA method. Please read the **README** files in each folder to find how to install and use our proposed codes. 

## Citation
Please cite the following paper in the case you use our codes for research proposes:
```
@article{yujing2024,
    title   = {Robust Video List Decoding in Error-prone Transmission Systems Using a Deep Learning Approach},
    author  = {Zhang, Yujing and Coulombe, Stéphane and Coudoux, François-Xavier and Guichemerre, Alexis and Corlay, Patrick},
    journal = {Accepted by IEEE Access},
    year    = {2024}
}
```
