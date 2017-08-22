SegModel
=====


This repository is for [Semantic Segmentation via Structured Patch Prediction, Context CRF and Guidance CRF](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shen_Semantic_Segmentation_via_CVPR_2017_paper.pdf).

    @inproceedings{shen2017segmodel,
      author = {Falong Shen, Gan Rui, Shuicheng Yan and Gang Zeng},
      title = {Semantic Segmentation via Structured Patch Prediction, Context CRF and Guidance CRF},
      booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
    }

Installation
----
This library is based on [Caffe](https://github.com/BVLC/caffe). [CuDNN 7](https://developer.nvidia.com/cudnn) and [NCCL 1](https://github.com/NVIDIA/nccl) are required. Please follow
the installation instruction of [Caffe](https://github.com/BVLC/caffe).


Include
----
* Imlplementation details introduced in the [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shen_Semantic_Segmentation_via_CVPR_2017_paper.pdf), 
including training code and test code.
* GPU Memory reuse by setting different flow for each data blob, which saves about half of memory in the training stage.
* Multi-GPU efficient running.
* Support multi-batch normalization.
* Support training generative adversial networks.

Scripts
----
Matlab code. Please execute the scripts in Matlab folder.

* Cityscapes pretrained models</br>
    [model 0(182M)](http://host.robots.ox.ac.uk/pascal/VOC/)</br>
    [model 1(182M)](http://host.robots.ox.ac.uk/pascal/VOC/)</br>
    [model 2(182M)](http://host.robots.ox.ac.uk/pascal/VOC/)</br>


Datasets
---- 
* [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/) (Widely used to evaluate segmenation alogrithms.)

PASCAL VOC 2012 semantic segmentation benchmark contains 20 foreground object classes and one background class.
The original dataset contains 1464 train, 1449 val, and 1456 test pixel-level labeled images for training, validation, and testing, respectively. 
The dataset is augmented by the extra annotations provided by  in 10582 training images. However, the label strategy of the extra annotations is not exactly consistant 
with the original annotation. Please refer the label images for more details.

* [Cityscapes](https://www.cityscapes-dataset.com/) (High resolution. Road parsing.)

Cityscapes dataset consists of 2975 training images and 500 validation images . Both have pixel-wise annotations. There are also another about 19,998 image with
coarse annotation. There are 19 categories in this dataset and there is no background category. All the images are about street scene in some European cities and are taken by
car-carried cameras. It should be noticed that the size of every image is 1024 ×2048 in this dataset.

* [MIT Secene Parsing](http://sceneparsing.csail.mit.edu/) (Too many categories.)

The data for this benchmark comes from ADE20K Dataset which contains more than 20K scene-centric images exhaustively annotated with objects and object parts. Specifically, the benchmark is divided into 20K images for training, 2K images for validation, and another batch of held-out images for testing. There are totally 150 semantic categories included for evaluation, which include stuffs like sky, road, grass, and discrete objects like person, car, bed. Note that there are non-uniform distribution of objects occuring in the images, mimicking a more natural object occurrence in daily scene.

