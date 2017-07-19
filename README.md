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
This library is based on [Caffe](https://github.com/BVLC/caffe). [CuDNN](https://developer.nvidia.com/cudnn) and [NCCL](https://github.com/NVIDIA/nccl) are required. Please follow
the installation instruction of [Caffe](https://github.com/BVLC/caffe).


Include
----
* Imlplementation details introduced in the [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shen_Semantic_Segmentation_via_CVPR_2017_paper.pdf), 
including training code and test code.
* GPU Memory reuse by setting different flow for each data blob, which saves about half of memory in the training stage.
* Multi-GPU efficient running.
* Support multi-batch normalization.
* Support training generative adversial networks.

Inference
----

Training
----

Datasets
----
* Pascal VOC 2012

* Cityscapes

* MIT Secene Parsing 


Pretrained Model
----
* ImageNet model
  
* Pascal VOC 2012

* Cityscapes

* MIT Secene Parsing 

