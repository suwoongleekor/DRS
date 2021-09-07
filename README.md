# Under construction

refined_pseudo_segmentation_labels_DRS_CAAS2:_
https://www.dropbox.com/s/jbd39bm7x2y9pjs/refined_pseudo_segmentation_labels_DRS_CAAS2.zip?dl=0

CAAS-DRS ckpt
https://www.dropbox.com/s/x0ko3ho9icndrn2/best_deeplabv3plus_resnet101_voc_os16.pth?dl=0



# Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation (AAAI 2021)

Official pytorch implementation of our paper:
Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation [[Paper]](https://arxiv.org/abs/2103.07246), Beomyoung Kim, Sangeun Han, and Junmo Kim, AAAI 2021

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/discriminative-region-suppression-for-weakly/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=discriminative-region-suppression-for-weakly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/discriminative-region-suppression-for-weakly/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=discriminative-region-suppression-for-weakly)

We propose the discriminative region suppression (DRS) module that is a simple yet effective method to expand object activation regions. DRS suppresses the attention on discriminative regions and spreads it to adjacent non-discriminative regions, generating dense localization maps.

[2021.06.10] we support DeepLab-V3 segmentation network! 

<img src = "https://github.com/qjadud1994/DRS/blob/main/docs/DRS_CAM.png" width="60%" height="60%">

![DRS module](https://github.com/qjadud1994/DRS/blob/main/docs/DRS_module.png)

## Setup

1. Dataset Preparing

    * [Download PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit)
    * you can obtain `SegmentationClassAug/` [[link]](https://drive.google.com/drive/folders/1_ik8n5Q4C77X-aIfKiqidFEDQ6zY9JNM?usp=sharing) (augmented with SBD dataset).
    * [Download saliency maps](https://drive.google.com/drive/folders/1I-456-_OFVWhZdCBBPW9NSIkr0H3FBeP?usp=sharing) used for background cues.
    
    ~~~
    # dataset structure
    VOC2012/
        --- Annotations/
        --- ImageSets/
        --- JPEGImages/
        --- SegmentationClassAug/
        --- saliency_map/
        --- refined_pseudo_segmentation_labels/
    ~~~


2. Requirements
    `pip install -r requirements.txt`


## Training & Pseudo Segmentation Labels Generation
* step1 : training the classifier with DRS modules
* step2 : training the refinement network for the localization maps refinement
* step3 : pseudo segmentation labels generation

~~~ 
# all-in-one
bash run.sh 
~~~

| Model | pretrained |
| :----:  | :----:    |
| VGG-16 with the learnable DRS | [DRS_learnable/best.pth](https://drive.google.com/drive/folders/1AyKsOmJd_241BNCYp_MKs9mCdr0i27Qs?usp=sharing) |
| Refinement network | [Refine_DRS_learnable/best.pth](https://drive.google.com/drive/folders/1w50rhVTGBJXW4oCJ88DpsieXABSWJhaG?usp=sharing) |
|                    |          |
| Pseudo Segmentation Labels | [refined_pseudo_segmentation_labels/](https://drive.google.com/drive/folders/1IS9_YCrRJwz3c7y3KwTET2_dYUPlNYo6?usp=sharing) |


## Training the DeepLab-V2 using pseudo labels
We adopt the DeepLab-V2 pytorch implementation from https://github.com/kazuto1011/deeplab-pytorch.

* According to the [DeepLab-V2 pytorch implementation](https://github.com/kazuto1011/deeplab-pytorch#download-pre-trained-caffemodels) , we requires an initial weights [[download]](https://drive.google.com/file/d/1Wj8Maj9KGQgwtDfvIp8FChsdAIgDvliT/view?usp=sharing).

~~~
cd DeepLab-V2-PyTorch/

# motify the dataset path (DATASET.ROOT)
vi configs/voc12.yaml

# 1. training the DeepLab-V2 using pseudo labels
bash train.sh

# 2. evaluation the DeepLab-V2
bash eval.sh
~~~

## Training the DeepLab-V3+ using pseudo labels
We adopt the DeepLab-V3+ pytorch implementation from https://github.com/VainF/DeepLabV3Plus-Pytorch.

Note that **DeepLab-V2** suffers from the small batch issue, therefore, they utilize COCO pretrained weight and freeze batch-normalization layers; DeepLab-V2 without COCO-pretrained weight cannot reproduce their performance even in fully-supervised setting.

In contrast, **DeepLab-V3 does not require the COCO-pretrained weight** due to the recent large memory GPUs and Synchronized BatchNorm.
We argue that the choice of DeepLab-V3 network is more reasonable and better to measure the quality of pseudo labels.

~~~
cd DeepLabV3Plus-Pytorch/

# training & evaluation the DeepLab-V3+ using pseudo labels
vi run.sh # modify the dataset path --data_root
bash run.sh
~~~

| Model | mIoU | mIoU + CRF | pretrained |
| :----:  | :----: | :----: | :----: |
| DeepLab-V2 with ResNet-101 | 69.4% | 70.4% | [[link]](https://drive.google.com/drive/folders/1zJnRI5WRnv4cL9XY5jAojwIcO7MrUwun?usp=sharing)
| DeepLab-V3+ with ResNet-101 | 70.4% | 71.0% | [[link]](https://drive.google.com/file/d/1W1LV3gvBPRr2lIlWdvqZ-cs87qYT8Nax/view?usp=sharing)

* Note that the pretrained weight path
`./DeepLab-V2-Pytorch/data/models/Deeplabv2_pseudo_segmentation_labels/deeplabv2_resnet101_msc/train_cls/checkpoint_final.pth`


<img src = "https://github.com/qjadud1994/DRS/blob/main/docs/DRS_segmap.png" width="60%" height="60%">

## Citation
We hope that you find this work useful. If you would like to acknowledge us, please, use the following citation:
~~~
@inproceedings{kim2021discriminative,
    title={Discriminative Region Suppression for Weakly-Supervised Semantic Segmentation},
    author={Kim, Beomyoung and Han, Sangeun and Kim, Junmo},
    year={2021},
    booktitle={AAAI Conference on Artificial Intelligence},
}
~~~
