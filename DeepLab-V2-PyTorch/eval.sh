# Evaluation DeepLab-V2 trained with pseudo segmentation labels

DATASET=voc12_4gpu
LOG_DIR=Deeplabv2_new_download

CUDA_VISIBLE_DEVICES=3 python main.py test \
-c configs/${DATASET}.yaml \
-m data/models/${LOG_DIR}/deeplabv2_resnet101_msc/train_cls/checkpoint_final.pth  \
--log_dir=${LOG_DIR}

# evaluate the model with CRF post-processing
CUDA_VISIBLE_DEVICES=3 python main.py crf \
-c configs/${DATASET}.yaml \
--log_dir=${LOG_DIR}
