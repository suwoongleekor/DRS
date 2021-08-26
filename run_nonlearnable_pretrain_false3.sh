# Training Classifier with the DRS_non_learnable_pretrain_false, lr=0.01

C1=2
C2=3
CASE=DRS_non_learnable_pretrain_false3

CUDA_VISIBLE_DEVICES=${C1},${C2} python scripts/train_cls.py \
    --img_dir=/data_root/WSSS/DRS_log/${CASE}/ \
    --lr=0.01 \
    --epoch=15 \
    --decay_points='5,10' \
    --delta=0.55 \
    --logdir=logs/DRS_non_learnable_pretrain_false \
    --save_folder=checkpoints/${CASE} \
    --show_interval=50 \
    --from_scratch

# Evaluation pseudo segmentation labels generated by the classifier
 CUDA_VISIBLE_DEVICES=${C1} python scripts/test_cls.py \
     --checkpoint=checkpoints/${CASE}/best.pth \
     --img_dir=/data_root/WSSS/DRS_log/${CASE}/ --delta=0.55


# Generating localization maps for the refinement learning
CUDA_VISIBLE_DEVICES=${C1} python scripts/localization_map_gen.py \
    --img_dir=/data_root/WSSS/DRS_log/${CASE}/ \
    --checkpoint=checkpoints/${CASE}/best.pth \
    --delta=0.55


# Refinement learning
CUDA_VISIBLE_DEVICES=${C1},${C2} python scripts/train_refine.py \
    --img_dir=/data_root/WSSS/DRS_log/${CASE}/ \
    --lr=0.0001 \
    --epoch=30 \
    --decay_points='10,20' \
    --logdir=logs/Refine_${CASE} \
    --save_folder=checkpoints/Refine_${CASE} \
    --show_interval=50


# Evaluation pseudo segmentation labels generated by the refinement newtork
# CUDA_VISIBLE_DEVICES=${C1} python scripts/test_refine.py \
#    --checkpoint=checkpoints/Refine_${CASE}/best.pth --img_dir=/dataset/WSSS/VOC2012/


# Pseudo segmentation label generation
CUDA_VISIBLE_DEVICES=${C1} python scripts/pseudo_seg_label_gen.py \
    --img_dir=/data_root/WSSS/DRS_log/${CASE}/ \
    --checkpoint=checkpoints/Refine_${CASE}/best.pth

