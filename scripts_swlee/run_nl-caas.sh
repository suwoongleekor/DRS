#sh ./scripts_swlee/run_nl-caas.sh DRS_NL_CAAS2 0 1 13570 0.3
#sh ./scripts_swlee/run_nl-caas.sh DRS_NL_CAAS3 2 3 13571 0.1
#sh ./scripts_swlee/run_nl-caas.sh DRS_NL_CAAS4 4 5 13572 0.03
#sh ./scripts_swlee/run_nl-caas.sh DRS_NL_CAAS5 6 7 13573 0.01


IMG_DIR_NAME=/data_root/WSSS/DRS_log/$1/
LOG_DIR_NAME=$1
C1=$2
C2=$3
VISPORT=$4
LMDA=$5


## Training Classifier with DRS-CAAS
#CUDA_VISIBLE_DEVICES=${C1},${C2} python scripts/train_cls.py \
#    --img_dir=${IMG_DIR_NAME} \
#    --lr=0.001 \
#     --epoch=15 \
#    --decay_points='5,10' \
#    --delta=0.55 \
#    --logdir=logs/${LOG_DIR_NAME} \
#    --save_folder=checkpoints/${LOG_DIR_NAME} \
#    --show_interval=50 \
#    --loss_cls_aware_mode='cls_aware' \
#    --lambda_cls=1. \
#    --lambda_cls_aware=${LMDA}
#
## Evaluation pseudo segmentation labels generated by the classifier
# CUDA_VISIBLE_DEVICES=${C1} python scripts/test_cls.py \
#     --checkpoint=checkpoints/${LOG_DIR_NAME}/best.pth \
#     --img_dir=${IMG_DIR_NAME} \
#     --delta=0.55
#
#
## Generating localization maps for the refinement learning
#CUDA_VISIBLE_DEVICES=${C1} python scripts/localization_map_gen.py \
#    --img_dir=${IMG_DIR_NAME} \
#    --checkpoint=checkpoints/${LOG_DIR_NAME}/best.pth \
#    --delta=0.55
#
#
## Refinement learning
#CUDA_VISIBLE_DEVICES=${C1},${C2} python scripts/train_refine.py \
#    --img_dir=${IMG_DIR_NAME} \
#    --lr=0.0001 \
#    --epoch=30 \
#    --decay_points='10,20' \
#    --logdir=logs/Refine_${LOG_DIR_NAME} \
#    --save_folder=checkpoints/Refine_${LOG_DIR_NAME} \
#    --show_interval=50
#
#
## Evaluation pseudo segmentation labels generated by the refinement network
# CUDA_VISIBLE_DEVICES=${C1} python scripts/test_refine.py \
#    --checkpoint=checkpoints/Refine_${LOG_DIR_NAME}/best.pth \
#    --img_dir=${IMG_DIR_NAME}
#
#
## Pseudo segmentation label generation
#CUDA_VISIBLE_DEVICES=${C1} python scripts/pseudo_seg_label_gen.py \
#    --img_dir=${IMG_DIR_NAME} \
#    --checkpoint=checkpoints/Refine_${LOG_DIR_NAME}/best.pth


cd DeepLabV3Plus-Pytorch

ROOT=${IMG_DIR_NAME}
MODEL=deeplabv3plus_resnet101 # deeplabv3plus_resnet101, deeplabv3_resnet101
ITER=20000
BATCH=32
LR=0.05

# training with 2 GPUs
python main.py --data_root ${ROOT} --model ${MODEL} --gpu_id ${C1},${C2} --amp --total_itrs ${ITER} --batch_size ${BATCH} --lr ${LR}  --crop_val --enable_vis --vis_port ${VISPORT} --ckpt ${ROOT}checkpoints/best_${MODEL}_voc_os16.pth
# evalutation with crf
python eval.py --gpu_id ${C1},${C2} --data_root ${ROOT} --model ${MODEL}  --val_batch_size 16  --ckpt ${ROOT}checkpoints/best_${MODEL}_voc_os16.pth --crop_val


