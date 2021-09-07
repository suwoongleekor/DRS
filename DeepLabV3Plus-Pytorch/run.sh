ROOT=/data_root/WSSS/VOC2012/
MODEL=deeplabv3plus_resnet101 # deeplabv3plus_resnet101, deeplabv3_resnet101
ITER=20000
BATCH=32
LR=0.05
VISPORT=13570
CASE=DRS_non_learnable # DRS_non_learnable, DRS_NL_CAAS2


# training with 2 GPUs
CUDA_VISLBLE_DEVICES=0,1 python main.py --data_root ${ROOT} --model ${MODEL} --gpu_id 0,1 --amp --total_itrs ${ITER} --batch_size ${BATCH} --lr ${LR}  --crop_val --enable_vis --vis_port ${VISPORT}


# evalutation with crf
CUDA_VISIBLE_DEVICES=0,1 python eval.py --gpu_id 0,1 --data_root ${ROOT} --model ${MODEL}  --val_batch_size 16  --ckpt checkpoints/best_${MODEL}_voc_os16.pth  --crop_val


## test
#python main.py --data_root ${ROOT} --model ${MODEL} --enable_vis --vis_port ${VISPORT} --gpu_id 0,1 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/${CASE}/best_${MODEL}_voc_os16.pth --test_only --save_val_results
