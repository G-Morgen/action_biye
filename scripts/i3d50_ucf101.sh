CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model i3d_resnet50 --num_frames 16 -b 64 --dataset ucf101 --learning_rate 1e-3 --epoch 50 -lr_step 20 40 --sample dense --rgb_dr 0.8 --val_freq 1 --num_workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model i3d_resnet34 --num_frames 16 -b 64 --dataset ucf101 --learning_rate 1e-3 --epoch 50 -lr_step 20 40 --sample dense --rgb_dr 0.8 --val_freq 1 --num_workers 8

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model i3d_resnet50 --num_frames 16 -b 64 --dataset hmdb51 --learning_rate 1e-3 --epoch 50 -lr_step 20 40 --sample dense --rgb_dr 0.8 --val_freq 1 --num_workers 8
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model i3d_resnet34 --num_frames 16 -b 64 --dataset hmdb51 --learning_rate 1e-3 --epoch 50 -lr_step 20 40 --sample dense --rgb_dr 0.8 --val_freq 1 --num_workers 8
