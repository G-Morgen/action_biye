# Action-BiYe

This is a reimplementation of CCF-Net and DFL-Net in PyTorch. 

## Training

To train a new model, use the `train.py` script.

The command to reproduce the original TSN experiments of RGB modality on UCF101 can be 

```bash
python main.py --dataset ucf101  \
   --model resnet50 --num_frames 5 \
   --gd 20 --lr 0.001 -lr_step 30 60 --epochs 80 \
   -b 16 --dropout 0 \
   --modality rgb
```

For flow models:

```bash
python main.py --dataset ucf101  \
   --model resnet50 --num_frames 5 \
   --gd 20 --lr 0.002 -lr_step 80 160 --epochs 200 \
   -b 16 --dropout 0.8 \
   --modality rgb
```

For CCF models:

```bash
python train_ccfnet.py --dataset ucf101  \
   --model resnet50 --num_frames 5 \
   --gd 20 --lr 0.0001 -lr_step 20 30 --epochs 400 \
   -b 16 --dropout 0.8 \
   --modality fusion
```

For DFL models

```bash
python train.py --dataset ucf101  \
   --model dfl_resnet50 --num_frames 5 \
   --gd 20 --lr 0.0001 -lr_step 20 30 --epochs 400 \
   -b 16 --dropout 0.8 \
   --modality rgb
```

## Testing

After training, there will checkpoints saved by pytorch, for example `./4T/zhujian/ckpt/resnet50/rgb/model.ckpt`.

Use the following command to test its performance in the standard TSN testing protocol:

```bash
python val.py --dataset ucf101 --modality rgb \
   --model resnet50 --num_frames 25 

```

