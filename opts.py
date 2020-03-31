import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action Recognition Training')

    # train/val/test setting
    parser.add_argument('-b','--batch_size',type=int,default=32,help='batch_size we used')
    parser.add_argument('-lr','--learning_rate',type=float,default=5e-3,help='learning_rate for the optim')
    parser.add_argument('-lr_step',type=int,nargs='+',default=[30,60],help='learning rate reduce epochs')
    parser.add_argument('--epochs',type=int,default=70,help='training epoch')
    parser.add_argument('--val_freq',type=int,default=5,help='validataion every k epochs')
    parser.add_argument('--print_freq',type=int,default=20,help='validataion every k epochs')
    parser.add_argument('--num_frames',type=int,default=3,help='video num frames for trainval')
    parser.add_argument('--num_workers','--j',type=int,default=8,help='dataset loading workers')
    parser.add_argument('--model',type=str,default='g2d_resnet34',help='training model')
    parser.add_argument('--clip_gradient','--gd',type=int,default=20)
    parser.add_argument('--dataset',type=str,default='ss2',help='dataset')
    parser.add_argument('--resume',type=bool,default=False,help='whether to load ckpt')
    parser.add_argument('--rgb_dr',type=float,default=0.5,help='dropout rate')
    parser.add_argument('--image_size',type=int,default=224,help='image_size')
    parser.add_argument('--gpus',type=int,nargs='+',default=None,help='gpus')
    parser.add_argument('--sample_clips',type=int,default=1,help='dataset loading workers')
    parser.add_argument('--modality',type=str,default='rgb',choices=['rgb','flow','fusion'])
    parser.add_argument('--sample',type=str,default='sparse',choices=['sparse','dense'])
    parser.add_argument('--split',type=int,default=0)
    parser.add_argument('--interval',type=int,default=2)

    parser.add_argument('--is_validate',type=bool,default=False)
    parser.add_argument('--partial_bn','--pb',type=bool,default=False)
    parser.add_argument('--use_caffer_pre','--use_cp',type=bool,default=False)
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

    # ccf fusion 
    parser.add_argument('--eval_type',type=str,default='fusion',choices=['fusion','joint'])

    return parser