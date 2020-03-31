import torch
import torch.nn as nn 
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
import torchvision

import os 
import numpy as np 
from dataset import ucf101
from dataset import hmdb51
from dataset import mouse
from dataset import Larva   

from get_model import get_model
import ccfnet
import time 

from opts import arg_parser
parser = arg_parser()
args = parser.parse_args()

best_prec1 = 0

ckpt_path = '/4T/zhujian/ckpt'

def build_dir():
    if os.path.exists(os.path.join('logdir',args.dataset,str(args.split),args.modality,args.model)) is False:
        os.makedirs(os.path.join('logdir',args.dataset,str(args.split),args.modality,args.model))
    if os.path.exists(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'last')) is False:
        os.makedirs(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'last'))
    if os.path.exists(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'best')) is False:
        os.makedirs(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'best'))

def load_pretrained_model():
    rgb_ckpt = torch.load(os.path.join(ckpt_path,args.dataset,str(args.split),'rgb',args.model,'best','model.ckpt'))
    flow_ckpt = torch.load(os.path.join(ckpt_path,args.dataset,str(args.split),'flow',args.model,'best','model.ckpt'))

    rgb = {}
    flow = {}
    for k,v in rgb_ckpt.items():
        rgb[k[7:]] = v
    for k,v in flow_ckpt.items():
        flow[k[7:]] = v
    
    return rgb, flow

build_dir()

def main():
    global best_prec1
    batch_size = args.batch_size
    lr = args.learning_rate
    epochs = args.epochs
    val_freq = args.val_freq
    num_frames = args.num_frames
    num_workers = args.num_workers
    
    assert args.modality == 'fusion'
    if args.dataset == 'ucf101':
        train_dataset, valid_dataset, num_classes = ucf101.make_data(num_frames=num_frames,
                batch_size=args.batch_size, model=args.model, modality=args.modality, split=args.split,
                sample=args.sample,num_workers=args.num_workers)
        input_size = 224
    elif args.dataset == 'hmdb51':
        train_dataset, valid_dataset, num_classes = hmdb51.make_data(num_frames=num_frames,
                batch_size=args.batch_size, model=args.model, modality=args.modality, split=args.split,
                sample=args.sample,num_workers=args.num_workers)
        input_size = 224
    elif args.dataset == 'mouse':
        train_dataset, valid_dataset, num_classes = mouse.make_data(num_frames=num_frames,
                batch_size=args.batch_size, model=args.model, modality=args.modality, split=args.split,
                sample=args.sample,num_workers=args.num_workers)
        input_size = 224
    elif args.dataset == 'Larva':
        train_dataset, valid_dataset, num_classes = Larva.make_data(num_frames=num_frames,
                batch_size=args.batch_size, model=args.model, modality=args.modality, split=args.split,
                sample=args.sample,num_workers=args.num_workers)
        input_size = 224

    print(f'Dataset {args.dataset},use split {args.split}')
    model = ccfnet.ccfnet(args.model, num_classes,  input_size, args.num_frames, dropout=args.rgb_dr)
    
    rgb_ckpt, flow_ckpt = load_pretrained_model()
    model.rgb_model.load_state_dict(rgb_ckpt)
    model.flow_model.load_state_dict(flow_ckpt)
    print(f'load pretrained {args.model} of {args.dataset} successfully')

    model = nn.DataParallel(model,device_ids=args.gpus).cuda()
    param = model.parameters()
    param = filter(lambda p: p.requires_grad, param)

    if args.resume:
        if os.path.exists(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'last')):
            print('=> loading checkpoint {}'.format(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'last')))
            ckpt = torch.load(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'last','model.ckpt'))
            model.load_state_dict(ckpt)
        else:
            print('=> no checkpoint found at {}'.format(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'last')))
    
    torch.backends.cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.is_validate:
        prec1 = validate(valid_dataset, model, criterion, 0)
        return 

    optimizer = torch.optim.SGD(
        param,
        args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_step)

        train(train_dataset, model, criterion, optimizer, epoch)

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:

            prec1 = validate(valid_dataset, model, criterion, epoch)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if is_best:
                torch.save(model.state_dict(),os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'best','model.ckpt'))
            torch.save(model.state_dict(),os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'last','model.ckpt'))


def train(data_loader, model, criterion, optim, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if 'i3d' not in args.model and args.partial_bn :
        model.module.partialBN(True)

    model.train()

    end = time.time()
    for i, sample in enumerate(data_loader):

        data_time.update(time.time() - end)

        label = sample['label_num']
        rgb = sample['rgb']
        flow = sample['flow']

        rgb = rgb.cuda(async=True)
        flow = flow.cuda(async=True)
        label = label.cuda(async=True)

        r_out, f_out, fusion_out = model(rgb, flow)
        if args.eval_type == 'fusion':
            output = fusion_out
        else:
            output = r_out + f_out + fusion_out
        loss = criterion(output, label)

        prec_1, prec_5 = accuracy(output.data, label, topk=(1,2))
        losses.update(loss.item(), rgb.size(0))
        top1.update(prec_1.data, rgb.size(0))
        top5.update(prec_5.data, rgb.size(0))
        
        optim.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
                # print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, lr=optim.param_groups[-1]['lr'])))

    print(f'one epoch time {batch_time.sum}s')
    if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
    
        template = "Epoch:{}, Loss: {:.2f}, Prec1 :{:.2f}\n"
        with open(os.path.join('logdir',args.dataset,str(args.split),args.modality,args.model,'train_log.txt'),'a') as f:
            f.writelines(
                template.format(
                    epoch, losses.avg, top1.avg
                )
            )


def validate(data_loader, model, criterion, epoch):

    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():

        end = time.time()
        for i, sample in enumerate(data_loader):
            
            label = sample['label_num']
            rgb = sample['rgb']
            flow = sample['flow']

            rgb = rgb.cuda(async=True)
            flow = flow.cuda(async=True)
            label = label.cuda(async=True)

            r_out, f_out, fusion_out = model(rgb, flow)
            if args.eval_type == 'fusion':
                output = fusion_out
            else:
                output = r_out + f_out + fusion_out
            loss = criterion(output, label)

            prec_1, prec_5 = accuracy(output.data, label, topk=(1,2))
            losses.update(loss.item(), rgb.size(0))
            top1.update(prec_1.data, rgb.size(0))
            top5.update(prec_5.data, rgb.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    i, len(data_loader), batch_time=batch_time, loss=losses,
                    top1=top1)))

        template = "Epoch:{}, Loss: {:.2f}, Prec1 :{:.2f}\n"
        with open(os.path.join('logdir',args.dataset,str(args.split),args.modality,args.model,'valid_log.txt'),'a') as f:
            print(
                template.format(
                    epoch, losses.avg, top1.avg
                )
            )
            f.writelines(
                template.format(
                    epoch, losses.avg, top1.avg
                )
            )

        return top1.avg


class LabelRecord(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.record = []

    def update(self, val):
        self.record.extend(val)
         

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optim, epoch, lr_step):

    decay = 0.1 ** (sum(epoch >= np.array(lr_step)))
    lr = args.learning_rate * decay
    decay = args.weight_decay
    for param_group in optim.param_groups:
        # param_group['lr'] = lr
        if args.partial_bn:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']
        else:
            param_group['lr'] = lr
    

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100 / batch_size))
    
    return res

if __name__ == "__main__":
    main()