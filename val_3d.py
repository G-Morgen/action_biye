import torch
import torch.nn as nn 
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os 
import numpy as np 
from dataset import ucf101_val3d
from dataset import hmdb51_val3d
from get_model import get_model

import sklearn.metrics
import time 
from opts import arg_parser

parser = arg_parser()
args = parser.parse_args()

best_prec1 = 0
torch.backends.cudnn.benchmark = True
ckpt_path = '/4T/zhujian/ckpt'

def main():
    global best_prec1
    batch_size = args.batch_size
    lr = args.learning_rate
    epochs = args.epochs
    val_freq = args.val_freq
    num_frames = args.num_frames
    num_workers = args.num_workers
    sample_clips = args.sample_clips
    

    if args.dataset == 'ucf101':
        dataset, num_classes = ucf101_val3d.make_data(num_frames=num_frames,
                    sample='dense', model=args.model, modality=args.modality,
                    batch_size=args.batch_size,num_workers=4)
        input_size = 224
    elif args.dataset == 'hmdb51':
        dataset, num_classes = hmdb51_val3d.make_data(num_frames=num_frames,
                    sample='dense', model=args.model, modality=args.modality,
                    batch_size=args.batch_size,num_workers=4)
        input_size = 224

    model = get_model(args.model, args.modality, num_classes, 0, input_size, args.num_frames)
    model = nn.DataParallel(model,device_ids=args.gpus).cuda()

    print('=> loading checkpoint {}'.format(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'best')))
    ckpt = torch.load(os.path.join(ckpt_path,args.dataset,str(args.split),args.modality,args.model,'best','model.ckpt'))
    model.load_state_dict(ckpt)
    
    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    validate(dataset, model, criterion)

        
def validate(data_loader, model, criterion):

    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    recall = AverageMeter()

    target_label = LabelRecord()
    pred_label = LabelRecord()
    score = []
    gt = []

    with torch.no_grad():

        end = time.time()
        for i, sample in enumerate(data_loader):
            
            label = sample['label_num']
            inputs = sample['data']
            b, sc, c, t, h, w = inputs.size()
            input_var = inputs.cuda(async=True).view(b*sc, c, t, h, w)
            label = label.cuda(async=True)

            output = model(input_var)
            
            output = output.view(b,sc,-1)
            output = torch.softmax(output, dim=-1)
            output = torch.mean(output, dim=1)

            prec_1, prec_5 = accuracy(output.data, label, topk=(1,5))
            pred_label.update(output.argmax(-1).cpu().numpy())
            target_label.update(label.cpu().numpy())
            recall_score = sklearn.metrics.recall_score(target_label.record, pred_label.record, average='macro')

            score.extend(output.data.cpu().numpy())
            gt.extend(label.data.cpu().numpy())

            top1.update(prec_1.data, inputs.size(0))
            top5.update(prec_5.data, inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Recall@1  ({recall:.3f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(data_loader), batch_time=batch_time, recall=recall_score * 100,
                    top1=top1, top5=top5)))

        np.save(os.path.join('logdir',args.dataset,str(args.split),args.modality,args.model,'score'),score)
        np.save(os.path.join('logdir',args.dataset,str(args.split),args.modality,args.model,'gt'),gt)
        template = "Prec1 :{:.2f}, Prec5 :{:.2f}, Recall:{:.2f}, Sample Clips:{:2d}, num_frames: {:2d}\n"
        with open(os.path.join('logdir',args.dataset,str(args.split),args.modality,args.model,'valid_result.txt'),'a') as f:
            print(
                template.format(
                    top1.avg, top5.avg, recall_score * 100, args.sample_clips, args.num_frames
                )
            )
            f.writelines(
                template.format(
                    top1.avg, top5.avg, recall_score * 100, args.sample_clips, args.num_frames
                )
            )

        d = sklearn.metrics.classification_report(target_label.record, pred_label.record, digits=4)
        with open(os.path.join('logdir',args.dataset,str(args.split),args.modality,args.model,'report.txt'),'a') as f:
            f.writelines(d)
            print(d)
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

    lr = args.learning_rate * 0.1 ** (sum(epoch >= np.array(lr_step)))

    for param_group in optim.param_groups:
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