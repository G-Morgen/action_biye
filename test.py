import torch
import torch.nn as nn 
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn

import pandas as pd
import os 
import numpy as np 
from dataset import ss
# from dataset import ss_
from dataset import jester_test
from get_model import get_model

import time 
from opts import arg_parser
parser = arg_parser()
args = parser.parse_args()

best_prec1 = 0
torch.backends.cudnn.benchmark = True

def main():
    global best_prec1
    batch_size = args.batch_size
    lr = args.learning_rate
    epochs = args.epochs
    val_freq = args.val_freq
    num_frames = args.num_frames
    num_workers = args.num_workers
    sample_clips = args.sample_clips
    
    if os.path.exists(os.path.join('ckpt',args.dataset,args.model,'last')) is False:
        os.makedirs(os.path.join('ckpt',args.dataset,args.model,'last'))
    if os.path.exists(os.path.join('ckpt',args.dataset,args.model,'best')) is False:
        os.makedirs(os.path.join('ckpt',args.dataset,args.model,'best'))
    if os.path.exists(os.path.join('logdir',args.dataset,args.model)) is False:
        os.makedirs(os.path.join('logdir',args.dataset,args.model))

    if args.dataset == 'ss2':
        dataset, num_classes = ss.make_data(num_frames=num_frames,batch_size=args.batch_size,num_workers=8)
        input_size = 112
    elif args.dataset == 'jester':
        dataset, num_classes = jester_test.make_data(mode='test',num_frames=num_frames,sample_times=sample_clips,batch_size=args.batch_size,num_workers=8)
        input_size = 112

    model = get_model(args.model, num_classes, 0, input_size, args.num_frames)

    model = nn.DataParallel(model,device_ids=args.gpus).cuda()
    
    print('=> loading checkpoint {}'.format(os.path.join('ckpt',args.dataset,args.model,'best')))
    ckpt = torch.load(os.path.join('ckpt',args.dataset,args.model,'best','model.ckpt'))
    model.load_state_dict(ckpt)
    
    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    validate(dataset, model, criterion)

        
def validate(data_loader, model, criterion):

    model.eval()

    batch_time = AverageMeter()

    num2label = data_loader.dataset.label
    '''
            result.append(str(test_name[0]) + ';'+ video_predict_class[-1])                                               
            k = {'a':result}
            dataframe = pd.DataFrame(k)
            # dataframe = dataframe['a'].map(str) + dataframe['b']
            dataframe.to_csv(os.path.join(self.train_log_path,'test.csv'),index=False,sep=';',header=False
    '''
    id_list = []
    label = []
    with torch.no_grad():

        end = time.time()
        for i, sample in enumerate(data_loader):
            
            inputs = sample['data']
            ids = sample['id'].cpu().numpy()
            id_list.extend(ids)

            b,sc, c, t, h, w = inputs.size()
            input_var = inputs.cuda(async=True).view(b*sc, c, t, h, w)
            
            output = model(input_var)

            output = output.view(b,sc,-1)
            output = torch.mean(output, dim=1)

            a = output.argmax(dim=1)
            d = a.cpu().numpy()
            for j in d:
                label.append(num2label[j])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(data_loader), batch_time=batch_time
                    )))
    
    result = []
    for a,b in zip(id_list,label):
        result.append(str(a)+";"+b.strip())
    
    k = {'a':result}
    dataframe = pd.DataFrame(k)
    dataframe.to_csv(os.path.join('logdir',args.dataset,args.model,'test.csv'),index=False,sep=';',header=False)


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