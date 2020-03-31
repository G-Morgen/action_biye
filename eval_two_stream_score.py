import numpy as np 
from opts import arg_parser
import os
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


parser = arg_parser()
args = parser.parse_args()

if args.dataset == 'mouse':
    classes = ['drink', 'eat', 'groom', 'hang', 'head', 'rear', 'rest', 'walk']
elif args.dataset == 'hmdb51':
    all_classes = np.genfromtxt('./dataset/hmdb51_full.txt',dtype='U')
    classes = all_classes[:,1]
elif args.dataset == 'ucf101':
    classInd = "./dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt"
    all_classes = np.genfromtxt(classInd,'U')
    classes = all_classes[:,1]



def plot_confusion_matrix(gt, pred, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(gt, pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

rgb = np.load(os.path.join('logdir',args.dataset,str(args.split),'rgb',args.model,'score.npy'))
flow = np.load(os.path.join('logdir',args.dataset,str(args.split),'flow',args.model,'score.npy'))
gt = np.load(os.path.join('logdir',args.dataset,str(args.split),'rgb',args.model,'gt.npy'))

gt = np.array(gt)
fusion = np.array(rgb) + np.array(flow) * 1.5

rgb_score = sklearn.metrics.accuracy_score(gt, rgb.argmax(-1))
print(f'rgb score {rgb_score}')
# plt.figure()
# plot_confusion_matrix(gt, rgb.argmax(-1), classes=classes)
# plt.savefig('figure/rgb_cm.jpg',bbox_inches='tight',pad_inches=0.0)

flow_score = sklearn.metrics.accuracy_score(gt, flow.argmax(-1))
print(f'flow score {flow_score}')
# plt.figure()
# plot_confusion_matrix(gt, flow.argmax(-1), classes=classes)
# plt.savefig('figure/flow_cm.jpg',bbox_inches='tight',pad_inches=0.0)

fusion_score = sklearn.metrics.accuracy_score(gt, fusion.argmax(-1))
print(f'fusion score {fusion_score}')
# plt.figure()
# plot_confusion_matrix(gt, fusion.argmax(-1), classes=classes)
# plt.savefig('figure/fusion_cm.jpg',bbox_inches='tight',pad_inches=0.0)

# score = sklearn.metrics.classification_report(gt, fusion.argmax(-1), digits=4)
# print(score)
# plt.show()