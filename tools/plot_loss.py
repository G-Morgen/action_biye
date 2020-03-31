from matplotlib import pyplot as plt 
from matplotlib.pyplot import MultipleLocator
import numpy as np 

tsn_resnet34_valid = np.genfromtxt("/home/zhujian/action_recognition/pytorch/action_biye/logdir/ucf101/flow/resnet34/valid_log.txt",dtype='U')
tsn_resnet34_train = np.genfromtxt("/home/zhujian/action_recognition/pytorch/action_biye/logdir/ucf101/flow/resnet34/train_log.txt",dtype='U')

i3d_resnet34_valid = np.genfromtxt("/home/zhujian/action_recognition/pytorch/action_biye/logdir/ucf101/rgb/i3d_resnet34/valid_log.txt",dtype='U')
i3d_resnet34_train = np.genfromtxt("/home/zhujian/action_recognition/pytorch/action_biye/logdir/ucf101/rgb/i3d_resnet34/train_log.txt",dtype='U')

i3d_resnet50_valid = np.genfromtxt("/home/zhujian/action_recognition/pytorch/action_biye/logdir/ucf101/rgb/i3d_resnet50/valid_log.txt",dtype='U')
i3d_resnet50_train = np.genfromtxt("/home/zhujian/action_recognition/pytorch/action_biye/logdir/ucf101/rgb/i3d_resnet50/train_log.txt",dtype='U')

tsn_resnet34_valid_loss = list(tsn_resnet34_valid[:,2])
tsn_resnet34_train_loss = list(tsn_resnet34_train[:,2])

i3d_resnet34_valid_loss = list(i3d_resnet34_valid[:,2])
i3d_resnet34_train_loss = list(i3d_resnet34_train[:,2])

i3d_resnet50_valid_loss = list(i3d_resnet50_valid[:,2])
i3d_resnet50_train_loss = list(i3d_resnet50_train[:,2])

tsnresnet34valid_loss = [float(i[:-1]) for i in tsn_resnet34_valid_loss]
tsnresnet34train_loss = [float(i[:-1]) for i in tsn_resnet34_train_loss]

i3dresnet34valid_loss = [float(i[:-1]) for i in i3d_resnet34_valid_loss]
i3dresnet34train_loss = [float(i[:-1]) for i in i3d_resnet34_train_loss]

i3dresnet50valid_loss = [float(i[:-1]) for i in i3d_resnet50_valid_loss]
i3dresnet50train_loss = [float(i[:-1]) for i in i3d_resnet50_train_loss]

a_tsn = range(1,len(tsn_resnet34_valid_loss)+1) 
b_tsn = range(1,len(tsn_resnet34_train_loss)+1) 

a = range(1,len(i3d_resnet34_valid_loss)+1) 
b = range(1,len(i3d_resnet34_train_loss)+1) 

c = range(1,len(i3d_resnet50_valid_loss)+1) 
d = range(1,len(i3d_resnet50_train_loss)+1) 

plt.plot(a_tsn, tsnresnet34valid_loss, color='g', linestyle='dashed')
plt.plot(b_tsn, tsnresnet34train_loss, label='TSN_resnet34',color='g')


plt.plot(a, i3dresnet34valid_loss, color='b', linestyle='dashed')
plt.plot(b, i3dresnet34train_loss, label='I3D_resnet34',color='b')

plt.plot(c, i3dresnet50valid_loss, color='r', linestyle='dashed')
plt.plot(d, i3dresnet50train_loss, label='I3D_resnet50',color='r')

# plt.axis([0,len(valid_epoch),0,5])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.tick_params(axis='both',which='major',labelsize=6)

x_major_locator=MultipleLocator(5)
#把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(1)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(0,60)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(0,5)
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
plt.legend()
plt.show()