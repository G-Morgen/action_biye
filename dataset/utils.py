import numpy as np 
import os
from sklearn.model_selection import train_test_split, StratifiedKFold

mouse_label = {
        'drink' : ['drink', 'd'],
        'eat' : ['eat', 'e'],
        'groom': ['groomback', 'groom', 'gb', 'g'],
        'hang': ['hang', 'ha'],
        'micromovement': ['head', 'he'],
        'rear': ['rear', 'r'],
        'rest': ['rest', 'rs'],
        'walk': ['walk', 'w']
    }

def read_mouse_file(split):
    mouse_path = "/home/zhujian/dataset/mouse/"
    
    class_index = {
        'd':0, 'drink':0,
        'eat':1, 'e':1,
        'groomback':2, 'groom':2, 'gb':2, 'g':2,
        'hang':3, 'ha':3,
        'head':4, 'he':4,
        'rear':5, 'r':5,
        'rest':6, 'rs':6,
        'walk':7, 'w':7
    }
    all_dir = sorted(os.listdir(mouse_path))
    
    path_list = []
    label_list = []
    for dirs in all_dir:
        for i in os.listdir(os.path.join(mouse_path,dirs)):
            path = os.path.join(mouse_path, dirs, i)
            label = class_index[i.split('_')[1]]
            path_list.append(path)
            label_list.append(label)
    
    skf = StratifiedKFold(2, shuffle=True, random_state=1234)
    X = np.array(path_list)
    y = np.array(label_list)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if i == split:
            break
    
    train_list = list(zip(X_train, y_train))
    test_list = list(zip(X_test, y_test))

    return train_list, test_list


def read_larva_file(split):
    mouse_path = "/home/zhujian/dataset/Larva/"
    mouse_path_2 = "/home/zhujian/dataset/Larva_new/"
    class_index = {
        'crawling':0,
        'head turn':1,
    }
    all_dir = sorted([os.path.join(mouse_path,i) for i in os.listdir(mouse_path)]) \
                + sorted([os.path.join(mouse_path_2, i) for i in os.listdir(mouse_path_2)])
    
    path_list = []
    label_list = []
    for dirs in all_dir:
        for i in os.listdir(dirs):
            path = os.path.join(dirs, i)
            label = class_index[dirs.split('/')[-1]]
            path_list.append(path)
            label_list.append(label)
    
    skf = StratifiedKFold(2, shuffle=True, random_state=1234)
    X = np.array(path_list)
    y = np.array(label_list)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if i == split:
            break
    
    train_list = list(zip(X_train, y_train))
    test_list = list(zip(X_test, y_test))

    return train_list, test_list

def read_file(split):
        
        train_1 = "./dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt"
        train_2 = "./dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist02.txt"
        train_3 = "./dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist03.txt"

        test_1 = "./dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt"
        test_2 = "./dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist02.txt"
        test_3 = "./dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist03.txt"

        classInd = "./dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt"
        train = [train_1, train_2, train_3]
        test = [test_1, test_2, test_3]
        train = np.genfromtxt(train[split],dtype='U')
        test = np.genfromtxt(test[split],dtype='U')
        train_list = []
        test_list = []

        all_classes = np.genfromtxt(classInd,'U')
        classes_index = dict(zip(all_classes[:,1],all_classes[:,0]))
        index_classes = dict(zip(all_classes[:,0],all_classes[:,1]))
        for i in train:
            c = i[0].split('/')[0]
            if c in all_classes[:,1]:
                train_list.append([i[0], int(classes_index[c]) - 1])

        for i in test:
            c = i.split('/')[0]
            if c in all_classes[:,1]:
                test_list.append([i, int(classes_index[c]) - 1])
        
        return train_list, test_list


def read_hmdb_file(splits):

        all_classes = np.genfromtxt('./dataset/hmdb51_full.txt',dtype='U')

        train_list = []
        test_list = []
        classes_index = dict(zip(all_classes[:,1],all_classes[:,0]))
        index_classes = dict(zip(all_classes[:,0],all_classes[:,1]))
        for i in all_classes:
            a = os.path.join("./dataset/test_train_splits/testTrainMulti_7030_splits/",i[1]+'_test_split%d.txt' % int(splits + 1))

            with open(a,'r') as f:
                for line in f.readlines():
                    path = line.split(' ')[0]
                    idx = line.split(' ')[1]
                    if int(idx) == 1:
                        train_list.append([i[1]+'/'+path, int(classes_index[i[1]])])
                    elif int(idx) == 2:
                        test_list.append([i[1]+'/'+path, int(classes_index[i[1]])])

        return train_list, test_list

def run_check_mouse():
    a,b = read_mouse_file(0)
    print(a, b)
    print(len(a), len(b))


def ucf_label():
    from matplotlib import pyplot as plt
    train_list, test_list = read_file(0)
    classInd = "./dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt"
    all_classes = np.genfromtxt(classInd,'U')
    classes_index = dict(zip(all_classes[:,1],all_classes[:,0]))
    index_classes = dict(zip(all_classes[:,0],all_classes[:,1]))

    a = []
    for i in train_list:
        a.append(i[1])
    for i in test_list:
        a.append(i[1])

    b,c = np.unique(a,return_counts=True)
    print(c.min(),c.max(),c.mean())
    plt.bar(b,c)
    plt.xlabel('Label')
    plt.ylabel('Number of image per label')
    plt.show()


def mouse_label():
    from matplotlib import pyplot as plt
    train_list, test_list = read_mouse_file(11)

    a = []
    b = []
    for i in train_list:
        a.append(i[1])
    for i in test_list:
        b.append(i[1])

    c,d = np.unique(a,return_counts=True)
    e,f = np.unique(b,return_counts=True)
    
    print(c,d)
    print(d.min(),d.max(),d.mean())
    print(e,f)
    print(f.min(),f.max(),f.mean())
    

def Larva_label():
    from matplotlib import pyplot as plt
    train_list, test_list = read_larva_file(0)
    print(len(train_list), len(test_list))
    print(train_list)
    a = []
    b = []
    for i in train_list:
        a.append(i[1])
    for i in test_list:
        b.append(i[1])

    c,d = np.unique(a,return_counts=True)
    e,f = np.unique(b,return_counts=True)
    
    print(c,d)
    print(d.min(),d.max(),d.mean())
    print(e,f)
    print(f.min(),f.max(),f.mean())

if __name__ == "__main__":
    # run_check_mouse()
    # mouse_label()
    Larva_label()