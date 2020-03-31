
import os
import threading
import glob
import shutil
from tqdm import tqdm
import multiprocessing

NUM_THREADS = 5
VIDEO_ROOT = "/4T/zhujian/dataset/Larva/Larva_new"        # Downloaded webm videos
FRAME_ROOT = "/home/zhujian/dataset/Larva_new/"  # Directory for extracted frames


def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def extract(video_path, out_path, tmpl='%06d.jpg'):
    
    cmd = 'ffmpeg -i \"{}\" -threads 1 -vf scale=-1:256 -q:v 0 \"{}/%06d.jpg\"  -loglevel error'.format(video_path,
                                                                                             out_path)
                                                                                        
    os.system(cmd)


def target(video_list):
    for video in tqdm(video_list):
        # name = video.split('.')[-2]
        video_classes = video
        a = glob.glob(os.path.join(VIDEO_ROOT, video_classes, '*'))
        for i in a:
            if os.path.isdir(i):
                continue
            name = i.split('/')[-1][:-4]
            # print(name)
            try:
                os.makedirs(os.path.join(FRAME_ROOT, video_classes, name))
            except:
                print('exsit path %s' % os.path.join(FRAME_ROOT, video_classes, name) )
            extract(i, os.path.join(FRAME_ROOT, video_classes, name))


if __name__ == '__main__':
    if not os.path.exists(VIDEO_ROOT):
        raise ValueError('Please download videos and set VIDEO_ROOT variable.')
    if not os.path.exists(FRAME_ROOT):
        os.makedirs(FRAME_ROOT)

    video_list = os.listdir(VIDEO_ROOT)
    print(len(video_list))
    target(video_list)
    