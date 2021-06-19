import numpy as np
import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src', type=str,
        help='path to source dataset root'
    )
    parser.add_argument(
        '--tar', type=str,
        help='path to target datasetroot ' +\
            '(will be created if it does not exist)'
    )
    return vars(parser.parse_args())

def yolofy_dataset(src, tar):
    # prepare tar dataset structure
    os.makedirs(tar, exist_ok=True)
    for d in ['images', 'labels']:
        os.makedirs(os.path.join(tar, d, 'train'), exist_ok=True)
        os.makedirs(os.path.join(tar, d, 'test'), exist_ok=True)
    ftest = 0.3
    
    classes = sorted([d for d in os.listdir(src) 
                      if os.path.isdir(os.path.join(src, d))])
    
    class2idx = {clss:i for i,clss in enumerate(classes)} # 0-indexed... 
    idx2class = {i:clss for i,clss in enumerate(classes)}
    
    # all images in custom dataset are 640x480
    width, height = 640, 480
    
    for clss in classes:
        with open(os.path.join(src, f'boxes_{clss}.txt')) as f:
            lines = f.read().splitlines()
        
        for line in lines:
            if not line:
                continue
            name = line.split(':')[0]
            idx = name.split('.')[0]
            x1, y1, x2, y2 = map(int, line.split(':')[1].split(','))
            full_src_imgname = os.path.join(src, clss, f'{name}')
            
            roll = np.random.rand()
            train_test = 'train' if roll > ftest else 'test'
            full_tar_imgname = os.path.join(tar, 'images', train_test, f'{clss}_{name}')            
            full_tar_lblname = os.path.join(tar, 'labels', train_test, f'{clss}_{idx}.txt')
            
            # copy image
            shutil.copyfile(full_src_imgname, full_tar_imgname)
            
            # write label
            x_c = (x1 + x2) / (2 * width)
            y_c = (y1 + y2) / (2 * height)
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            with open(full_tar_lblname, 'w') as f:
                f.write(f'{class2idx[clss]} {x_c} {y_c} {w} {h}')

def main():
    args = parse_args()
    yolofy_dataset(args['src'], args['tar'])

if __name__ == '__main__':
    main()