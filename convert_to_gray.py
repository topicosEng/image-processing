import argparse
import os
from matplotlib import image as mpimg
import cv2
from time import time

def main(origPath: str, storePath: str):

    if origPath == storePath:
        raise ValueError("Origin and destiny paths mustn't be the same!")
    else:
        if not os.path.isdir(origPath):
            os.makedirs(origPath)
        if not os.path.isdir(storePath):
            os.makedirs(storePath)

    for imgpath in os.listdir(origPath):
        img = mpimg.imread(origPath+"/"+imgpath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mpimg.imsave(storePath+"/"+imgpath, img_gray)

if __name__ == "__main__":

    startTime = time()

    parser = argparse.ArgumentParser(description='Convert images of a directory from rgb to gray scale.')

    parser.add_argument(
        '-o', '--origin', required=True, type=str, help='PATH to origin')
    parser.add_argument(
        '-s', '--save', required=True, type=str, help='PATH to save')

    args = parser.parse_args()

    main(args.origin, args.save)

    duration = round(time()-startTime, 2)
    print(f"\n### This program was executed in {duration} seconds! ###\n")