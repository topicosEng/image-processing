import argparse
import os
from matplotlib import image as mpimg
import cv2
from time import time
import enhanceLib
import torchvision

def main(origPath: str, storePath: str, method_name: str):

    if origPath == storePath:
        raise ValueError("Origin and destiny paths mustn't be the same!")
    else:
        if not os.path.isdir(origPath):
            os.makedirs(origPath)
        if not os.path.isdir(storePath):
            os.makedirs(storePath)

    method = enhanceLib.findEnhancer(method_name)

    for imgpath in os.listdir(origPath):
        img = mpimg.imread(origPath+"/"+imgpath)
        print(f"Enhancing image {imgpath}\n")
        img_new = method(img)
        if(method_name == 'ldn'):
            torchvision.utils.save_image(img_new, storePath+"/"+imgpath)
        else:
            mpimg.imsave(storePath+"/"+imgpath, img_new)

if __name__ == "__main__":

    startTime = time()

    parser = argparse.ArgumentParser(description='Convert images of a directory from rgb to gray scale.')

    parser.add_argument(
        '-o', '--origin', required=True, type=str, help='PATH to origin')
    parser.add_argument(
        '-s', '--save', required=True, type=str, help='PATH to save')
    parser.add_argument(
        '-n', '--method', required=True, type=str, help='Method to apply')

    args = parser.parse_args()

    main(args.origin, args.save, args.method)

    duration = round(time()-startTime, 2)
    print(f"\n### This program was executed in {duration} seconds! ###\n")