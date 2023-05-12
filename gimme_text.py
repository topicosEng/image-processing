import argparse
import os
from matplotlib import image as mpimg
import cv2
from time import time
import enhanceLib
import torchvision

def main():

    f = open("./text.txt", "r")
    text = f.read()
    f.close()

    f = open("result.txt", "x")
    f.close()
    f = open("result.txt", "a")
    
    for i in range(20,41):
      print(i)
      f.write(text.replace("%", str(i)).replace("test", "training"))
    f.close()

if __name__ == "__main__":

    startTime = time()

    # parser = argparse.ArgumentParser(description='Convert numbers from text.')

    # parser.add_argument(
    #     '-n', '--number', required=True, type=int, help='Number to apply')

    # args = parser.parse_args()

    main()

    duration = round(time()-startTime, 2)
    print(f"\n### This program was executed in {duration} seconds! ###\n")