import os
from time import time
import numpy as np
from metricsLib import MSE, RMSE, PSNR, EME_AME
import pandas as pd
from matplotlib import image as mpimg

def main():

    methods = np.array(["he", "dhe", "ying", "gammacorrection", "wavelet", "lime", "underwater", "bilateral", "tv", "ldn"])
    headers = np.array(["CNR", "AMBE", "IEM", "PSNR", "EME", "AME", "colourIndex", "averGrad", "Entropy2d"])

    for i in range(1,41):
        print(f"Image {i}")
        image_name = str(i)
        if len(image_name) == 1:
            image_name = '0' + image_name
        if i < 21:
            image_name = image_name + "_test"
        else:
            image_name = image_name + "_training"
        directory = "./image_"+image_name + "/"
        metrics_name = directory+image_name+"_metrics.xlsx"
        metrics = pd.read_excel(metrics_name)
        metrics = metrics.iloc[:,1:]
        metrics = metrics.set_axis(methods)
        for method in methods:
            metrics.loc[method, 'AME'] = -metrics.loc[method, 'AME']
        os.remove(metrics_name)
        #os.remove(directory+image_name+"_metrics.tex")
        metrics.to_excel(metrics_name)

if __name__ == "__main__":

    startTime = time()

    main()

    duration = round(time()-startTime, 2)
    print(f"\n### This program was executed in {duration} seconds! ###\n")