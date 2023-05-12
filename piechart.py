from time import time
import numpy as np
import pandas as pd
from matplotlib import image as mpimg
import matplotlib.pyplot as plt

def main():

    methods = np.array(["he", "dhe", "ying", "gammacorrection", "wavelet", "lime", "underwater", "bilateral", "tv", "ldn"])
    headers = np.array(["RMSE", "CNR", "AMBE", "IEM", "PSNR", "EME", "AME", "colourIndex", "averGrad", "Entropy2d"])
    datas = []

    for i in range(1,41):
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
        datas.append(metrics)
    metric = "IEM"
    results = pd.DataFrame(np.zeros((len(methods), 1)), methods, [metric])
    for i in range(40):
        results.loc[datas[i].loc[datas[i][metric] == max(datas[i].loc[:,metric])].index.values[0], metric] += 1
    plot = results.plot.pie(y=metric, figsize=(5, 5), autopct='%1.1f%%')
    plt.show()

if __name__ == "__main__":

    startTime = time()

    main()

    duration = round(time()-startTime, 2)
    print(f"\n### This program was executed in {duration} seconds! ###\n")