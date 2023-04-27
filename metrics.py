import argparse
import os
from matplotlib import image as mpimg
from metricsLib import MSE, RMSE, CNR, AMBE, IEM, PSNR, EME_AMEE, colourIndex, averGrad, calcEntropy2d 
import numpy as np
from time import time
import pandas as pd

def calculateImageMetrics(origPath: str, storePath: str):
    
    i = 0
    headers = np.array(["RMSE", "CNR", "AMBE", "IEM", "PSNR", "EME", "AMEE", "colourIndex", "averGrad", "Entropy2d"])
    metrics = np.zeros((len(os.listdir(origPath)), len(headers)))

    for imgind1 in os.listdir(origPath):
        imgind2 = os.listdir(storePath)[i]
        print(f"Taking metrics of images {imgind1} and {imgind2}\n")
        if i == 0:
            names = np.array(imgind1)
        else:
            names = np.append(names, imgind1)
        ## faz array com todas as métricas, cada coluna(posição) é uma métrica,
        ##  soma a cada iteração, depois tira a média
        
        img1 = mpimg.imread(origPath+"/"+imgind1)
        img2 = mpimg.imread(storePath+"/"+imgind2)

        print(f"Image {imgind1} rmse:")
        rmse = RMSE(MSE(img1, img2))
        metrics[i,0] = rmse
        print(f"{rmse}\n")
        print(f"Image {imgind1} psnr:")
        psnr = PSNR(img2, MSE(img1, img2))
        metrics[i,4] = psnr
        print(f"{psnr}\n")
        
        print(f"Image {imgind1} cnr:")
        cnr = CNR(img1, img2)
        metrics[i,1] = cnr
        print(f"{cnr}\n")
        print(f"Image {imgind1} ambe:")
        ambe = AMBE(img1, img2)
        metrics[i,2] = ambe
        print(f"{ambe}\n")
        print(f"Image {imgind1} iem:")
        iem = IEM(img1, img2)
        metrics[i,3] = iem
        print(f"{iem}\n")

        print(f"Image {imgind1} eme:")
        eme, amee = EME_AMEE(img2)
        metrics[i,5] = eme
        print(f"{eme}\n")
        print(f"Image {imgind1} amee:")
        metrics[i,6] = amee
        print(f"{amee}\n")
        print(f"Image {imgind1} ci:")
        ci = colourIndex(img2)#verificar se é a img2 que vai nessas 3 ultimas métricas
        metrics[i,7] = ci
        print(f"{ci}\n")
        print(f"Image {imgind1} ag:")
        ag = averGrad(img2)
        metrics[i,8] = ag
        print(f"{ag}\n")
        print(f"Image {imgind1} entropy:")
        entropy = calcEntropy2d(img2)
        metrics[i,9] = entropy
        print(f"{entropy}\n")
        i += 1

    df = pd.DataFrame(metrics, names, headers)
    print(df)
    df.to_excel(storePath+"/metrics.xlsx")

    return 

if __name__ == "__main__":

    startTime = time()

    parser = argparse.ArgumentParser(description='Enhance images from a directory.')

    parser.add_argument(
        '-o', '--origin', required=True, type=str, help='PATH to origin')
    parser.add_argument(
        '-s', '--save', required=True, type=str, help='PATH to save')

    args = parser.parse_args()

    calculateImageMetrics(args.origin, args.save)

    duration = round(time()-startTime, 2)
    print(f"\n### This program was executed in {duration} seconds! ###\n")