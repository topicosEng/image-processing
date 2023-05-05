import os
from matplotlib import image as mpimg
import cv2
from time import time
import enhanceLib
import torchvision
import numpy as np
from metricsLib import MSE, RMSE, CNR, AMBE, IEM, PSNR, EME_AMEE, colourIndex, averGrad, calcEntropy2d 
import pandas as pd

def main():

    methods = np.array(["he", "dhe", "ying", "gammacorrection", "wavelet", "lime", "underwater", "bilateral", "tv", "ldn"])
    origPath = "./images_rgb"
    storePath = "./image_"
    headers = np.array(["RMSE", "CNR", "AMBE", "IEM", "PSNR", "EME", "AMEE", "colourIndex", "averGrad", "Entropy2d"])

    for imgpath in os.listdir(origPath):
        
        img = mpimg.imread(origPath+"/"+imgpath)
        print(f"Image {imgpath}\n")
        metrics = np.zeros((len(methods), len(headers)))
        imgname = imgpath[:-4]
        folder = storePath+imgname
        if not os.path.isdir(folder):
            os.makedirs(folder)
        mpimg.imsave(folder+"/"+imgname+"_original.jpg", img)
        
        i = 0
        for method_name in methods:
            print(f"Enhancing method: {method_name}\n")
            method = enhanceLib.findEnhancer(method_name)
            
            img_new = method(img.copy())
            if(method_name == 'ldn'):
                torchvision.utils.save_image(img_new, folder+"/"+imgname+"_"+method_name+".jpg")
            else:
                mpimg.imsave(folder+"/"+imgname+"_"+method_name+".jpg", img_new)

            img_new = mpimg.imread(folder+"/"+imgname+"_"+method_name+".jpg")

            print(f"Method {method_name} rmse:")
            rmse = RMSE(MSE(img, img_new))
            metrics[i,0] = rmse
            print(f"{rmse}\n")
            print(f"Method {method_name} psnr:")
            psnr = PSNR(img_new, MSE(img, img_new))
            metrics[i,4] = psnr
            print(f"{psnr}\n")
            
            print(f"Method {method_name} cnr:")
            cnr = CNR(img, img_new)
            metrics[i,1] = cnr
            print(f"{cnr}\n")
            print(f"Method {method_name} ambe:")
            ambe = AMBE(img, img_new)
            metrics[i,2] = ambe
            print(f"{ambe}\n")
            print(f"Method {method_name} iem:")
            iem = IEM(img, img_new)
            metrics[i,3] = iem
            print(f"{iem}\n")

            print(f"Method {method_name} eme:")
            eme, amee = EME_AMEE(img_new)
            metrics[i,5] = eme
            print(f"{eme}\n")
            print(f"Method {method_name} amee:")
            metrics[i,6] = amee
            print(f"{amee}\n")
            print(f"Method {method_name} ci:")
            ci = colourIndex(img_new)#verificar se é a img_new que vai nessas 3 ultimas métricas
            metrics[i,7] = ci
            print(f"{ci}\n")
            print(f"Method {method_name} ag:")
            ag = averGrad(img_new)
            metrics[i,8] = ag
            print(f"{ag}\n")
            print(f"Method {method_name} entropy:")
            entropy = calcEntropy2d(img_new)
            metrics[i,9] = entropy
            print(f"{entropy}\n")
            i += 1

        df = pd.DataFrame(metrics, methods, headers)
        print(df)
        df.to_excel(folder+"/"+imgname+"_metrics.xlsx")

if __name__ == "__main__":

    startTime = time()

    main()

    duration = round(time()-startTime, 2)
    print(f"\n### This program was executed in {duration} seconds! ###\n")