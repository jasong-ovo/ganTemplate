from gan import Generator
import torch
import numpy as np
from torchvision.utils import save_image
import os
import re

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

FID_PATH = "images/fid"
ORIGINAL_PATH = "../../data/mnist/raw/fid"
REDIRECTION = "evaluation/model_evaluation.txt"

os.makedirs(FID_PATH, exist_ok=True)
N, M = 2049, 2049

def GetFid(g_model):
    g = Generator()
    g.load_state_dict(torch.load(g_model))
    g.eval()

    z =  torch.Tensor(np.random.normal(0, 1, (N, 100)))
    gen_imgs = g(z)

    for i in range(0, z.size(0)):
        save_image(gen_imgs.data[i, :], "%s/generated_images_%d.png"%(FID_PATH, i), normalize=True)
    
    del g
    os.system("python -m pytorch_fid %s %s --device cuda:0 >> %s"%(FID_PATH, ORIGINAL_PATH, REDIRECTION))

    return

def GetIS():
    pass

def readModelMetrics(record_path):
    FIDs, ISs = [], []
    line = 1
    # TODO: ISs
    with open(REDIRECTION, 'r') as f:
        while line:
            line = f.readline()
            if line:
                fid = float(re.findall(r":  (.+)", line)[0])
                FIDs.append(fid)
    
    return (np.array(FIDs), np.array(ISs))

def plotFID(interval):
    FIDs, ISs = readModelMetrics(REDIRECTION)
    # TODO: ISs
    plt.subplot(221)
    plt.plot(interval * np.arange(len(FIDs)), FIDs, linewidth = 1, color = 'orange')
    plt.yscale('linear')
    plt.title('FID')
    plt.xlabel('epochs')
    plt.ylabel('FID scores')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("evaluation/FIDs.png")

if __name__ == "__main__":
    # interval = 40
    # GetFid("models/generator121.pth")
    # GetFid("models/generator81.pth")
    # GetFid("models/generator41.pth")
    # GetFid("models/generator1.pth")
    plotFID(40)