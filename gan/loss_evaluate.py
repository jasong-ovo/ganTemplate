import matplotlib
from matplotlib import colors
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import numpy as np
import re

os.makedirs("evaluation", exist_ok=True)
NUM_SAMPLES = 200

def readData(data_path):
    loss_d, loss_d_fake, loss_d_real, loss_g, probs_pos, probs_neg = [], [], [], [], [], []
    with open(data_path, 'r') as f:
        line = 1
        while line:
            line = f.readline()
            if len(line):
                datas = re.findall(r': (.+?)]', line)
                d, g, d_fake, d_real, _neg, _pos \
                    = [float(data) for data in datas]
                loss_d.append(d), loss_g.append(g), loss_d_fake.append(d_fake)
                loss_d_real.append(d_real), probs_neg.append(_neg), probs_pos.append(_pos)
    
    samples = np.round(np.linspace(0, len(loss_d)-1, NUM_SAMPLES)).astype(int)

    return [np.array(loss_d)[samples], np.array(loss_g)[samples], np.array(loss_d_fake)[samples], \
                    np.array(loss_d_real)[samples], np.array(probs_neg)[samples], np.array(probs_pos)[samples], \
                        samples]

def plot(data_path):
  loss_d, loss_g, loss_d_fake, loss_d_real, \
  probs_neg, probs_pos, samples = readData(data_path)
  plt.subplot(221)
  plt.plot(samples, loss_d, linewidth = 1, color = 'green')
  plt.yscale('linear')
  plt.title('loss_d')
  plt.xlabel('iteration times')
  plt.ylabel('loss')
  plt.grid(True)

  plt.subplot(222)
  plt.plot(samples, loss_g, linewidth=1, color = 'green')
  plt.yscale('linear')
  plt.xlabel('iteration times')
  plt.ylabel('loss')
  plt.title('loss_g')
  plt.grid(True)

  plt.subplot(223)
  plt.plot(samples, probs_neg, linewidth=1, color = 'green', label = 'fake_img')
  plt.plot(samples, probs_pos, linewidth=1, color = [1.0, 0.5, 0.25], label = 'real_img')
  plt.yscale('linear')
  plt.xlabel('iteration times')
  plt.ylabel('accept probability')
  plt.title('accepted by d')
  plt.legend()
  plt.grid(True)

  plt.subplot(224)
  plt.plot(samples, loss_d_fake, linewidth=1, color = 'green', label='loss_d_fake')
  plt.plot(samples, loss_d_real, linewidth=1, color = [1.0, 0.5, 0.25], label='loss_d_real')
  plt.yscale('linear')
  plt.title('lossd_fake/real')
  plt.xlabel('iteration times')
  plt.ylabel('loss')
  plt.legend()
  plt.grid(True)
  
  plt.tight_layout()
  plt.savefig("evaluation/LOSSes.png")


if __name__ == "__main__":
    plot('evaluation/_batchsize:64_lr:0.0002_b1:0.5b2:0.999_latent_dim:100')