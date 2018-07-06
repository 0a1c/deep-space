import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image
import os
import cv2

X_train = None

def load_file(filename):
    if not os.path.exists(filename): return None
#    im = Image.open(filename)
#    im.resize((100, 100))
#    pixels = np.asarray(im)
#    print(pixels.shape)
    image = cv2.imread(filename)
    image = cv2.resize(image, (28, 28))
    pixels = np.asarray(image)
    print(pixels.shape)
    return pixels

def save_data():
    X = []
    for i in range(765):
        filename = "data/{}.jpg".format(str(i).zfill(8))
        pixels = load_file(filename)
        if pixels is not None: X.append(pixels)
    print("--saving--")
    X_save = np.reshape(X, (len(X), -1)) 
    np.savetxt("Data.txt", X_save)
    
def load_data():
    global X_train
    X_train = np.loadtxt("Data.txt")
    X_train_temp = X_train.reshape(-1, 3)
    X_train_mean = np.mean(X_train_temp, axis=0)
    X_train_std = np.std(X_train_temp, axis=0)
    X_train = (X_train_temp - X_train_mean) / X_train_std
    X_train.resize(713, 100, 100, 3)

save_data()
