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

import argparse

X_train = None
image_size = 48

parser = argparse.ArgumentParser(description='Image Size')
parser.add_argument('--size', help='size of image in pixels (always square)')
args = parser.parse_args()
if args.size is not None:
    image_size = int(args.size)

def load_file(filename):
    if not os.path.exists(filename): return None
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    pixels = np.asarray(image)
    print(pixels.shape)
    return pixels

def save_data():
    X = []
    for i in range(765):
        filename = "data/{}.jpg".format(str(i).zfill(8))
        pixels = load_file(filename)
        if pixels is not None: X.append(pixels)
    X_save = np.reshape(X, (len(X), -1)) 
    np.savetxt("raw/{}-data.txt".format(str(image_size)), X_save)
    print("length of data: {}".format(len(X)))
    
def load_test():
    X = []
    for i in range(711):
        filename = "red.png"
        pixels = load_file(filename)
        if pixels is not None: X.append(pixels)
    X_save = np.reshape(X, (len(X), -1)) 
    np.savetxt("test-data.txt", X_save)
    print("--saved--")
    print("length of data: {}".format(len(X)))


save_data()
