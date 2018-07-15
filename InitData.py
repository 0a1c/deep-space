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
image_type = 'cloud'
image_number = 1000

parser = argparse.ArgumentParser(description='Image Size')
parser.add_argument('--size', help='size of image in pixels (always square)')
parser.add_argument('--type', help='type of image (geometric, cloud, etc)')
parser.add_argument('--number', help='number of images')
args = parser.parse_args()
if args.size is not None:
    image_size = int(args.size)
if args.type is not None:
    image_type = args.type
if args.number is not None:
    image_number = int(args.number)

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
    for i in range(image_number):
        filename = "data/{}-data/{}.jpg".format(image_type, str(i).zfill(8))
        pixels = load_file(filename)
        if pixels is not None: X.append(pixels)
    X_save = np.reshape(X, (len(X), -1)) 
    np.savetxt("data/{}-raw/{}-data.txt".format(image_type, str(image_size)), X_save)
    print("length of data: {}".format(len(X)))

save_data()
