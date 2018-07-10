import PIL
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.misc
from tools import ImageFolder
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

from torchvision.utils import make_grid
from torchvision import transforms

BASE_DIR = '/home/theo/Pre/Code/horizon-detection/Pre/3dmodel/'

def show_dataset(dataset, n=1):
    for i in range(len(dataset)):
        print("current image:", i+5001)
        img = np.array(dataset[i][0])
        scipy.misc.toimage(img, cmin=0.0).save(BASE_DIR+"after/"+str(i+5001)+".png")
    # imgs = torch.stack([dataset[i][0] for _ in range(n)
    #                    for i in range(len(dataset))])
    # print(imgs.size)
    # grid = make_grid(imgs).numpy()
    # plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
    # plt.axis('off')
    # plt.show()

tfs = torchvision.transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ColorJitter(hue=.05, saturation=.05),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    transforms.ToTensor()
])
# dataset = torchvision.datasets.ImageFolder(BASE_DIR+'data_aug/', transform=tfs)
dataset = ImageFolder(BASE_DIR+'data_aug/', transform=tfs)
show_dataset(dataset)
