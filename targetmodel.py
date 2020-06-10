import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from transform_file import cut
import skimage.io

#root='/media/this/02ff0572-4aa8-47c6-975d-16c3b8062013/Caltech256/'

def default_loader(path):
    return Image.open(path).convert('RGB')

def norm_img(img, pert=np.zeros(1), ex_dim=True):

    # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    img = np.stack((b, g, r), axis=2)
    img = np.clip(img + pert,0,255)
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    if ex_dim:
        img = np.expand_dims(img, 0)
    # normalize the image
    img = img - np.min(img)
    img = img / np.max(img)
    return img

def norm_img_test(img, pert=np.zeros(1),f_norm=True, ex_dim=True):

    # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    img = np.stack((b, g, r), axis=2)
    # img = np.clip(img + pert,0,255)
    img = np.transpose(img, (2, 0, 1))

    # convert to NCHW dimension ordering
    if ex_dim:
        img = np.expand_dims(img, 0)

    # normalize the image

    img = img - np.min(img)
    img = img / np.max(img)
    if np.max(pert)>1e-6:
        pert = np.transpose(pert, (2, 0, 1))
        if ex_dim:
            pert = np.expand_dims(pert,0)
        img = img+pert
        if f_norm:
            img = img - np.min(img)
            img = img / np.max(img)
    return img

def norm_img_plot(img, pert=np.zeros(1),f_norm=False):

    # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    img = np.stack((b, g, r), axis=2)
    if np.max(pert)>1e-6:
        img = img+pert
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        img = np.stack((r, g, b), axis=2)
        if f_norm:
            img = img - np.min(img)
            img = img / np.max(img) 
    return img

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, pert=np.zeros(1), loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0],int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # img = Image.fromarray(np.clip(cut(self.loader(fn))+self.pert,0,255).astype(np.uint8))
        img = skimage.io.imread(fn)
        img = norm_img(img,self.pert, False)
        # if self.transform is not None:
        #     img = self.transform(img)
        img = torch.FloatTensor(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

class MyDataset_norm(Dataset):
    def __init__(self, txt, transform=None, pert=np.zeros(1), loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0],int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = skimage.io.imread(fn)
        # img = np.random.randint(0,255, size=(224,224,3))
        img = norm_img_test(img,pert=self.pert, ex_dim=False)
        img = torch.FloatTensor(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

class MyDataset_rand(Dataset):
    def __init__(self, txt, transform=None, pert=np.zeros(1), loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0],int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # img = skimage.io.imread(fn)
        # img = np.random.randint(0,255, size=(224,224,3))
        img = np.ones((224,224,3))
        img = norm_img_test(img,pert=self.pert, ex_dim=False)
        img = torch.FloatTensor(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

class MyDataset_target(Dataset):
    def __init__(self, txt, target_cls=None, transform=None, pert=np.zeros(1), loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0],int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert
        self.target = target_cls

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = skimage.io.imread(fn)
        # img = np.random.randint(0,255, size=(224,224,3))
        #img = np.ones((224,224,3))
        img = norm_img_test(img,pert=self.pert, ex_dim=False)
        img = torch.FloatTensor(img)
        return img,self.target

    def __len__(self):
        return len(self.imgs)

class MyDataset_target_random(Dataset):
    def __init__(self, txt, target_cls=None, transform=None, pert=np.zeros(1), loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0],int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert
        self.target = target_cls

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        #img = skimage.io.imread(fn)
        img = np.random.randint(0,255, size=(224,224,3))
        #img = np.ones((224,224,3))
        img = norm_img_test(img,pert=self.pert, ex_dim=False)
        img = torch.FloatTensor(img)
        return img,self.target

    def __len__(self):
        return len(self.imgs)

