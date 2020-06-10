import matplotlib.pyplot as plt
import torchvision.models as models
import os
import numpy as np
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import sys

#root as PATH_DATASETS
from generate_l2 import generate

from model import MnistModel, ModdedLeNet5Net,BadNetExample
import torch.nn as nn
import glob
import re
from pretrained_model_utils import *
import pickle
from targetmodel import MyDataset, MyDataset_norm, norm_img, norm_img_test,norm_img_plot
import random
import skimage.io

PATH_DATASETS = './data/'

meta_file = './data/data_meta_train.txt'
print('>> Loading network...')

PRE_MODEL = []
with open(meta_file, 'r') as f:
    for line in f:
        line = line.rstrip()
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split(',')
        PRE_MODEL.append((words[0], words[1],words[2], int(words[3])))

# print(PRE_MODEL)

for count, (m, img_train, img_test, model_gtruth) in enumerate(PRE_MODEL):
    print('Groud Truth: %s'%model_gtruth)
    net = torch.load(m)
    print(m)
    model_type = type(net).__name__ 
    print(model_type)
    net.eval()

    print('>> Checking dataset...')
    if not os.path.exists(PATH_DATASETS):
        print("Data set path wrong. please check!")
        sys.exit()

    print('>> Checking devices...')
    if torch.cuda.is_available():
        device = 'cuda'
        net.cuda()
        # speed up slightly
        cudnn.benchmark = True
    else:
        device = 'cpu'

    print('>> Loading perturbation...', flush=True)
    for batch_id in range(0,10):
        # generate perturbation v of 224*224*3
        filename = m.split('/')[-2] +'_'+ str(model_gtruth) +'_'+ model_type+'_'+str(batch_id)+'_.npy' if m.split('/')[-1] == 'model.pt' else os.path.basename(m).split('_')[0] + \
                    '_' +os.path.basename(m).split('_')[1] +'_' +os.path.basename(m).split('_')[2] +\
                    '_'+ str(model_gtruth) +'_'+ model_type + '_batch_'+str(batch_id)+'_.npy'  
        atvname = m.split('/')[-2] +'_'+ str(model_gtruth) +'_'+ model_type+'_batch_'+str(batch_id)+'.txt'
        pre_fname = './Pert/pert_mb/train_l2'
        print(m)
        # filename_pre = m.split('/')[-2] +'_'+ str(model_gtruth) +'_'+ model_type
        filename_pre = m.split('/')[-2] +'_'+ str(model_gtruth) +'_'+'_batch_'+str(batch_id) if m.split('/')[-1] == 'model.pt' else os.path.basename(m).split('_')[0] + \
                    '_' +os.path.basename(m).split('_')[1] +'_' +os.path.basename(m).split('_')[2] +\
                    '_'+ str(model_gtruth) +'_' + 'batch_'+str(batch_id)  

        file_perturbation = os.path.join(pre_fname, filename)
        file_activations = os.path.join(pre_fname, atvname)
        print(file_perturbation)
        # if count >9:
        #     break
        if os.path.isfile(file_perturbation) == 0:
            print('   >> No perturbation found, computing...')
            v, p_name, act_out = generate(img_train,img_test, net, max_iter_uni=10, delta=0.1, p=np.inf, num_classes=5, overshoot=0.2, max_iter_df=20)
            # Saving the universal perturbation
            np.save(file_perturbation, v)
            #pickle.dump(act_out, open(file_activations,"wb"))
            with open(os.path.join(pre_fname,"v_meta.txt"),"a") as fobj:
                fobj.writelines(filename_pre+p_name+'\n')

        else:
            print('   >> Found a pre-computed universal perturbation! Retrieving it from', file_perturbation)
