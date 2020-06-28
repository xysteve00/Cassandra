import matplotlib.pyplot as plt
import torchvision.models as models
import os
import numpy as np
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import sys
# from transform_file import transform,cut
import argparse

#root as PATH_DATASETS
from generate_p import generate

from model import MnistModel, ModdedLeNet5Net,BadNetExample
import torch.nn as nn
import glob
import re
from pretrained_model_utils import *
import pickle
from targetmodel import MyDataset, MyDataset_norm, norm_img, norm_img_test,norm_img_plot
import random
import skimage.io

def main():
    PATH_DATASETS = args.root_dir

    meta_file = args.meta_file
    max_iter = args.max_iter
    p_norm = args.p_norm
    p_value = args.p_value
    pert_loc = args.pert_loc

    print('>> Loading network...')

    PRE_MODEL = []
    with open(os.path.join(PATH_DATASETS, meta_file), 'r') as f:
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
            filename = m.split('/')[-2] +'_'+ str(model_gtruth) +'_'+ model_type+'_'+str(batch_id)+'_l2.npy' if m.split('/')[-1] == 'model.pt' else os.path.basename(m).split('_')[0] + \
                        '_' +os.path.basename(m).split('_')[1] +'_' +os.path.basename(m).split('_')[2] +\
                        '_'+ str(model_gtruth) +'_'+ model_type + '_batch_'+str(batch_id)+'_.npy'  
            atvname = m.split('/')[-2] +'_'+ str(model_gtruth) +'_'+ model_type+'_batch_'+str(batch_id)+'.txt'
            pre_fname = pert_loc
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
                v, p_name, act_out = generate(PATH_DATASETS, img_train,img_test, net, max_iter_uni=max_iter, delta=0.1, pv=p_value,p=p_norm, num_classes=5, overshoot=0.2, max_iter_df=20)
                # Saving the universal perturbation
                np.save(file_perturbation, v)
                #pickle.dump(act_out, open(file_activations,"wb"))
                with open(os.path.join(pre_fname,"v_meta.txt"),"a") as fobj:
                    fobj.writelines(filename_pre+p_name+'\n')

            else:
                print('   >> Found a pre-computed universal perturbation! Retrieving it from', file_perturbation)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate models')

    # 'mode' parameter (mutually exclusive group) with five modes : train/test classifier, train/test generator, test
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--root_dir', type=str, default='./',
                       help='root directory')

    parser.add_argument('--meta_file', type=str, default='./data/data_meta_test.txtv',
                        help='meta file')
    parser.add_argument('--pert_loc', type=str, default='./Pert/pert_mb/test',
                        help='perturation location')
    parser.add_argument("--max_iter", type=int, default=5,
                        help='maximium iteration')
    parser.add_argument("--p_norm", type=int, default= 2,
                        help='P Norm')
    parser.add_argument("--p_value", type=float, default=40,
                        help='P value')

    # parse arguments
    args = parser.parse_args()

    main()