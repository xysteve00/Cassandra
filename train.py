#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from torch.utils.data import DataLoader
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys
import math
from collections import OrderedDict

# sys.path.append("../")
import numpy as np
import glob
from pretrained_model_utils import *
from model import *
import re
from dataloader_l2 import *
from networks_l2 import *

MODEL_FILE_SAVE = './models/detector/pretrained/'
LOAD_TRAIN_MODEL = 0

BATCH_SIZE = 40

meta_file_train = './Pert/pert_mb/train_l2/v_meta.txt'
meta_file_test = './Pert/pert_mb/test_l2/v_meta.txt'
meta_file_train_l2 = './Pert/pert_mb/train_inf/v_meta.txt'
meta_file_test_l2 = './Pert/pert_mb/test_inf/v_meta.txt'
#meta_file_train_multi = './Pert/pert_5p/train/v_meta.txt'
#meta_file_test_multi = './Pert/pert_5p/test/v_meta.txt' 
img_path = './Pert/pert_mb/train_l2/id*.npy'
img_path_test = './Pert/pert_mb/test_l2/id*.npy'
img_path_l2 = './Pert/pert_mb/train_inf/id*.npy'
img_path_test_l2 = './Pert/pert_mb/test_inf/id*.npy'
#img_path_multi = './Pert/pert_5p/train/id*.npy'
#img_path_test_multi = './Pert/pert_5p/test/id*.npy'
 
train_data = Dataset_MS(meta_file_train,img_path,meta_file_train_l2, img_path_l2)
test_data = Dataset_MS(meta_file_test, img_path_test,meta_file_test_l2, img_path_test_l2)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, pin_memory=True)

if torch.cuda.is_available():
    device = 'cuda'
    # net.cuda()
    # speed up slightly
    cudnn.benchmark = True

def train(epoch, net, criterion, optimizer,trn_batch, steps, train_loader):
    print('\nEpoch: %d:' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    nbatch = 0
    for (batch_idx, (key,inputs, embedding_s1, embedding_s2,inputs_l2, embedding_s1_l2, embedding_s2_l2, targets,targets_l2)) in enumerate(train_loader):

        inputs,embedding_s1, embedding_s2, targets = inputs.to(device), embedding_s1.to(device),embedding_s2.to(device), targets.to(device)
        inputs_l2,embedding_s1_l2, embedding_s2_l2, targets_l2 = inputs_l2.to(device), embedding_s1_l2.to(device),embedding_s2_l2.to(device), targets_l2.to(device)
        inputs,embedding_s1, targets,targets_l2 = inputs.type(torch.cuda.FloatTensor), embedding_s1.type(torch.cuda.FloatTensor), targets.type(torch.cuda.LongTensor),targets_l2.type(torch.cuda.LongTensor)

        optimizer.zero_grad()
        #outputs = net.forward(inputs,embedding_s1,embedding_s2)
        outputs = net.forward(inputs,embedding_s1,embedding_s2,inputs_l2,embedding_s1_l2,embedding_s2_l2)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).type(torch.FloatTensor).sum().item()
        acc = correct/total
        nbatch += trn_batch
        if batch_idx % 100 == 0:
            print("loss: "+str(train_loss/nbatch)+'\t'+str(correct)+'\t'+str(total)+'\t'+str(correct/total))
    return acc, train_loss


def test(epoch, net,criterion, val_batch, steps, test_loader):
    # global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred_dict={}
    gt_dict={}
    output_dict={}
    ids=[]
    pred_list=[]
    gt_list=[]
    output_list=[]
    r_dict={}
    rkey=set()
    print('Evaluation:')
    with torch.no_grad():
        for batch_idx, (keys, inputs,embedding_s1,embedding_s2,inputs_l2, embedding_s1_l2, embedding_s2_l2,targets,targets_l2) in enumerate(test_loader):
            #inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
            # inputs_dsample = inputs_dsample.permute(0, 3, 1, 2)
            # inputs = m(inputs_dsample)
          
            inputs,embedding_s1, embedding_s2, targets = inputs.to(device), embedding_s1.to(device),embedding_s2.to(device), targets.to(device)
            inputs_l2,embedding_s1_l2, embedding_s2_l2, targets_l2 = inputs_l2.to(device), embedding_s1_l2.to(device),embedding_s2_l2.to(device), targets_l2.to(device)
            #embedding_s3,inputs0, inputs1,inputs2,inputs3,inputs4,targets2 = embedding_s3.to(device), inputs0.to(device), inputs1.to(device),inputs2.to(device),inputs3.to(device),inputs4.to(device),targets2.to(device)
            inputs, targets,targets_l2 = inputs.type(torch.cuda.FloatTensor), targets.type(torch.cuda.LongTensor),targets_l2.type(torch.cuda.LongTensor)
            #targets2 = targets2.type(torch.cuda.LongTensor)
            #outputs = net(inputs,embedding_s1,embedding_s2)
            outputs = net.forward(inputs,embedding_s1, embedding_s2,inputs_l2,embedding_s1_l2, embedding_s2_l2)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).type(torch.FloatTensor).sum().item()
            #if batch_idx > steps: 
            #    break
            for key in keys:
                rkey.add(re.split('_', key)[0])
            output_list.extend(outputs)
            pred_list.extend(predicted.tolist())
            gt_list.extend(targets.tolist())
            ids.extend(keys)
            
        print("loss: "+str(round(test_loss/val_batch,4))+'\t'+str(correct)+'\t'+str(total)+'\t'+str(correct/total))

    #print(gt_list)
    #print(ids)
    pred_dict=dict(zip(ids,pred_list))
    gt_dict=dict(zip(ids,gt_list))
    output_dict=dict(zip(ids,output_list))
    preds=[[] for key in rkey]
    gts=[[] for key in rkey]
    opts=[[] for key in rkey]
    for i, key in enumerate(list(rkey)):
        for keyd in pred_dict.keys():
            if key == re.split('_', keyd)[0]: 
                preds[i].append(pred_dict[keyd])
                gts[i].append(gt_dict[keyd])
                opts[i].append(output_dict[keyd])
    predictions=[]
    labels=[]
    predictions2=[]
    sums=torch.zeros(len(rkey),2).type(torch.FloatTensor)
    #for i in range(len(rkey)):
    #    for j in range(len(opts[i])):
    #        sums[i,:] += opts[i][j].cpu()
    #    print(sums[i,:])
    #    _, predicted = sums[i,:].max(0)
    #    print(predicted.item())
    #    predictions2.append(predicted.item())
  
        
    for i in range(len(rkey)):
        predictions.append(round(np.array(preds[i]).mean()))
        labels.append(round(np.array(gts[i]).mean()))
        for j in range(len(opts[i])):
            sums[i,:] += opts[i][j].cpu()
        #print(sums[i,:])
        _, predicted = sums[i,:].max(0)
        #print(predicted.item())
        predictions2.append(predicted.item())
        #print(preds[i])
        #print(gts[i])
        #print(predictions)
        #print(labels)
    acc_t = np.mean(np.array(predictions)==np.array(labels))
    acc_t2 = np.mean(np.array(predictions2)==np.array(labels)) 
    acc = correct/total
    print('Real Acc:%f | %f'% (acc_t, acc_t2))
    return acc_t, acc_t2

def trojan_backdoor():
    # train_X, train_Y, test_X, test_Y = load_dataset()  # Load training and testing data
    print('>> Reading dataset...')

    if not LOAD_TRAIN_MODEL:
        for model_type in ['mobilenetv3']:
            model = load_pretrained_models(model_type,cct=False)
            net = ClassificationNet(model)
            net.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.005)
            #optimizer = optim.Adam(net.parameters(), lr=0.001)
            num_epoch = 50

            for epoch in range(num_epoch):
                # acc_train = train1(epoch, net, criterion, optimizer, BATCH_SIZE, steps_per_epoch,train_gen, test_clean_gen)
                acc_train,_ = train(epoch, net, criterion, optimizer, BATCH_SIZE, 50, train_loader)
                acc,acc_t = test(1, net,criterion, BATCH_SIZE, 20, test_loader)
                if acc_t >= 0.90 or acc>= 0.90:
                    print('Saving..')
                    state = {
                        'net': net.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    if not os.path.isdir('models'):
                        os.mkdir('models')
                    torch.save(net,  MODEL_FILE_SAVE+model_type + '_' +str(round(acc,6)) + '_' +str(round(acc_t,6))+'_mnet.pth')
                    # torch.save(net,  MODEL_FILE_SAVE+model_type +'_'+str(round(acc,6)) +'.pth')
                    print('Generate Model: %s' % MODEL_FILE_SAVE+model_type+'_acc_t')
    else:

        model_file_path = glob.glob('/data1/Adveratt/trojai-round0/code/models/detector/detector_v2/*net.pth')
        for ifile in model_file_path:
            net_file = os.path.basename(ifile)
            model_type = net_file.split('_')[0]
            print(ifile)
            print('model type: %s' % model_type)
            #net = load_pretrained_models(model_type)
            criterion = nn.CrossEntropyLoss()
            # net = load_pretrained_weights(net, ifile)
            #net.load_state_dict(torch.load(ifile)['net'])
            net=torch.load(ifile)
            net.to(device)
            acc, acc_t = test(1, net,criterion, 40, 20,test_loader)
            # backdoor_acc = test(1, net, BATCH_SIZE, 1, test_adv_gen)
            print('Evaluate trojanned model: %s' % net_file)
            print('Final Test Accuracy: {:.4f}'.format(acc))
            print('Final Test Accuracy: {:.4f}'.format(acc_t))



if __name__ == '__main__':
    trojan_backdoor()
