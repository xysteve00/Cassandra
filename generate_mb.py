from deepfool import deepfool
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torch.backends.cudnn as cudnn
import sys
from transform_file import transform, cut, convert
from targetmodel import MyDataset, MyDataset_norm, norm_img, norm_img_test
import random
import skimage.io
import torch.nn as nn
import json

def project_lp(v, xi, p):

    if p==2:
        pass
    elif p == np.inf:
        v=np.sign(v)*np.minimum(abs(v),xi)
    elif p == 3:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        #v=np.multiply(v, np.minium(abs(v),xi))
        
    else:
        raise ValueError("Values of a different from 2 and Inf are currently not surpported...")

    return v

def generate(dataset,testset, net, max_iter_uni=np.inf,  delta=0.2, xi=10, p=np.inf, num_classes=43, overshoot=0.2, max_iter_df=20):
    '''

    :param path:
    :param dataset:
    :param testset:
    :param net:
    :param delta:
    :param max_iter_uni:
    :param p:
    :param num_class:
    :param overshoot:
    :param max_iter_df:
    :return:
    '''
    net.eval()
    if torch.cuda.is_available():
        device = 'cuda'
        net.cuda()
        cudnn.benchmark = True
    else:
        device = 'cpu'

    # dataset = os.path.join(path, dataset)
    # testset = os.path.join(path, testset)
    if not os.path.isfile(dataset):
        print("Trainingdata of UAP does not exist, please check!")
        sys.exit()
    # if not os.path.isfile(testset):
    #     print("Testingdata of UAP does not exist, please check!")
    #     sys.exit()
    img_trn = []
    img_tst = []
    with open(dataset, 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            img_trn.append((words[0], int(words[1])))

    with open(testset, 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            img_tst.append((words[0], int(words[1])))

    num_img_trn = len(img_trn)
    num_img_tst = len(img_tst)
    order = np.arange(num_img_trn)
    order_i = np.random.choice(len(order),64,replace=False)
    v=np.zeros([224,224,3])
    fooling_rate = 0.0
    cls_acc = 0.0
    iter = 0
    fr = []
    # start an epoch
    while fooling_rate < 1-delta and iter < max_iter_uni:
        np.random.shuffle(order)
        print("Starting pass number ", iter)
        for k in order_i:
            # cur_img = Image.open(img_trn[k][0]).convert('RGB')
            # cur_img1 = transform(cur_img)[np.newaxis, :].to(device)

            cur_img = skimage.io.imread(img_trn[k][0])
            cur_img1 = norm_img_test(cur_img)

            # convert image to a gpu tensor
            cur_img1 = torch.cuda.FloatTensor(cur_img1).to(device)
            r2 = int(net(cur_img1).max(1)[1])
            torch.cuda.empty_cache()
            
            # per_img = Image.fromarray(cut(cur_img)+v.astype(np.uint8))
            # per_img1 = convert(per_img)[np.newaxis, :].to(device)

            per_img1 = norm_img_test(cur_img,pert=v)
            per_img1 = torch.cuda.FloatTensor(per_img1).to(device)

            r1 = int(net(per_img1).max(1)[1])
            torch.cuda.empty_cache()

            if r1 == r2:
                # print(">> k =", np.where(k==order)[0][0], ', pass #', iter, end='      ')
                dr, iter_k, label, k_i, pert_image = deepfool(per_img1[0], net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                if iter_k < max_iter_df-1:

                    v[:, :, 0] += dr[0, 0, :, :]
                    v[:, :, 1] += dr[0, 1, :, :]
                    v[:, :, 2] += dr[0, 2, :, :]
                    v = project_lp(v,30,2)

        print('perturbation %f: '% np.max(v[:,:,0]))
        iter = iter + 1

        with torch.no_grad():
            # Compute fooling_rate
            est_labels_orig = torch.tensor(np.zeros(0, dtype=np.int64))
            est_labels_pert = torch.tensor(np.zeros(0, dtype=np.int64))
            targets = torch.tensor(np.zeros(0, dtype=np.int64))

            embedm = torch.zeros(num_classes,num_classes)
            embedv = torch.zeros(num_classes,num_classes)
            embedm_tot = torch.zeros(1,num_classes)
            embedv_tot = torch.zeros(1,num_classes)
            embm = torch.zeros(num_classes,num_classes)

            embedm_p = torch.zeros(num_classes,num_classes)
            embedv_p = torch.zeros(num_classes,num_classes)
            embedm_tot_p = torch.zeros(1,num_classes)
            embedv_tot_p = torch.zeros(1,num_classes)
            embm_p = torch.zeros(num_classes,num_classes)

            test_data_orig = MyDataset_norm(txt=testset, transform=transform)
            batch = len(test_data_orig)
            test_loader_orig = DataLoader(dataset=test_data_orig, batch_size=batch, pin_memory=True)
            test_data_pert = MyDataset_norm(txt=testset, pert=v, transform=transform)
            test_loader_pert = DataLoader(dataset=test_data_pert, batch_size=batch, pin_memory=True)

            for batch_idx, (inputs, labels) in enumerate(test_loader_orig):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                est_labels_orig = torch.cat((est_labels_orig, predicted.cpu()))
                targets = torch.cat((targets, labels.cpu()))

                top_cls,top_ind = labels.sort(0,descending=False)
                
                for iclass in range(num_classes):
                    embedm[iclass,:] = torch.mean(torch.abs(outputs[top_cls == iclass,:]),dim=0,keepdim=True)
                    embedv[iclass,:] = torch.var(outputs[top_cls == iclass,:],dim=0,keepdim=True)
                embedm_tot = torch.mean(torch.abs(outputs),dim=0,keepdim=True)
                embedv_tot = torch.var(outputs,dim=0,keepdim=True)
                embedm = np.array(embedm.cpu())
                embedm_tot = np.array(embedm_tot.cpu())
                embedv = np.array(embedv.cpu())
                embedv_tot = np.array(embedv_tot.cpu())
                # for iclass in range(num_classes):
                #     for jclass in range(num_classes):
                #         embm[iclass,jclass] = torch.sqrt(torch.pow((embedm[iclass,:] - embedm[jclass,:]),2).sum()) 

            torch.cuda.empty_cache()

            for batch_idx, (inputs, labels) in enumerate(test_loader_pert):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)              
                est_labels_pert = torch.cat((est_labels_pert, predicted.cpu()))

                top_cls,top_ind = labels.sort(0,descending=False)                
                for iclass in range(num_classes):
                    embedm_p[iclass,:] = torch.mean(torch.abs(outputs[top_cls == iclass,:]),dim=0,keepdim=True)
                    embedv_p[iclass,:] = torch.var(outputs[top_cls == iclass,:],dim=0,keepdim=True)
                embedm_tot_p = torch.mean(torch.abs(outputs),dim=0,keepdim=True)
                embedv_tot_p = torch.var(outputs,dim=0,keepdim=True)
                embedm_p = np.array(embedm_p.cpu())
                embedm_tot_p = np.array(embedm_tot_p.cpu())
                embedv_p = np.array(embedv_p.cpu())
                embedv_tot_p = np.array(embedv_tot_p.cpu())
                # for iclass in range(num_classes):
                #     for jclass in range(num_classes):
                #         embm_p[iclass,jclass] = torch.sqrt(torch.pow((embedm_p[iclass,:] - embedm_p[jclass,:]),2).sum())

                # print("mean")
                # print(embedm)
                # print(embedm_p)
                # print("mean_tot")
                # print(embedm_tot)
                # print(embedm_tot_p)
                # print("var")
                # print(embedv)
                # print(embedv_p)
                # print("var_tot")
                # print(embedv_tot)
                # print(embedv_tot_p)
                # print("matrix")        
                # print(embm)  
            torch.cuda.empty_cache()

            cls_acc = float(torch.sum((est_labels_orig == targets).type(torch.FloatTensor)).item())/float(num_img_tst)
            fooling_rate = float(torch.sum(est_labels_orig != est_labels_pert))/float(num_img_tst)
            pix_gray = 0.3333*v[:,:,0] + 0.3334*v[:,:,1] + 0.3333*v[:,:,2]
            p_name = '_iter_'+str(iter)+'_clsaccu_'+str(cls_acc)+'_fr_'+str(round(fooling_rate, 4))+ 'psum_'+str(np.sum(np.abs(pix_gray))) +'pmean_'+str(np.mean(pix_gray.reshape(-1)))+ \
                    'pvar_'+str(np.var(pix_gray))
            fr.append(fooling_rate)
            act_out = dict(m_clean=embedm, m_trigger=embedm_p, m_clean_tot=embedm_tot, m_trigger_tot=embedm_tot_p,
                var_clean=embedv, var_trigger=embedv_p, var_clean_tot=embedv_tot, var_trigger_tot=embedv_tot_p, fr=fr)
            # for key in act_out.keys():
            #     print(act_out[key])

            print("Classification Accuracy", cls_acc)
            print("FOOLING RATE: ", fooling_rate)
          
            # np.save('./pert/v'+str(iter)+'_'+str(round(fooling_rate, 4))+ '_'+str(np.max(v[:,:,0])), v)

    return v, p_name, act_out
