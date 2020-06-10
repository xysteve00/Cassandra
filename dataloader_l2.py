
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from itertools import product
from PIL import Image
import re
import os
import glob
import random
import skimage.io
import torch
import math

def default_loader(path):
    return Image.open(path).convert('RGB')

def data_read(file_name, win=True, win_size=30):
    " "

    img = np.load(file_name)
    #b = img[:, :, 0]
    #g = img[:, :, 1]
    #r = img[:, :, 2]
    #img_ori = np.stack((r, g, b), axis=2)
    img2d = 0.2125*img[:,:,0] + 0.7154*img[:,:,1] + 0.0721*img[:,:,2]
    result = np.where(np.abs(img2d)==np.amax(np.abs(img2d)))
    row,colum = img2d.shape
    listOfCordinates = list(zip(result[0], result[1]))
    #print(listOfCordinates)
    p,q=listOfCordinates[0]
    n1=p-win_size//2
    n2=p+win_size//2
    n3=q-win_size//2
    n4=q+win_size//2
    n_up = n1
    n_down=n2
    n_left = n3
    n_right=n4
    if n1<0:
        n_up=np.maximum(0,n1)
        n_down = n2-n1
    if n2>row:
        n_down=np.minimum(n2,row)
        n_up = n1-(n2-row)
    if n3<0:
        n_left=np.maximum(0,n3)
        n_right = n4-n3
    if n4>colum:
        n_right=np.minimum(n4,colum)
        n_left = n3-(n4-colum)
    #img_dct = dct_freq_features(img2d)
    img2d = (img2d - np.min(img2d))/np.max(img2d)
    if win:
        img2d_rs=img2d[n_up:n_down,n_left:n_right]
    else:
        img2d_rs = img2d[::2,::2]
    #print(img2d_rs.shape)
    img1d=np.array(img2d_rs.reshape(-1))
    if img1d.shape[0]!=win_size**2:
        print('img1d shape error')
    #img1d = img_dct

    return img1d

def data_read_win(file_name, win=True, win_size=50):
    " "

    img = np.load(file_name)
    #b = img[:, :, 0]
    #g = img[:, :, 1]
    #r = img[:, :, 2]
    #img_ori = np.stack((r, g, b), axis=2)
    #img2d = 0.2125*img[:,:,0] + 0.7154*img[:,:,1] + 0.0721*img[:,:,2]
    #result = np.where(np.abs(img2d)==np.amax(np.abs(img2d)))
    row,colum,chan = img.shape
    #listOfCordinates = list(zip(result[0], result[1]))
    #p,q=listOfCordinates[0]
    p_sum_0 = 0.
    p_f,q_f = 0, 0
    for p,q in product(list(range(win_size//2,row-win_size//2,8)),list(range(win_size//2,colum-win_size//2,8))):
        n_up,n_down,n_left,n_right = find_boundary(row, colum,p,q,win_size)        
        #img2d = (img2d - np.min(img2d))/np.max(img2d)
        if win:
            #img2d_rs=img2d[n_up:n_down,n_left:n_right]
            img3c = img[n_up:n_down,n_left:n_right,:]
            #img3c = img3c[::2,::2,:]
        else:
            img3c = img[::2,::2,:]
        img3c_norm = (img3c-np.min(img))/np.max(img)
        img1d=np.array(img3c_norm.reshape(-1))
        p_sum = np.sum(np.abs(img3c.reshape(-1)))
        if p_sum > p_sum_0:        
            p_f, q_f = p, q
            p_sum_0 = p_sum
            img1d_f = img1d
            img_save = img3c
    #Image.fromarray(np.uint8(10*img)).save('./snap/'+os.path.basename(file_name).split('.')[0]+'.png')
    #Image.fromarray(np.uint8(10*img_save)).save('./snap/win_'+os.path.basename(file_name).split('.')[0]+'.png')
    #if img1d.shape[0]!=win_size**2*3/4:
    #    print('img1d shape error')

    return img1d_f

def find_boundary(row, colum, p, q, win_size):
    n1=p-win_size//2
    n2=p+win_size//2
    n3=q-win_size//2
    n4=q+win_size//2
    n_up = n1
    n_down=n2
    n_left = n3
    n_right=n4
    if n1<0:
        n_up=np.maximum(0,n1)
        n_down = n2-n1
    if n2>row:
        n_down=np.minimum(n2,row)
        n_up = n1-(n2-row)
    if n3<0:
        n_left=np.maximum(0,n3)
        n_right = n4-n3
    if n4>colum:
        n_right=np.minimum(n4,colum)
        n_left = n3-(n4-colum)
    return n_up, n_down, n_left, n_right

def read_model(meta_file, pert_path):
    PRE_MODEL = []
    with open(meta_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            PRE_MODEL.append((words[0], words[1], words[2], int(words[3]))) 
            # construct dict for perturbations
    file_names = glob.glob(pert_path)
    pert_key = []
    pert_value = []
    for pert_f in file_names:
        pert_key.append(re.split('_', os.path.basename(pert_f))[0])
        pert_value.append(pert_f)
    dict_pert = dict(zip(pert_key,pert_value))
    return PRE_MODEL, dict_pert

def read_prop_mb(pert_path, file_meta):
    ""

    fr = []
    psum = []
    pmean = []
    pvar = []
    gt = []
    ids = []

    img = []
    id_img = []
    filenames =  glob.glob(pert_path)
    for filename in filenames:
        img.append(filename)
        id_img.append(re.split('_', os.path.basename(filename))[0]+'_'+re.split('_',os.path.basename(filename))[3])


    with open(file_meta, 'r') as fobj:
        for line in fobj:
            line = line.rstrip()
            line = line.rstrip('\n')
            line = line.rstrip()
            items = line.split('_')
            fr.append(float(items[10][0:-4]))
            psum.append(float(items[11][0:-5]))
            pmean.append(float(items[12][0:-5]))
            pvar.append(float(items[13]))
            gt.append(int(items[1]))
            ids.append(items[0]+'_'+items[4])

    fooling_rate = np.array(fr)
    psum_arr = np.array(psum)
    pvar = np.array(pvar)
    pmean = np.array(pmean)
    ids_arr = np.array(ids)
    gt_arr = np.array(gt)
    prob = psum_arr/(fooling_rate+1e-6)/100

    #prob_s = np.concatenate((prob.reshape(-1,1), gt_arr.reshape(-1,1),np.array(ids).reshape(-1,1)),axis = 1)
    #embedding = np.concatenate((fooling_rate.reshape(-1,1), psum_arr.reshape(-1,1), prob.reshape(-1,1)), axis=1)
    embedding = prob.reshape(-1,1)
    #embedding = np.concatenate((fooling_rate.reshape(-1,1), psum_arr.reshape(-1,1)), axis=1)
    fr_v_dict = dict(zip(ids,list(embedding))) 
    img_id = dict(zip(id_img, img))
    gt_id = dict(zip(ids,gt))
    return fr_v_dict, img_id,gt_id

def read_prop(pert_path, file_meta):
    ""

    #file_act = '/data1/Adveratt/trojai-round0/code/Pert/prod/test_50/id*.txt'
    #file_meta = '/data1/Adveratt/trojai-round0/code/Pert/prod/test_50/v_meta.txt'
    #files = glob.glob(file_act)
    #file_meta2 = '/data1/Adveratt/trojai-round0/code/Pert/pert_5p/test/v_meta.txt' 

    fr = []
    psum = []
    pmean = []
    pvar = []
    gt = []
    ids = []

    img = []
    id_img = []
    filenames =  glob.glob(pert_path)
    for filename in filenames:
        img.append(filename)
        id_img.append(re.split('_', os.path.basename(filename))[0])


    with open(file_meta, 'r') as fobj:
        for line in fobj:
            line = line.rstrip()
            line = line.rstrip('\n')
            line = line.rstrip()
            items = line.split('_')
            
            fr.append(float(items[8][0:-4]))
            psum.append(float(items[9][0:-5]))
            pmean.append(float(items[10][0:-5]))
            pvar.append(float(items[11]))
            gt.append(int(items[1]))
            ids.append(items[0])

    fooling_rate = np.array(fr)
    psum_arr = np.array(psum)
    pvar = np.array(pvar)
    pmean = np.array(pmean)
    ids_arr = np.array(ids)
    gt_arr = np.array(gt)
    prob = psum_arr/fooling_rate/100

    #prob_s = np.concatenate((prob.reshape(-1,1), gt_arr.reshape(-1,1),np.array(ids).reshape(-1,1)),axis = 1)
    #embedding = np.concatenate((fooling_rate.reshape(-1,1), psum_arr.reshape(-1,1), prob.reshape(-1,1)), axis=1)
    embedding = prob.reshape(-1,1)
    #embedding = np.concatenate((fooling_rate.reshape(-1,1), psum_arr.reshape(-1,1)), axis=1)
    fr_v_dict = dict(zip(ids,list(embedding))) 
    img_id = dict(zip(id_img, img))
    gt_id = dict(zip(ids,gt))
    return fr_v_dict,img_id,gt_id

def read_multi_prop(pert_path, file_meta, num=40):
    ""

    id_img = []
    filenames =  glob.glob(pert_path)
    num = len(filenames)//5
    img_path = [[] for i in range(num)]
    img = [[] for i in range(num)]
    img_key = set()
    img_id_dict = {}
    
    for filename in filenames:
        ids = re.split('_', os.path.basename(filename))[0]
        img_key.add(ids)
    print(len(list(img_key)))

    for filename in filenames:
        ids = re.split('_', os.path.basename(filename))[0]
        for index, key in enumerate(img_key):
            if ids == key:
                img_path[index].append(filename)
    img_id_dict = dict(zip(list(img_key), img_path))

    with open(file_meta, 'r') as fobj:
        fr5 = [[] for i in range(num)]
        psum5 = [[] for i in range(num)]
        pmean5 = [[] for i in range(num)]
        pvar5 = [[] for i in range(num)]
        gt_n_5 = [[] for i in range(num)]
        ids_n_5 = [[] for i in range(num)]
        gt5 = []
        ids5 = []
        i =0
        for line in fobj:
            line = line.rstrip()
            line = line.rstrip('\n')
            line = line.rstrip()
            items = line.split('_')
            #k= math.floor(i//5)       
            for index, key in enumerate(img_key):
                if items[0] == key:
                    fr5[index].append(float(items[10][0:-4]))
                    psum5[index].append(float(items[11][0:-5]))
                    pmean5[index].append(float(items[12][0:-5]))
                    pvar5[index].append(float(items[13]))
                    gt_n_5[index].append(int(items[1]))
                    ids_n_5[index].append(items[0])

        for index, key in enumerate(img_key):          
                gt5.append(gt_n_5[index][0])
                ids5.append(ids_n_5[index][0])       

    fooling_rate5 = np.array(fr5)
    pvar5 = np.array(pvar5)
    pmean5 = np.array(pmean5)
    prob5=np.zeros((num,5),dtype=np.float64)
    p_norm = np.zeros((num,5), dtype=np.float64)
    for i in range(num):
        prob5[i,:]= np.array(psum5[i])/fooling_rate5[i,:]/100
        p_norm[i,:] = (prob5[i,:])/np.max(prob5[i,:])

    p_score_var = np.var(p_norm, axis=1).reshape(-1,1)
    p_score_mean = np.mean(p_norm, axis=1).reshape(-1,1)
    #p_score = np.concatenate((p_norm.min(1).reshape(-1,1),gt_arr5, np.array(ids5).reshape(-1,1)), axis=1)
    embedding_2 = prob5

    #embedding_2 = np.concatenate((fooling_rate5, p_norm), axis=1)
    fr_v_dict = dict(zip(ids5,list(embedding_2)))
    gt_id = dict(zip(list(img_key),gt5))

    for index, key in enumerate(img_id_dict.keys()):
        #print(key)
        #print(ids5[index])
        #print(gt5[index])
        #print(img_id_dict[key])
        #print(list(embedding_2)[index])
        #print(img_id_dict[key])
        if len(img_id_dict[key]) != 5:
            print('number of files does not match the bucket size') 
        if len(gt_n_5[index]) != 5:
            print('number of meta files entry does not match the bucket size')        
        #print()  

    return fr_v_dict,img_id_dict,gt_id

def norm_img(img_path, pert=np.zeros(1), ex_dim=True):
    img = np.load(img_path)
    # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
    #b = img[:, :, 0]
    #g = img[:, :, 1]
    #r = img[:, :, 2]
    #img = np.stack((r, g, b), axis=2)
    # img = np.clip(img + pert,0,255)
    img = np.transpose(img, (2, 0, 1))

    # convert to NCHW dimension ordering
    if ex_dim:
        img = np.expand_dims(img, 0)

    # normalize the image

    img = img - np.min(img)
    img = img / np.max(img)
    #if np.max(pert):
    #    pert = np.transpose(pert, (2, 0, 1))
    #   / if ex_dim:
    #        pert = np.expand_dims(pert,0)
    #    img = img+pert
    #    img = img - np.min(img)
    #    img = img / np.max(img) 
    return img

class Dataset_MS(Dataset):
    def __init__(self, meta_file, pert_path,meta_file_l2, pert_path_l2,num=40):

        self.imgs_s1 = []
        self.imgs_s1_l2 = []
        self.imgs_s2_1 = []
        self.imgs_s2_2 = []
        self.imgs_s2_3 = []
        self.imgs_s2_4 = []
        self.imgs_s2_0 = []
        self.embedding_s1 = []
        self.embedding_s1_l2 = []
        self.embedding_s2 = []
        self.labels = []
        self.labels_l2 = []
        self.labels_multi = []
        self.ids = []

        #self.meta_file_multi, self.pert_path_multi = meta_file_multi, pert_path_multi
        fr_v, id_img,labels = read_prop_mb(pert_path, meta_file)
        fr_v_l2, id_img_l2,labels_l2 = read_prop_mb(pert_path_l2, meta_file_l2)
        #fr_v_multi, id_img_multi, labels_multi = read_multi_prop(pert_path_multi,
        #                                                                 meta_file_multi, num=num)
        for key in fr_v.keys():
            self.embedding_s1.append(fr_v[key])
            self.embedding_s1_l2.append(fr_v_l2[key])
            self.imgs_s1.append(id_img[key])
            self.imgs_s1_l2.append(id_img_l2[key])
            self.labels.append(labels[key])
            self.labels_l2.append(labels_l2[key])
            self.ids.append(key) 
            #key = re.split('_',key)[0]
            #self.embedding_s2.append(fr_v_multi[key])
            #self.imgs_s2_0.append(id_img_multi[key][0])
            #self.imgs_s2_1.append(id_img_multi[key][1])
            #self.imgs_s2_2.append(id_img_multi[key][2])
            #self.imgs_s2_3.append(id_img_multi[key][3])
            #self.imgs_s2_4.append(id_img_multi[key][4])
            #self.labels_multi.append(labels_multi[key]) 
        
    def __getitem__(self, index):
        embed1, img1, label = self.embedding_s1[index], self.imgs_s1[index],self.labels[index]
        embed1_l2, img1_l2, label_l2 = self.embedding_s1_l2[index], self.imgs_s1_l2[index],self.labels_l2[index]
        embed2 = data_read(img1).reshape(1,-1)
        embed2_l2 = data_read(img1_l2).reshape(1,-1)
        #embed_s2, label_multi = self.embedding_s2[index],self.labels_multi[index]

        img1 = norm_img(img1,ex_dim=False)
        img1_l2 = norm_img(img1_l2,ex_dim=False)
        #img_s2_0, img_s2_1, img_s2_2 = self.imgs_s2_0[index], self.imgs_s2_1[index], self.imgs_s2_2[index]
        #img_s2_3, img_s2_4 = self.imgs_s2_3[index], self.imgs_s2_4[index]
        #img_s2_0 = norm_img(img_s2_0, ex_dim=False)
        #img_s2_1 = norm_img(img_s2_1, ex_dim=False)
        #img_s2_2 = norm_img(img_s2_2, ex_dim=False)
        #img_s2_3 = norm_img(img_s2_3, ex_dim=False)
        #img_s2_4 = norm_img(img_s2_4, ex_dim=False)

        img1 = torch.FloatTensor(img1)
        img1_l2 = torch.FloatTensor(img1_l2)
        #img_s2_0 = torch.FloatTensor(img_s2_0)
        #img_s2_1 = torch.FloatTensor(img_s2_1)
        #img_s2_2 = torch.FloatTensor(img_s2_2)
        #img_s2_3 = torch.FloatTensor(img_s2_3)
        #img_s2_4 = torch.FloatTensor(img_s2_4)
        embed1 = torch.FloatTensor(embed1)
        embed1_l2 = torch.FloatTensor(embed1_l2)
        embed2 = torch.FloatTensor(embed2)
        embed2_l2 = torch.FloatTensor(embed2_l2)
        #embed_s2 = torch.FloatTensor(embed_s2)
        return self.ids[index],img1,embed1, embed2,img1_l2,embed1_l2, embed2_l2, label, label_l2

    def __len__(self):
        return len(self.labels)

class Dataset_Test(Dataset):
    def __init__(self, img_train, img_test,model_gtruth, pert=np.zeros(1), transform=None, loader=default_loader):
        self.imgs = []
        self.transform = transform
        self.loader = loader
        self.pert = pert

        # for (_, img_train, img_test, model_gtruth) in enumerate(PRE_MODEL):
            # pert_id = re.split('/', img_train)[-2]
            # img_ori = np.load(dict_pert[pert_id])              
        with open(img_train, 'r') as fh, open(img_test, 'r') as ft:
            for line in fh:
                line = line.rstrip()
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split(',')
                self.imgs.append((words[0],int(model_gtruth)))
            for line in ft:
                line = line.rstrip()
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split(',')
                self.imgs.append((words[0],int(model_gtruth)))

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.fromarray(np.clip(cut(self.loader(fn))+self.pert,0,255).astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)

        return img,label

    def __len__(self):
        return len(self.imgs)
