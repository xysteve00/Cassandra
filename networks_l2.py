import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import torchvision.models as models
from pointnet_30_30 import *
#from pointnet2 import *

class StreamOne(nn.Module):
    def __init__(self):
        super(StreamOne, self).__init__()
        self.embedding = PointNetCls()  
        self.embedding_mlp = PointNetCls(num_p=5)
    def get_embedding(self, x):
        #x = self.embedding(x)
        #x = x.view(x.size(0), -1)
        return x
    def get_embedding_pn(self, x):
        #print(x.size())
        x = self.embedding(x)
        #x = x.view(x.size(0), -1)
        return x
    def get_embedding_mlp(self,x):
        x = self.embedding_mlp(x)
        #x = x.view(x.size(0), -1)
        return x        


class StreamTwo(nn.Module):
    def __init__(self, model, slnum=1280, dense=512,num_classes=2,out_two=False):
        super(StreamTwo, self).__init__()

        self.net_layer = nn.Sequential(*list(model.children())[:-1])
        self.fc_dropout = nn.Dropout(p=0.5)

        self.projection = nn.Linear(slnum, dense) # output layer
        self.classifier = nn.Linear(dense,num_classes)  
        self.Linear_layer = nn.Linear(slnum, num_classes)
        self.out_two = out_two

    def get_embedding(self, x):
        x = self.net_layer(x)
        x = x.view(x.size(0), -1)
        if self.out_two:
            x = F.relu(self.projection(x))
            x = self.fc_dropout(x)
            x = self.classifier(x)
            return x 
        #x = self.Linear_layer(x)
        return x

class ClassificationNet(nn.Module):
    def __init__(self, model, StreamOne = StreamOne, StreamTwo=StreamTwo,dense=512, n_classes=2,
                 drop_out=True,out_two=False):
        super(ClassificationNet, self).__init__()
        self.embedding_one = StreamOne()
        self.embedding_two = StreamTwo(model,out_two=False) 
        self.embedding_two_l2 = StreamTwo(model,out_two=False)

        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(4, 2) if out_two else nn.Linear(1280+256, n_classes)
        self.fc1_m = nn.Linear(4, 2) if out_two else nn.Linear(1280, n_classes)
        self.fc1_t = nn.Linear(4, 2) if out_two else nn.Linear(2048, n_classes)
        self.fc_dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(1281,n_classes) 
        self.classifier_2560 = nn.Linear(2562,n_classes) 
        self.classifier_m = nn.Linear(1280*5+256,n_classes)  
        self.classifier_hidden = nn.Linear(1280*5, 1280) 
        self.classifier_t = nn.Linear(512*5,n_classes) 
        self.drop_flag = drop_out    

    def forward_stream_one(self, x, e1, e2):
        output1 = self.embedding_one.get_embedding(e1)
        output_pn = self.embedding_one.get_embedding_pn(e2)
        #print(output1.size())
        output2 = self.embedding_two.get_embedding(x)
        #print(output2.size())
        output = self.fc1(torch.cat((output1, output2),dim=-1))
        if self.drop_flag:
            output = self.classifier(self.fc_dropout(self.nonlinear(output)))

        return output

    def forward(self,x,e1,e2,x_l2,e1_l2,e2_l2,stream=2):
        # concatenate 5 perturbations as 15 channels
        if stream == 3:
            input_concat = torch.cat((img0,img1,img2,img3,img4),dim=1)
            output = self.embedding_two.get_embedding(input_concat)
            if self.drop_flag:
                output = self.fc1_m(self.fc_dropout(self.nonlinear(output)))
            else:
                output = self.fc1_m(output)
        elif stream == 1:
            # stream one
            output1 = self.embedding_one.get_embedding(e1)
            #output_pn = self.embedding_one.get_embedding_pn(e2)
            output2 = self.embedding_two.get_embedding(x)
            #output = self.fc1_m(self.fc_dropout(output2))
            #output = self.fc1(torch.cat((output2,output1),dim=-1))
            if self.drop_flag:
                output = self.classifier(self.fc_dropout(self.nonlinear(torch.cat((output1, output2),dim=-1))))
                #output = self.fc1(self.fc_dropout(torch.cat((output_pn,output2),dim=-1)))
            #output = F.softmax(output, dim=1)
        elif stream == 2:
            # stream one
            output1 = self.embedding_one.get_embedding(e1)
            #output_pn = self.embedding_one.get_embedding_pn(e2)
            output2 = self.embedding_two.get_embedding(x)
            output2_l2 = self.embedding_two_l2.get_embedding(x_l2)
            #output = self.fc1_m(self.fc_dropout(output2))
            if self.drop_flag:
                #output = self.classifier(self.fc_dropout(self.nonlinear(torch.cat((output1, output2),dim=-1))))
                output = self.classifier_2560(self.fc_dropout(self.nonlinear(torch.cat((output2,output1, output2_l2,e1_l2),dim=-1))))
                #output = self.fc1(self.fc_dropout(torch.cat((output_pn,output2),dim=-1)))
            output = F.softmax(output, dim=1)
        else:
            # Stream two : each perturbation as one stream
        
            #output_s = self.embedding_two.get_embedding(x)
            #output_e1 = self.embedding_one.get_embedding(e1)

            output0 = self.embedding_two.get_embedding(img0)
            output1 = self.embedding_two.get_embedding(img1)
            output2 = self.embedding_two.get_embedding(img2)
            output3 = self.embedding_two.get_embedding(img3)
            output4 = self.embedding_two.get_embedding(img4)
            output_e3 = self.embedding_one.get_embedding(e3)
            # mlp input without concatenation
            output_e2 = self.embedding_one.get_embedding_pn(e2)
            # concatenate properties before mlp
            #output_e2_e3 = torch.cat((e2, output_e3.unsqueeze(1)), dim=-1)
            #output_e2_e3 = self.embedding_one.get_embedding_mlp(output_e2_e3)
            # concatenate without mlp

            # embedding before classification
            output_cct = torch.cat((output0, output1, output2, output3, output4, output_e2), dim=-1)
            output = self.classifier_m(self.fc_dropout(self.nonlinear(output_cct)))

            # stream_one + stream_two
            #output = self.classifier_hidden(self.fc_dropout(self.nonlinear(output_cct)))
            #output1 = self.embedding_one.get_embedding(e1)
            #output = self.fc1(torch.cat((output1, output),dim=-1))

        return output




