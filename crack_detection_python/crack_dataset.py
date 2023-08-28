# -*- coding: utf-8 -*-
import os
import shutil
import torch
# from torch.utils.data import *
from torch.utils import data
from imutils import paths
import numpy as np
import random
from PIL import Image
from torchvision.transforms import transforms
import cv2

def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),-1)
    return img 
def cv_imwrite(path,img):
    cv2.imencode('.jpg', img)[1].tofile(path)

def checkfiles(root):
    if(not os.path.exists(root)):
        os.mkdir(root)
    else:
        shutil.rmtree(root)
        os.mkdir(root)
def ensurefiles(root):
    if(not os.path.exists(root)):
        os.mkdir(root)
    else:
        pass
    
class crack_loader(data.Dataset):
    def __init__(self, img_file, imgSize, PreprocFun=None):
        with open(img_file,'r',encoding='utf-8') as f:
            lines = f.readlines()
        self.img_paths = [line.strip().split(',') for line in lines]
        self.img_size = imgSize
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        raw_img_path,label_img_path,mask_img_path = self.img_paths[index]
        raw_img = Image.open(raw_img_path)
        # raw_img = cv_imread(raw_img_path)
        label_img = cv_imread(label_img_path)
        mask_img = cv_imread(mask_img_path)
        try:        
            height1, width1 = label_img.shape
        except:
            label_img = label_img.mean(2)
            height1, width1 = label_img.shape
        if height1 != self.img_size or width1 != self.img_size:
            label_img = cv2.resize(label_img, (self.img_size,self.img_size))
        raw_img = self.PreprocFun(raw_img)

        return raw_img, torch.tensor(label_img/255.0,dtype=torch.long), torch.tensor(mask_img,dtype=torch.long)#.unsqueeze(0)#torch.tensor(raw_img,dtype=torch.float32) 

    def transform(self, img):
        transform_pre = transforms.Compose(
            [
                transforms.Resize((512,512)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
        img = transform_pre(img)
        return img
        

if __name__ == "__main__":
    train_set = r'./scheme_set/train.txt'
    train_loader = crack_loader(train_set,512)
    for i in train_loader:
        print(i)
