# coding=utf8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import torch.nn.functional as F
from crack_dataset import crack_loader,cv_imwrite,cv_imread,checkfiles
import numpy as np
from PIL import Image
from torchvision.transforms import transforms


def load_GPUS(model,model_path,kwargs):
    state_dict = torch.load(model_path,**kwargs)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model
    
def load_checkpoint(model,model_file):
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def transform(img):
    transform_pre = transforms.Compose(
            [
                transforms.Resize((360,640)),#((288,512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )
    img = transform_pre(img)
    return img


def hello(a):
    return a
    # return str("0000")


def detect_single_img(dir):
    raw_img = Image.open(dir)
    transform_pre = transforms.Compose(
            [
                # transforms.Resize((512,512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    raw_img = transform_pre(raw_img)
    device = torch.device("cuda:0") 
    from model.unet import UNet
    from model.ddr import ddrnet
    model = ddrnet()
    model.to(device)
    checkpoint = torch.load(r'F:\qt_project\damage_detection\crack_detection_python\pyh\checkpoint_19.pth.tar')
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model_state_dict'].items()})
    model.eval()
    with torch.no_grad():
        output = model(raw_img.unsqueeze(0).float().to(device))
        output = output.squeeze()
        pred = torch.argmax(torch.softmax(output,dim=0),dim=0).squeeze()
        pred = pred.cpu().detach().numpy()
        pred = pred.astype(np.uint8)
        pred = np.array([pred for j in range(3)]).transpose(1,2,0)*255
        cv_imwrite(r'F:/qt_project/damage_detection/crack_detection_python/result' + '/'  + 'qq.jpg',pred)
        dir = str('F:/qt_project/damage_detection/crack_detection_python/result' + '/'  + 'qq.jpg')
        return dir





# a = r'C:\Users\rs\Desktop\多损伤检测\data\crack\img\000422.jpg'
# dir = detect_single_img(a)
# print(dir)
# result = isinstance(dir, str)
# print(result)