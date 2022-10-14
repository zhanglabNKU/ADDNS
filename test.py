

import numpy as np
import os, torchvision, torch, sys
# import torch.nn as nn
from torchvision import transforms, utils
# from torch.autograd import Variable
# import skimage.io as io
# import numpy as np
# from torchvision import models
# from skimage.measure import compare_ssim as ssim
from network import addns
from utils import *
import warnings

# from evaluate import *
import time

from skimage import data, exposure, img_as_float
# import matplotlib.pyplot as plt

# warnings.filterwarnings("ignore")
time1=time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
net = addns(1, 2)
net.to(device)
data_path = './result/'

##different sharing mechanism
# net.load_state_dict(torch.load('./sharing_mechanism/1234best/best_fusion_model_kl0_checkpoint.pt'),map_location = device)
# net.load_state_dict(torch.load('./sharing_mechanism/1298/best_fusion_model_kl0_checkpoint.pt'),map_location = device)
# net.load_state_dict(./sharing_mechanism/9876/best_fusion_model_kl0_checkpoint.pt'),map_location = device)
# net.load_state_dict(/./sharing_mechanism/34567/best_fusion_model_kl0_checkpoint.pt'),map_location = device)

## our fusion model
net.load_state_dict(torch.load('./model/best_fusion_model_kl0_checkpoint.pt',map_location = device))

net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    ]
)


# lists = open("/home/hww/fusion0820/data/test_557.txt", 'r')
lists = open("./demo_test.txt", 'r')


ssim_ct=[]
ssim_mr = []
mi_ct = []
mi_mr = []

for line in lists:
    line = line.strip('\n')
    line = line.rstrip()
    words = line.split()
    img_ct_read = io.imread(words[0])
    img_mr_read = io.imread(words[1])

    img_ct = transform(img_ct_read)
    img_mr = transform(img_mr_read)

    img_ct = torch.unsqueeze(img_ct, 1)
    img_mr = torch.unsqueeze(img_mr, 1)

    img_fusion, oo = net(img_ct.to(device),img_mr.to(device))
    print("spend time",time.time()-time1)
    oo = oo.cpu()
    img_fusion = img_fusion.cpu()


    if not os.path.exists('./result/'):
        os.mkdir('./result/')
        

    with torch.no_grad():
        fusion_path = data_path+str(os.path.basename(words[0])[-5:-4]) +'.png'
        print(fusion_path)
        io.imsave(fusion_path, np.array(img_fusion * 255, dtype='uint8'))






