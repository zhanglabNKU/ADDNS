
from __future__ import print_function
import os, random, torchvision, torch
import torch.nn as nn
from torchvision import transforms, utils
import torch.optim as optim
from torch.autograd import Variable
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from skimage.measure import compare_ssim as ssim
from network import addns
from utils import *
#from torch_ssim import ssim as tcssim

import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GLOBAL_SEED = 1
BATCH_SIZE = 1
criterion = nn.MSELoss()
n_epochs = 30

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)



def mi(img1,img2):
    img1_ = np.array(img1.cpu().detach().numpy())
    img2_ = np.array(img2.cpu().detach().numpy())
    # img1_ = np.resize(img1_, (img1_.shape[0], img1_.shape[1]))
    img2_ = np.reshape(img2_, -1)
    img1_ = np.reshape(img1_, -1)
    mutual_infor = metrics.mutual_info_score(img1_, img2_)
    return mutual_infor

def ws(img1,img2):
    p = img1.reshape(-1, )
    q = img2.reshape(-1, )
    s1 = wasserstein_distance(p.cpu().detach().numpy(), q.cpu().detach().numpy())
    # s = torch.from_numpy(s1).cuda()
    return s1

class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg16(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):

        c = torch.cat((fakeIm, fakeIm,fakeIm), dim=0)
        img1 = torch.cat((c, c,c), dim=1)
        # print(img1.shape)

        e = torch.cat((realIm, realIm,realIm), dim=0)
        img2 = torch.cat((e, e,e), dim=1)
        # print(img2.shape)


        # img1 = np.stack((fakeIm,) * 3, axis=-1)
        # img2 = np.stack((realIm,) * 3, axis=-1)
        f_fake = self.contentFunc.forward(img1)
        f_real = self.contentFunc.forward(img2)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


def perceptual_loss(img1,img2):
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    loss_PerceptualLoss = content_loss.get_loss(img1, img2)
    return loss_PerceptualLoss


def calc_loss(ct, mr, img_fusion, outputs):

    loss = criterion(outputs[:, 0, :, :], ct) + criterion(outputs[:, 1, :, :], mr)
    ws_loss = (ws(img_fusion, mr)+ws(img_fusion, mr))/2

    # pixel_loss = (criterion(img_fusion, ct)+criterion(img_fusion, mr))/2
    # mi_loss = (mi(img_fusion, mr)+mi(img_fusion, mr))/2
    # ssim_loss = (tcssim(img_fusion, ct, 11, False) + tcssim(img_fusion, mr, 11, False))[0]
    percep_loss = (perceptual_loss(img_fusion, mr)+perceptual_loss(img_fusion, mr))/2
    loss = loss+ ws_loss+ percep_loss#pixel_loss +ssim_loss##+ mi_loss

    return loss


def train(trainloader,):
    start_time = time.time()
    train_losses = []
    valid_losses = []
    avg_train_losses = []

    avg_valid_losses = []
    net = addns(1, 2)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.03,
                          momentum=0.9, weight_decay=0.0005)

    if not os.path.exists('./model'):
        os.mkdir('./model')

    for epoch in range(1, n_epochs + 1):
        net.train()
        for i, data in enumerate(trainloader, 1):
            ct, mr = data
            ct, mr = ct.to(device), mr.to(device)
            optimizer.zero_grad()

            img_fusion, outputs = net(torch.cat((ct, mr), dim=1))

            loss = calc_loss(ct, mr, img_fusion, outputs)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        net.eval()
        for i, data in enumerate(valloader, 1):
            ct, mr = data
            ct, mr = ct.to(device), mr.to(device)
            img_fusion, outputs = net(torch.cat((ct, mr), dim=1))

            loss = total_loss(ct, mr, img_fusion, outputs)

            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

    torch.save(net.state_dict(), '{}_checkpoint_{}.pt'.format('./models/model_kl',
                                                              epoch))
    print('Done')
    print("time:" % (time.time() - start_time))

def test(net, data_path,txt_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ]
    )

    lists = open(txt_path, 'r')

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

        img_fusion, oo = net(img_ct.to(device), img_mr.to(device))

        oo = oo.cpu()
        img_fusion = img_fusion.cpu()

        if not os.path.exists('./result/'):
            os.mkdir('./result/')

        with torch.no_grad():
            fusion_path = data_path + str(os.path.basename(words[0])[-5:-4]) + '.png'
            # fusion_path = data_path + str(os.path.basename(words[0])[:-4]) + '.png'
            print(fusion_path)
            io.imsave(fusion_path, np.array(img_fusion * 255, dtype='uint8'))



TRAIN = False

if __name__ == "__main__":
    if TRAIN:
        # Training
        train_data = MyDataset(txt='./train_list_whole.txt',
                               transform=transforms.ToTensor())
        # print("traindata type",type(train_data))
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                                  drop_last=True,
                                                  num_workers=2,
                                                  shuffle=True,
                                                  worker_init_fn=worker_init_fn)

        val_data = MyDataset(txt='./val_list_whole.txt',
                             transform=transforms.ToTensor())
        valloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE,
                                                drop_last=True,
                                                num_workers=2,
                                                shuffle=True,
                                                worker_init_fn=worker_init_fn)

        test_data = MyDataset(txt='./test_list_whole.txt',
                              transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,
                                                 drop_last=True,
                                                 num_workers=2,
                                                 shuffle=False,
                                                 worker_init_fn=worker_init_fn)

        train(trainloader,valloader,testloader)
    else:

        time1 = time.time()
        net = addns(1, 2)
        net.to(device)

        data_path = './result/'
        model_path = './model/best_fusion_model_ws_epoch11_Mon_Nov_21_19_53_35_2022_checkpoint.pt'
         # txt_path= "./test_557.txt"

        txt_path= "./demo_test.txt"

        ##different sharing mechanism
        # net.load_state_dict(torch.load('./sharing_mechanism/1234best/best_fusion_model_kl0_checkpoint.pt'),map_location = device)
        # net.load_state_dict(torch.load('./sharing_mechanism/1298/best_fusion_model_kl0_checkpoint.pt'),map_location = device)
        # net.load_state_dict(./sharing_mechanism/9876/best_fusion_model_kl0_checkpoint.pt'),map_location = device)
        # net.load_state_dict(/./sharing_mechanism/34567/best_fusion_model_kl0_checkpoint.pt'),map_location = device)
        ## our fusion model
        net.load_state_dict(torch.load(model_path, map_location=device))
        
        net.eval()
        test(net, data_path,txt_path)
        print("spend time", time.time() - time1)



