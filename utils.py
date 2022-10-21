from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import torch

class MyDataset(Dataset):
    def __init__(self, txt, transform = None, target_transform = None):
        super(MyDataset,self).__init__()
        lists = open(txt, 'r')
        imgs = []
        for line in lists:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        ct, mr = self.imgs[index]
        img_ct = io.imread(ct)
        img_mr = io.imread(mr)
        if self.transform is not None:
            img_ct = self.transform(img_ct)
            img_mr = self.transform(img_mr)
        return img_ct, img_mr

    def __len__(self):
        return len(self.imgs)

# the background
BG_THRESHOLD = 17.0/256
def bg(x1,x2):
    bg_mask = ((x1[0][0] <= BG_THRESHOLD) & (x2[0][0] <= BG_THRESHOLD))
    fg_mask = ((x1[0][0]> BG_THRESHOLD) | (x2[0][0] > BG_THRESHOLD))
    return bg_mask,fg_mask

def scale(image, fg = None):
    img = image[0][0]
    fg_vals = torch.masked_select(img, fg)
    minv = fg_vals.min()
    maxv = fg_vals.max()
    img = (img - minv)
    img = img / (maxv - minv)
    img[img > 1] = 1
    img[img < 0] = 0
    return 1-img

def chg(r, bg_mask, chg_bg = True):
    if chg_bg:
        r[bg_mask] = 0
    return r
