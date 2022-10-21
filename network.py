from parts_1234 import *
import torch.nn as nn
from utils import *
class addns(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(addns, self).__init__()

        ## the framework without the sharing mechanism
        # self.inc_gen = indense(2, 2, 64, 32)
        # self.down1_gen = dense_down(2, 32, 64, 64)
        # self.down2_gen = dense_down(2, 64, 64, 128)
        # self.down3_gen = dense_down(2, 128, 64, 256)
        # self.down4_gen = dense_down(2, 256, 64, 256)
        # self.up1_gen = dense_up(2, 512, 64, 128)
        # self.up2_gen = dense_up(2, 256, 64,  64)
        # self.up3_gen = dense_up(2, 128, 64, 32)
        # self.up4_gen = dense_up(2, 64, 64, 32)
        # self.outc_gen = outconv(32, 1)
        # self.outc = outconv(32, n_classes)
        # self.inc = indense(2, n_channels, 64,32)
        # self.down1 = down(32, 64)
        # self.down2 = down(64, 128)
        # self.down3 = down(128, 256)
        # self.down4 = down(256, 256)
        # self.up1 = up(512, 128)
        # self.up2 = up(256, 64)
        # self.up3 = up(128, 32)
        # self.up4 = up(64, 32)
        # self.outc = outconv(32, n_classes)
        # self.relu = nn.ReLU(inplace=True)


        self.inc_gen = indense(2, 2, 64, 32)
        self.trans1 = Transition(130,32)
        self.trans12 = Transition(130,32)
        #
        self.down1_gen = dense_down(2, 32, 64, 64)
        self.trans2 = Transition(160, 64)
        self.trans22 = Transition(160, 64)
        #
        self.down2_gen = dense_down(2, 64, 64, 128)
        self.trans3 = Transition(192, 64*2)
        self.trans32 = Transition(192, 64*2)
        #
        self.down3_gen = dense_down(2, 128, 64, 256)
        self.trans4 = Transition(256, 128*2)
        self.trans42 =Transition(256, 128*2)
        #
        self.down4_gen = dense_down(2, 256, 64, 256)
        self.trans5 = Transition(256+128, 128*2)
        self.trans52 =Transition(256+128, 128*2)

        self.up1_gen = dense_up(2, 256*2, 64, 128)
        self.trans6 = Transition(512+128, 64*2)
        self.trans62 =Transition(512+128, 64*2)


        self.up2_gen = dense_up(2, 128*2, 64, 64)
        self.trans7 = Transition(256+128, 64)
        self.trans72 = Transition(256+128, 64)

        self.up3_gen = dense_up(2, 64*2, 64, 32)
        self.trans8 = Transition(192+64, 32)
        self.trans82 =Transition(192+64, 32)
        #
        self.up4_gen = dense_up(2, 64, 64, 32)
        self.trans9 = Transition(192, 32)
        self.trans92 = Transition(192, 32)
        self.outc_gen = out(32, 1)

        self.inc = conv(n_channels, 192)
        # self.trans9 = Transition(192, 32)
        self.down1 = down(32, 192+64)
        # self.trans8 = Transition(192, 32)
        self.down2 = down(32, 256+128)
        # self.trans7 = Transition(256, 32)
        self.down3 = down(64, 512+128)
        # self.trans6 = Transition(384, 64)
        self.down4 = down(128, 256+128)
        # self.trans5 = Transition(256, 128)
        self.up1 = up(192*2, 256)
        # self.trans4 = Transition(192, 128)
        self.up2 = up(160*2, 192)
        # self.trans3 = Transition(160, 64)
        self.up3 = up(160, 160)
        # self.trans2 = Transition(160, 32)
        self.up4 = up(96, 130)
        # self.trans1 = Transition(130, 32)
        self.outc = out(32, n_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1,x2):
        x11 = self.inc_gen(torch.cat((x1,x2), dim = 1))
        x_bg,x_fg = bg(x1,x2)
        x11 = self.trans1(x11)

        x22 = self.down1_gen(x11)
        x22 = self.trans2(x22)

        x3 = self.down2_gen(x22)
        x3 = self.trans3(x3)

        x4 = self.down3_gen(x3)
        x4 = self.trans4(x4)

        x5 = self.down4_gen(x4)
        x5 = self.trans5(x5)

        x = self.up1_gen(x5, x4)
        x6 = self.trans6(x)

        x = self.up2_gen(x6, x3)
        x7 = self.trans7(x)

        x = self.up3_gen(x7, x22)
        x8 = self.trans8(x)

        x = self.up4_gen(x8, x11)
        x9 = self.trans9(x)
        x = self.outc_gen(x9)

        fr = F.sigmoid(x)
        r = scale(fr,x_fg)
        r = chg(r, x_bg)

        x11 = self.inc(fr)
        x11 = self.trans92(x11)

        x22 = self.down1(x11)
        x22 = self.trans82(x22)
        #
        x3 = self.down2(x22)
        x3 = self.trans72(x3)
        #
        x4 = self.down3(x3)
        x4 = self.trans62(x4)
        #
        x5 = self.down4(x4)
        x5 = self.trans52(x5)
        x = self.up1(x5, x4)
        x= self.trans4(x)
        #
        x = self.up2(x, x3)
        x = self.trans3(x)
        x = self.up3(x, x22)
        x = self.trans2(x)
        x = self.up4(x, x11)
        x = self.trans1(x)

        x = self.outc(x)

        return r, F.sigmoid(x)



