from sklearn import metrics as mr
# from scipy.misc import imread
from skimage.measure import compare_ssim
import numpy as np
from PIL import Image


##MI
def mi(img1,img2):
    img1 = np.resize(img1, (img1.shape[0], img1.shape[1]))
    img2 = np.reshape(img2, -1)
    img1 = np.reshape(img1, -1)
    # print(img2.shape)
    # print(img1.shape)
    mutual_infor = mr.mutual_info_score(img1, img2)
    return mutual_infor

##ssim
def ssim(img1,img2):
    ssim = compare_ssim(img1, img2, multichannel=True)
    return ssim




# if __name__ == "__main__":
#     img_ct = "./data200/case5/ct1_014.gif"
#     img_mr = "./data200/case5/mr3_014.gif"
#
#     img2 = "./img/fusion_result.tif"
#     img3 = "./Medical-Image-Fusion-master/fusion_result.tif"
#     img_fw = "./fusion_result.tif"
#
#     im_ct = Image.open(img_ct)
#     im_mr = Image.open(img_mr)
#     im2 = Image.open(img2)
#     im3 = Image.open(img3)
#     im_fw = Image.open(img_fw)
#
#     im2_ct_ssim = ssim(np.array(im2),np.array(im_ct))
#     im2_mr_ssim = ssim(np.array(im2), np.array(im_mr))
#     im2_ct_mi = mi(np.array(im2),np.array(im_ct))
#     im2_mr_mi = mi(np.array(im2), np.array(im_mr))
#     print("\nimg2_ct_ssim:%.4f, img2_mr_ssim:%.4f, im2_ct_mi:%.4f, im2_mr_mi:%.4f"\
#           %(im2_ct_ssim,im2_mr_ssim,im2_ct_mi,im2_mr_mi))
#
#     im3_ct_ssim = ssim(np.array(im3), np.array(im_ct))
#     im3_mr_ssim = ssim(np.array(im3), np.array(im_mr))
#     im3_ct_mi = mi(np.array(im3), np.array(im_ct))
#     im3_mr_mi = mi(np.array(im3), np.array(im_mr))
#     print("\nimg3_ct_ssim:%.4f, img3_mr_ssim:%.4f, im3_ct_mi:%.4f, im3_mr_mi:%.4f"\
#           % (im3_ct_ssim, im3_mr_ssim, im3_ct_mi, im3_mr_mi))
#
#     ImFw_ct_ssim = ssim(np.array(im_fw), np.array(im_ct))
#     ImFw_mr_ssim = ssim(np.array(im_fw), np.array(im_mr))
#     ImFw_ct_mi = mi(np.array(im_fw), np.array(im_ct))
#     ImFw_mr_mi = mi(np.array(im_fw), np.array(im_mr))
#     print("\nImFw_ct_ssim:%.4f, ImFw_mr_ssim:%.4f, ImFw_ct_mi:%.4f, ImFw_mr_mi:%.4f"\
#           % (ImFw_ct_ssim, ImFw_mr_ssim, ImFw_ct_mi, ImFw_mr_mi))
