import torch
import torch.nn as nn
import torch.nn.functional as F
from . import model_irse
import numpy as np
import torchvision
from PIL import Image
class PerceptLoss(object):

    def __init__(self):
        pass

    def __call__(self, LossNet, fake_img, real_img):
        with torch.no_grad():
            real_feature = LossNet(real_img.detach())
        fake_feature = LossNet(fake_img)
        perceptual_penalty = F.mse_loss(fake_feature, real_feature)
        return perceptual_penalty

    def set_ftr_num(self, ftr_num):
        pass


class DiscriminatorLoss(object):

    def __init__(self, ftr_num=4, data_parallel=False):
        self.data_parallel = data_parallel
        self.ftr_num = ftr_num

    def __call__(self, D, fake_img, real_img):
        if self.data_parallel:
            with torch.no_grad():
                d, real_feature = nn.parallel.data_parallel(
                    D, real_img.detach())
            d, fake_feature = nn.parallel.data_parallel(D, fake_img)
        else:
            with torch.no_grad():
                d, real_feature = D(real_img.detach())
            d, fake_feature = D(fake_img)
        D_penalty = 0
        print(len(fake_feature))
        for i in range(self.ftr_num):
            f_id = -i - 1   # i:0123, f_id: -1, -2, -3, -4  # 最后一层to 倒数第4层
            D_penalty = D_penalty + F.l1_loss(fake_feature[f_id],
                                              real_feature[f_id])
        return D_penalty

    def set_ftr_num(self, ftr_num):
        self.ftr_num = ftr_num

class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

class GradientLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)




class ArcFaceLoss(object):
    def __init__(self):
        self.model = model_irse.IR_152([112, 112])
        ckpt = torch.load("/disk2/yingqing/data/irse-pretrained/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth")
        self.model.load_state_dict(ckpt)
        self.model.cuda().eval()
        # print(ckpt.keys())
    def __call__(self, gen_img, target_img):
        gen_img = resize_image(gen_img, 112).cuda()
        target_img = resize_image(target_img, 112).cuda()

        emb1 = self.model(gen_img).squeeze()
        emb2 = self.model(target_img).squeeze()
        # from numpy.linalg import norm
        sim = torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2))
        return sim


def resize_image(img, size):
    trans = torchvision.transforms.Resize(size)
    img = trans(torch_to_pil(img))
    img_torch = pil_to_torch(img)
    return img_torch

def g_loss(D, fake_img):
    fake_pred = D(fake_img)
    loss = F.softplus(-fake_pred).mean()

    return loss

def torch_to_pil(img):
    img_np = img.clone().detach().cpu().numpy()[0]
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)
    return Image.fromarray(ar)
    
def pil_to_torch(img):
    ar = np.array(img)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    img_np = ar.astype(np.float32) / 255.
    return torch.from_numpy(img_np)[None, :]

def l_loss(img1, img2):
    img1_y = 0.257 * img1[0, 0, :, :] + 0.564 * img1[0, 1, :, :] + 0.098 * img1[0, 2, :, :] + 16
    img2_y = 0.257 * img2[0, 0, :, :] + 0.564 * img2[0, 1, :, :] + 0.098 * img2[0, 2, :, :] + 16
    print(img2_y.size(), img1_y.size(), img1.size(), img2.size())
    print
    loss = torch.nn.functional.mse_loss(img1_y, img2_y)
    return loss 