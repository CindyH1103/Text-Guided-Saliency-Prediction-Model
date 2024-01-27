import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from config import nb_timestep, epsilon_DC, b_s, shape_r_out, shape_c_out, \
    shape_r_f, shape_c_f, nb_gaussian
from BLIP.blip_pretrain import BLIP_Pretrain
from sklearn.metrics import roc_auc_score
import math
import torch.utils.model_zoo as model_zoo
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from TransformerEncoder import transEncoder
from torch.autograd import Variable
import resnet
from config import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# -----------------------------------------------------------------------------------------------------------------------
# define the main frame of the dilated ResNet 50
class MyDRN(nn.Module):

    def __init__(self):
        super(MyDRN, self).__init__()

        # conv_1
        self.zeropad = nn.ZeroPad2d(padding=3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                               padding=0, dilation=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)  # to save the computing memory: inplace=True
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv_2
        self.myConvBlock1 = MyConvBlock(kernel_size=3, filters=[64, 64, 256], stride=1, in_channel=64,
                                        dilation=1, padding=1)
        self.myIdentityBlock1 = MyIdentityBlock(kernel_size=3, filters=[64, 64, 256], in_channel=256,
                                                dilation=1, padding=1)
        self.myIdentityBlock2 = MyIdentityBlock(kernel_size=3, filters=[64, 64, 256], in_channel=256,
                                                dilation=1, padding=1)

        # conv_3
        self.myConvBlock2 = MyConvBlock(kernel_size=3, filters=[128, 128, 512], stride=2, in_channel=256,
                                        dilation=1, padding=1)
        self.myIdentityBlock3 = MyIdentityBlock(kernel_size=3, filters=[128, 128, 512], in_channel=512,
                                                dilation=1, padding=1)
        self.myIdentityBlock4 = MyIdentityBlock(kernel_size=3, filters=[128, 128, 512], in_channel=512,
                                                dilation=1, padding=1)
        self.myIdentityBlock5 = MyIdentityBlock(kernel_size=3, filters=[128, 128, 512], in_channel=512,
                                                dilation=1, padding=1)

        # conv_4
        self.myConvBlock3 = MyConvBlock(kernel_size=3, filters=[256, 256, 1024], stride=1, in_channel=512,
                                        dilation=2, padding=2)
        self.myIdentityBlock6 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                dilation=2, padding=2)
        self.myIdentityBlock7 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                dilation=2, padding=2)
        self.myIdentityBlock8 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                dilation=2, padding=2)
        self.myIdentityBlock9 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                dilation=2, padding=2)
        self.myIdentityBlock10 = MyIdentityBlock(kernel_size=3, filters=[256, 256, 1024], in_channel=1024,
                                                 dilation=2, padding=2)

        # conv_5
        self.myConvBlock4 = MyConvBlock(kernel_size=3, filters=[512, 512, 2048], stride=1, in_channel=1024,
                                        dilation=4, padding=4)
        self.myIdentityBlock11 = MyIdentityBlock(kernel_size=3, filters=[512, 512, 2048], in_channel=2048,
                                                 dilation=4, padding=4)
        self.myIdentityBlock12 = MyIdentityBlock(kernel_size=3, filters=[512, 512, 2048], in_channel=2048,
                                                 dilation=4, padding=4)

        # conv_feat
        self.convfeat = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, stride=1,
                                  padding=1, dilation=1, groups=1, bias=False)

    def forward(self, x):
        # layer_1
        x = self.zeropad(x)  # [8,3,240,320] -> [8,3,246,326]
        x = self.conv1(x)  # [8,3,246,326] -> [8,64,120,160]
        x = self.bn1(x)  # size equal
        x = self.relu(x)  # size equal
        x = self.maxpool1(x)  # [8,64,120,160] -> [8,64,60,80]

        # layer_2
        x = self.myConvBlock1(x)  # size equal
        x = self.myIdentityBlock1(x)  # size equal
        x = self.myIdentityBlock2(x)  # size equal

        # layer_3
        x = self.myConvBlock2(x)  # [8,64,60,80] -> [8,512,30,40]
        x = self.myIdentityBlock3(x)  # size equal
        x = self.myIdentityBlock4(x)  # size equal
        x = self.myIdentityBlock5(x)  # size equal

        # layer_4
        x = self.myConvBlock3(x)  # [8,512,30,40] -> [8,1024,30,40]
        x = self.myIdentityBlock6(x)  # size equal
        x = self.myIdentityBlock7(x)  # size equal
        x = self.myIdentityBlock8(x)  # size equal
        x = self.myIdentityBlock9(x)  # size equal
        x = self.myIdentityBlock10(x)  # size equal

        # layer_5
        x = self.myConvBlock4(x)  # [8,1024,30,40] -> [8,2048,30,40]
        x = self.myIdentityBlock11(x)  # size equal
        x = self.myIdentityBlock12(x)  # size equal

        # layer_output
        x = self.convfeat(x)  # [8,2048,30,40] -> [8,512,30,40]

        return x


# 
# -----------------------------------------------------------------------------------------------------------------------
# deine the accessories for the proposed DRN frame
class MyIdentityBlock(nn.Module):

    def __init__(self, kernel_size, filters, in_channel, dilation, padding):
        super(MyIdentityBlock, self).__init__()
        nb_filter1, nb_filter2, nb_filter3 = filters
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=nb_filter1, kernel_size=1, stride=1,
                               padding=0, dilation=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_filter1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=nb_filter1, out_channels=nb_filter2, kernel_size=kernel_size,
                               stride=1, padding=padding, dilation=dilation, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_filter2)
        self.conv3 = nn.Conv2d(in_channels=nb_filter2, out_channels=nb_filter3, kernel_size=1, stride=1,
                               padding=0, dilation=1, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nb_filter3)

    def forward(self, x):
        Input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = Input + x
        x = self.relu(x)

        return x


class MyConvBlock(nn.Module):

    def __init__(self, kernel_size, filters, stride, in_channel, dilation, padding):
        super(MyConvBlock, self).__init__()
        nb_filter1, nb_filter2, nb_filter3 = filters
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=nb_filter1, kernel_size=1, stride=stride,
                               padding=0, dilation=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_filter1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=nb_filter1, out_channels=nb_filter2, kernel_size=kernel_size,
                               stride=1, padding=padding, dilation=dilation, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nb_filter2)
        self.conv3 = nn.Conv2d(in_channels=nb_filter2, out_channels=nb_filter3, kernel_size=1, stride=1,
                               padding=0, dilation=1, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nb_filter3)
        self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=nb_filter3, kernel_size=1, stride=stride,
                                  padding=0, dilation=1, groups=1, bias=False)

    def forward(self, x):
        y = self.shortcut(x)
        y = self.bn3(y)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = y + x
        x = self.relu(x)

        return x


# ----------------------------------------------------------------------------------------------------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out.append(x)
        # print(x.shape)
        x = self.layer3(x)
        out.append(x)
        # print(x.shape)
        x = self.layer4(x)
        out.append(x)

        return out


def resnet50_backbone():
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
    save_model = model_zoo.load_url(model_urls['resnet50'])
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model
# -----------------------------------------------------------------------------
# define the main frame of the attentive convLSTM
class MyAttentiveLSTM(nn.Module):

    def __init__(self, nb_features_in, nb_features_out, nb_features_att, nb_rows, nb_cols, w, h, embed_dim=512):
        super(MyAttentiveLSTM, self).__init__()

        # define the fundamantal parameters
        self.nb_features_in = nb_features_in
        self.nb_features_out = nb_features_out
        self.nb_features_att = nb_features_att
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols

        # define convs
        self.W_a = nn.Conv2d(in_channels=self.nb_features_att, out_channels=self.nb_features_att,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_a = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_att,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.V_a = nn.Conv2d(in_channels=self.nb_features_att, out_channels=1,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=False)

        self.W_i = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_i = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.W_f = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_f = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.W_c = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_c = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        self.W_o = nn.Conv2d(in_channels=self.nb_features_in, out_channels=self.nb_features_out,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.U_o = nn.Conv2d(in_channels=self.nb_features_out, out_channels=self.nb_features_out,
                             kernel_size=self.nb_rows, stride=1, padding=1, dilation=1, groups=1, bias=True)

        # define activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # define number of temporal steps
        text_width = 256
        self.W_Q = nn.Linear(in_features=text_width, out_features=embed_dim * w * h, bias=False)
        self.W_K = nn.Linear(in_features=1, out_features=embed_dim, bias=False)
        self.W_V = nn.Linear(in_features=1, out_features=embed_dim, bias=False)
        self.W_sum = nn.Linear(in_features=embed_dim, out_features=1)
        self.embed_dim = embed_dim
        self.sqrt_dim = np.sqrt(embed_dim)
        self.nb_ts = nb_timestep
        self.w = w
        self.h = h

    def forward(self, x, txt):
        # gain the current cell memory and hidden state
        h_curr = x
        c_curr = x
        txt = txt.view(txt.shape[0], txt.shape[2])
        # print(x.shape)

        for i in range(self.nb_ts):
            # the attentive model
            my_Z = self.V_a(self.tanh(self.W_a(h_curr) + self.U_a(x)))
            my_A_ = self.softmax(my_Z)

            # cross attention with text information
            Q_ = self.W_Q(txt).view(-1, self.w, self.h, self.embed_dim)
            K_ = self.W_K(my_A_.permute(0, 2, 3, 1))  # batch, w, h, c
            V_ = self.W_V(my_A_.permute(0, 2, 3, 1))  # batch, w, h, c

            # A_ = torch.bmm(Q_, K_.transpose(2, 3)) / self.sqrt_dim
            A_ = self.W_sum(self.softmax((Q_ * K_).sum(dim=3, keepdim=True) / self.sqrt_dim).repeat(1, 1, 1, self.embed_dim) * V_).permute(0, 3, 1, 2)
            my_A = my_A_ + A_

            AM_cL = my_A * x
            # the convLSTM model
            my_I = self.sigmoid(self.W_i(AM_cL) + self.U_i(h_curr))
            my_F = self.sigmoid(self.W_f(AM_cL) + self.U_f(h_curr))
            my_O = self.sigmoid(self.W_o(AM_cL) + self.U_o(h_curr))
            my_G = self.tanh(self.W_c(AM_cL) + self.U_c(h_curr))
            c_next = my_G * my_I + my_F * c_curr
            h_next = self.tanh(c_next) * my_O

            c_curr = c_next
            h_curr = h_next

        return h_curr


# define the non local neural network block
class MyNonLocBlock(nn.Module):
    pass


# define the learnable priors
class MyPriors(nn.Module):

    def __init__(self, gp):
        super(MyPriors, self).__init__()
        self.conv5_5 = nn.Conv2d(in_channels=(512 + nb_gaussian), out_channels=512, kernel_size=5,
                                 stride=1, padding=8, dilation=4, groups=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.gp = gp

    def forward(self, x):
        # cancate the generated gaussian priors with the original inputs
        gp_seq = []
        for c in range(b_s):
            gp_seq.append(self.gp)
        gp_seq = torch.cat(gp_seq, 0)
        x = torch.cat([gp_seq, x], 1)

        # the conv 5*5 (to learn the feature of center bias from generated gaussian kernels)
        x = self.conv5_5(x)
        x = self.relu(x)

        return x


def generate_gaussian_prior():
    # initialize the gaussian feature clip
    gp = torch.zeros(nb_gaussian, shape_r_f, shape_c_f)

    # initialize the gaussian distribution along x and y in each of the feature  map
    f_r = torch.zeros(shape_r_f, nb_gaussian)
    f_c = torch.zeros(shape_c_f, nb_gaussian)

    # lock the generated parameters
    random.seed(666)

    # randomly initialize the delta and mean for the gaussian prior
    delta_x = torch.zeros(nb_gaussian, 1)
    delta_y = torch.zeros(nb_gaussian, 1)
    for r in range(nb_gaussian):
        delta_x[r, 0] = random.uniform(0.1, 0.9)
        delta_y[r, 0] = random.uniform(0.2, 0.8)
    sigma_x = delta_x * (0.3 * ((shape_r_f - 1) * 0.5 - 1) + 0.8)
    sigma_y = delta_y * (0.3 * ((shape_c_f - 1) * 0.5 - 1) + 0.8)

    # randomly initialize gaussian distribution along X and Y in each of the feature map
    for gr in range(nb_gaussian):
        gaussian_cur = cv2.getGaussianKernel(shape_r_f, sigma_x[gr, 0].item())
        gaussian_cur = torch.from_numpy(gaussian_cur)
        f_r[:, gr] = gaussian_cur[:, 0]
    for gc in range(nb_gaussian):
        gaussian_cur = cv2.getGaussianKernel(shape_c_f, sigma_y[gc, 0].item())
        gaussian_cur = torch.from_numpy(gaussian_cur)
        f_c[:, gc] = gaussian_cur[:, 0]

    # generate each of the gaussian map
    for j in range(nb_gaussian):
        for m in range(shape_r_f):
            for n in range(shape_c_f):
                gp[j, m, n] = f_r[m, j] * f_c[n, j]

    gp.unsqueeze_(0)

    return gp


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        base_model = resnet.resnet50(pretrained=True)
        base_layers = list(base_model.children())[:8]
        self.encoder = nn.ModuleList(base_layers).eval()
        
        for p in self.parameters():
            p.requires_grad=False

    def forward(self, x):
        outputs = []
        for ii,layer in enumerate(self.encoder):
            x = layer(x)
            if ii in {5,6,7}:
                outputs.append(x)
        return outputs
# ---------------------------------------------------------------------------------------------------------------------
cfg1 = {
"hidden_size" : 768,
"mlp_dim" : 768*4,
"num_heads" : 12,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

cfg2 = {
"hidden_size" : 768,
"mlp_dim" : 768*4,
"num_heads" : 12,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

cfg3 = {
"hidden_size" : 512,
"mlp_dim" : 512*4,
"num_heads" : 8,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}


class CrossAttention(nn.Module):

    def __init__(self, c, w, h, embed_dim=512):
        super(CrossAttention, self).__init__()

        # define number of temporal steps
        text_width = 256
        embed_dim = c
        self.W_Q = nn.Linear(in_features=text_width, out_features=embed_dim * w * h, bias=False)
        self.W_K = nn.Linear(in_features=c, out_features=embed_dim, bias=False)
        self.W_V = nn.Linear(in_features=c, out_features=embed_dim, bias=False)
        # self.W_sum = nn.Linear(in_features=embed_dim, out_features=c)
        self.embed_dim = embed_dim
        self.sqrt_dim = np.sqrt(embed_dim)
        self.softmax = nn.Softmax(dim=1)
        self.w = w
        self.h = h
        # self.batchnorm = nn.BatchNorm2d(c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.relu = nn.ReLU(True)



    def forward(self, x, txt):
        # gain the current cell memory and hidden state
        txt = txt.view(txt.shape[0], txt.shape[2])
        x = x

        # cross attention with text information
        Q_ = self.W_Q(txt).view(-1, self.w, self.h, self.embed_dim)
        K_ = self.W_K(x.permute(0, 2, 3, 1))  # batch, w, h, c
        V_ = self.W_V(x.permute(0, 2, 3, 1))  # batch, w, h, c

        # A_ = self.W_sum(self.softmax((Q_ * K_).sum(dim=3, keepdim=True) / self.sqrt_dim).repeat(1, 1, 1, self.embed_dim) * V_).permute(0, 3, 1, 2)
        A_ = (self.softmax((Q_ * K_).sum(dim=3, keepdim=True) / self.sqrt_dim).repeat(1, 1, 1, self.embed_dim) * V_).permute(0, 3, 1, 2)
        # A_ = self.batchnorm(A_)
        # A_ = self.relu(A_)
        x = x + A_

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.conv4 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.conv5 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm1 = nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # define the learnable gaussian priors
        # self.gaussian_priors = MyPriors(gp=gaussian_prior)

        self.TransEncoder1 = TransEncoder(in_channels=2048, spatial_size=9*12, cfg=cfg1)
        self.TransEncoder2 = TransEncoder(in_channels=1024, spatial_size=18*24, cfg=cfg2)
        self.TransEncoder3 = TransEncoder(in_channels=512, spatial_size=36*48, cfg=cfg3)

        self.add = torch.add
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()

            
        self.conv4 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm6 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # for p in self.parameters():
        #     p.requires_grad=False

        # self.ca1 = CrossAttention(c=2048, w=9, h=12)
        # self.ca2 = CrossAttention(c=1024, w=18, h=24)
        # self.ca3 = CrossAttention(c=512, w=36, h=48)

    def forward(self, x):
        x3, x4, x5 = x
        # x3 = x

        # x5 = self.attentiveLSTM1(x5, txt)
        # x5 = self.ca1(x5, txt)
        x5 = self.TransEncoder1(x5)
        x5 = self.conv1(x5)
        x5 = self.batchnorm1(x5)
        x5 = self.relu(x5)
        x5 = self.upsample(x5)

        # print(x4.shape)
        # x4 = self.attentiveLSTM2(x4, txt)
        # print(x4.shape)
        # x4 = self.ca2(x4, txt)
        x4_a = self.TransEncoder2(x4)
        # print(x4_a.shape, x5.shape)
        x4 = x5 * x4_a
        x4 = self.relu(x4)
        x4 = self.conv2(x4)
        x4 = self.batchnorm2(x4)
        x4 = self.relu(x4)
        x4 = self.upsample(x4)

        # x3 = self.attentiveLSTM3(x3, txt)
        # x3 = self.ca3(x3, txt)
        x3_a = self.TransEncoder3(x3)
        x3 = x4 * x3_a
        x3 = self.relu(x3)
        x3 = self.conv3(x3)
        x3 = self.batchnorm3(x3)
        x3 = self.relu(x3)
        x3 = self.upsample(x3)

        x2 = self.conv4(x3)
        x2 = self.batchnorm4(x2)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.conv5(x2)
        x2 = self.batchnorm5(x2)
        x2 = self.relu(x2)

        x1 = self.upsample(x2)
        x1 = self.conv6(x1)
        x1 = self.batchnorm6(x1)
        x1 = self.relu(x1)
        x1 = self.conv7(x1)
        x = self.sigmoid(x1)

        return x
    

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class TransEncoder(nn.Module):

    def __init__(self, in_channels, spatial_size, cfg):
        super(TransEncoder, self).__init__()

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=cfg['hidden_size'],
                                          kernel_size=1,
                                          stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, spatial_size, cfg['hidden_size']))
        # self.W_Q = nn.Linear(in_features=256, out_features=cfg['hidden_size'] ** 2, bias=False)

        self.transformer_encoder = transEncoder(cfg)

    def forward(self, x):
        a, b = x.shape[2], x.shape[3]
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        # txt = self.W_Q(txt.view(txt.shape[0], txt.shape[2])).view(x.shape[0], x.shape[2], x.shape[2])
        # txt_T = torch.transpose(txt, 1, 2)

        # attn = torch.bmm(x, txt_T)
        # attn = attn.view(attn.shape[0]*attn.shape[1], attn.shape[2])
        # attn = nn.Softmax()(attn).view(x.shape[0], x.shape[1], x.shape[2])
        embeddings = x + self.position_embeddings
        x = self.transformer_encoder(embeddings)
        B, n_patch, hidden = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, a, b)

        return x

# ---------------------------------------------------------------------------------------------------------------------
# define the dice coefficient
class MyDiceCoef(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyDiceCoef, self).__init__()

    def forward(self, input, target):
        input = input.view(input.size(0), -1).detach()  # [8,1,240.320] -> [8, 76800]
        target = target.view(target.size(0), -1).detach()  # [8,1,240.320] -> [8, 76800]

        DC = []

        for i in range(b_s):
            DC.append((2 * torch.sum(input[i] * target[i]) + epsilon_DC) / \
                      (torch.sum(input[i]) + torch.sum(target[i]) + epsilon_DC))
            DC[i].unsqueeze_(0)

        DC = torch.cat(DC, 0)
        DC = torch.mean(DC)

        return DC


# define the correlation coefficient
class MyCorrCoef(nn.Module):

    def __init__(self):
        super(MyCorrCoef, self).__init__()

    def forward(self, input, target):
        input = input.view(input.size(0), -1).detach()
        target = target.view(target.size(0), -1).detach()
        CC = 0

        for i in range(b_s):
            input_ = input[i]
            target_ = target[i]
            combined = torch.stack((input_, target_))
            CC += torch.corrcoef(combined)[0, 1]

        CC /= input.size(0)

        return CC


class MyMSE(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyMSE, self).__init__()

    def forward(self, input, target):
        input = input.view(input.size(0), -1).detach()
        target = target.view(target.size(0), -1).detach()
        mse = F.mse_loss(input, target)

        return mse


class MyAUC(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MyAUC, self).__init__()

    def forward(self, input, target):
        input = input.view(input.size(0), -1).detach().cpu()
        target = target.view(target.size(0), -1).detach().cpu()
        thres_target = torch.sort(target, dim=1)[0][:, int(0.8 * target.size(1))].unsqueeze(1)
        thres_input = torch.sort(input, dim=1)[0][:, int(0.8 * input.size(1))].unsqueeze(1)
        targets = (target > thres_target)
        inputs = (input > thres_input).float()
        score = sum([roc_auc_score(targets[i].numpy(), inputs[i].numpy()) for i in range(targets.size(0))])
        score /= targets.size(0)
        return score


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


class MysAUC(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MysAUC, self).__init__()

    def forward(self, input, target):
        input = input.view(input.size(0), -1).detach().cpu()
        target = target.view(target.size(0), -1).detach().cpu()
        thres_target = torch.sort(target, dim=1)[0][:, int(0.8 * target.size(1))].unsqueeze(1)
        thres_input = torch.sort(input, dim=1)[0][:, int(0.8 * input.size(1))].unsqueeze(1)
        targets = (target > thres_target).numpy()
        inputs = (input > thres_input).float().numpy()
        score = 0
        for i in range(inputs.shape[0]):
            target_ = targets[i]
            input_ = inputs[i]
            positive_indices = np.where(target_ > 0)[0]
            negative_indices = np.where(target_ == 0)[0]
            np.random.shuffle(negative_indices)
            # negative_indices = shuffle_along_axis(negative_indices, axis=1)
            shuffled_indices = np.concatenate([positive_indices, negative_indices[:len(positive_indices)]])
            shuffled_target = np.zeros_like(target_)
            shuffled_target[shuffled_indices] = 1
            score += roc_auc_score(shuffled_target, input_)
        score /= inputs.shape[0]
        return score


class MyNormScanSali(nn.Module):

    def __init__(self):
        super(MyNormScanSali, self).__init__()

    def forward(self, input, target):

        input = input.view(input.size(0), -1).detach()
        target = target.view(target.size(0), -1).detach()

        NSS = 0
        for i in range(b_s):
            input_norm = (input[i] - torch.mean(input[i])) / torch.std(input[i])

            target_ = target[i]
            target_ = (target_ > 0)
            NSS += torch.sum(input_norm * target_) / torch.sum(target_)

        NSS /= input.size(0)

        return NSS

    
class mySim(nn.Module):
    def __init__(self):
        super(mySim, self).__init__()
    
    def forward(self, input, target):
        input = input.view(input.size(0), -1).detach()
        target = target.view(target.size(0), -1).detach()  

        sim = 0      
        for i in range(b_s):
            gt_map = target[i]
            pred_map = input[i]
            gt_map = (gt_map - torch.min(gt_map))/(torch.max(gt_map)-torch.min(gt_map))
            gt_map = gt_map/torch.sum(gt_map)
            
            pred_map = (pred_map - torch.min(pred_map))/(torch.max(pred_map)-torch.min(pred_map))
            pred_map = pred_map/torch.sum(pred_map)
            
            diff = torch.min(gt_map,pred_map)
            sim += torch.sum(diff)
        sim /= input.size(0)
        return sim


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = order_sim
        self.max_violation = max_violation
        self.blip = BLIP_Pretrain(image_size=(img_H, img_W), med_config="/home/huangyixin/homework2/config/med_config.json").cuda()


    def forward(self, outputs, image, s):
        # compute image-sentence score matrix
        im = outputs * image
        im = self.blip.visual_encoder(im).view(outputs.size(0), -1)
        s = s.view(s.size(0), -1)
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()