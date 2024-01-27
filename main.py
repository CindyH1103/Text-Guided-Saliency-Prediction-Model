# ------------------------------------- step 0 : Input needed packages ---------------------------------------
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.models import resnet50
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
from utilities import Dataset_salicon, SaliencyDataSet, redirect_log_file
from config import *
import glob
import random
import shutil
import cv2
from model import *
import matplotlib.pyplot as plt
import warnings
import argparse
import time

warnings.filterwarnings("ignore")

# ------------------------------------ step 1 : Data Pre-processing ----------------------------------------
# data pre-processing
def data_preprocessing(dataset="salicon"):
    normTransform = transforms.Normalize(NormMean_imgs, NormStd_imgs)
    imgsTransform = transforms.Compose([
        transforms.Resize([img_H, img_W]),
        transforms.
            transforms.ToTensor(),  # 0-255 automatically transformed to 0-1
        normTransform
    ])
    mapsTransform = transforms.Compose([
        transforms.Resize([shape_r_out, shape_c_out]),
        transforms.ToTensor()
    ])
    fixsTransform = transforms.Compose([
        transforms.Resize([shape_r_out, shape_c_out]),
        transforms.ToTensor()
    ])

    if dataset == "salicon":
        train_data = Dataset_salicon(imgs_train_path, maps_train_path, fixs_train_path,
                                     transform_img=imgsTransform, transform_map=mapsTransform,
                                     transform_fix=fixsTransform)
        val_data = Dataset_salicon(imgs_val_path, maps_val_path, fixs_val_path,
                                   transform_img=imgsTransform, transform_map=mapsTransform,
                                   transform_fix=fixsTransform)
        return train_data, val_data
    else:
        # then the dataset should be the saliency dataset
        dataset_set = SaliencyDataSet(transform_img=imgsTransform, transform_map=mapsTransform,
                                      transform_fix=fixsTransform)
        return dataset_set


# ------------------------------------ step 2 : Net Defining ------------------------------------------------
class SAMResnet(nn.Module):
    def __init__(self):
        super(SAMResnet, self).__init__()
        # define the dilated ResNet50 (with output_channel=512)
        # self.dcn = MyDRN()
        self.dcn = Encoder()

        # define the attentive convLSTM
        # self.attentiveLSTM = MyAttentiveLSTM(nb_features_in=512, nb_features_out=512,
        #                                      nb_features_att=512, nb_rows=3, nb_cols=3)

        # # define the learnable gaussian priors
        # self.gaussian_priors = MyPriors(gp=gaussian_prior)

        # define the final convolutional neural network
        # self.endconv = nn.Conv2d(in_channels=512, out_channels=1,
        #                          kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # self.upsampling = nn.UpsamplingBilinear2d([shape_r_out, shape_c_out])
        # self.relu = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.decoder = Decoder()
        self.fctxt = nn.Linear(256, 768 * 433)

    def forward(self, x, txt):

        # the dilated convlutional network based on ResNet 50
        x = self.dcn(x)
        # print(x.shape)

        # the convLSTM model
        # x = self.attentiveLSTM(x, txt)

        # the learnable prior block
        # x = self.gaussian_priors(x)

        # the non local neural block

        # the final convolutional neural network
        # x = self.endconv(x)
        # x = self.relu(x)
        # x = self.upsampling(x)
        # x = self.sigmoid(x)
        # print(x.shape)
        x = self.decoder(x)
        txt = self.fctxt(txt)

        return x, txt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, required=False, default="0")
    parser.add_argument('--dataset', type=str, required=False, default="saliency")
    parser.add_argument('--img_type', type=str, required=False, default="non_salient")
    parser.add_argument('--finetune', type=bool, required=False, default=True)
    parser.add_argument('--seed', type=int, required=False, default=123)
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ------------------------------------ step 3 : Optimizer and LOSS ----------------------------------------------------
if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu_id
    dataset = args.dataset
    img_type = args.img_type
    finetune = args.finetune
    seed = args.seed
    set_seed(seed)

    # generate the gaussian priors
    # gp = generate_gaussian_prior().to("cuda")

    # call the proposed model
    net = SAMResnet().cuda()
    # net.initialize_weights()
    if finetune:
        net.load_state_dict(torch.load('./model/net_params_SALICON.pkl'), strict=False)

    # build a Kullback-Leibler divergence
    criterion_KLD = nn.KLDivLoss().cuda()

    # build a dice coefficient
    criterion_DC = MyDiceCoef().cuda()

    # build a correlation coefficient
    criterion_CC = MyCorrCoef().cuda()

    # build a normalized scanpath saliency
    criterion_NSS = MyNormScanSali().cuda()

    criterion_AUC = MyAUC().cuda()

    criterion_sAUC = MysAUC().cuda()

    criterion_MSE = MyMSE().cuda()
    criterion_Sim = mySim().cuda()
    loss_contra = ContrastiveLoss().cuda()

    # define the optimizer
    # optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)
    optimizer = optim.Adam(net.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-8)
    # define the scheduler for learning rate decreasing
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # ------------------------------------ step 4 : model training ----------------------------------------------
    run_name = dataset
    if dataset == "saliency":
        run_name += '_' + img_type
    run_name += '_' + str(seed) + '_' + str(time.time())
    log_dir = os.path.join(log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir, comment='reproducing_01')
    redirect_log_file(dataset, img_type)
    # load the training and validation dataset
    if dataset == "salicon":
        train_data, val_data = data_preprocessing(dataset)
        train_loader = DataLoader(dataset=train_data, batch_size=b_s, shuffle=True)
        valid_loader = DataLoader(dataset=val_data, batch_size=b_s, shuffle=True)
    else:
        dataset_set = data_preprocessing(dataset)
        train_loader, valid_loader = dataset_set.get_loader(img_type)

    # the main loop
    for epoch in range(nb_epoch):
        loss_sigma = 0.0
        loss_val = 0.0
        correct_auc = 0.0
        correct_sauc = 0.0
        correct_cc = 0.0
        correct_nss = 0.0
        total = 0.0
        correct_auc_val = 0.0
        correct_sauc_val = 0.0
        correct_cc_val = 0.0
        correct_nss_val = 0.0
        total_val = 0.0
        scheduler.step()

        for i, data in enumerate(train_loader):

            inputs, maps, fixs, txt_feature = data
            inputs, maps, fixs, txt_feature = Variable(inputs), Variable(maps), Variable(fixs), Variable(txt_feature)

            # if epoch == 0 and i == 0:
            #         writer.add_graph(net, (inputs, txt))

            # forward
            optimizer.zero_grad()
            outputs, txt_feature = net(inputs, txt_feature)

            # compute loss ("2" is an experiential default)
            loss = scal_KLD * criterion_KLD(outputs, maps) + scal_CC * criterion_CC(outputs, maps) \
                   + scal_NSS * criterion_NSS(outputs, fixs) + scal_MSE * criterion_MSE(outputs, maps) + scal_Sim * criterion_Sim(outputs, maps) + scal_contra * loss_contra(outputs, inputs, txt_feature) + loss_bias

            # compute metric
            metric_cc = criterion_CC(outputs, maps)
            metric_nss = criterion_NSS(outputs, fixs)
            metric_auc = criterion_AUC(outputs, maps)
            metric_sauc = criterion_sAUC(outputs, maps)

            # backward
            loss.backward()
            optimizer.step()

            # statistics for predicted information with varied LOSS
            correct_auc += metric_auc.item()
            correct_sauc += metric_sauc.item()
            correct_cc += metric_cc.item()
            correct_nss += metric_nss.item()
            total += 1
            loss_sigma += loss.item()

            if i % 10 == 0:
                loss_avg = loss_sigma / 10
                loss_sigma = 0.0
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} AUC:{:.4f} sAUC: {:.4f} CC: {:.4f} NSS: {:.4f}".format(
                        epoch + 1, nb_epoch, i + 1, len(train_loader), loss_avg, correct_auc / total,
                        correct_sauc / total, correct_cc / total, correct_nss / total), flush=True)
                # record training loss
                writer.add_scalar('Loss_group/train_loss', loss_avg, epoch)
                # record accuracy
                writer.add_scalar('Metrics/train_auc', correct_auc / total, epoch)
                writer.add_scalar('Metrics/train_sauc', correct_sauc / total, epoch)
                writer.add_scalar('Metrics/train_cc', correct_cc / total, epoch)
                writer.add_scalar('Metrics/train_nss', correct_nss / total, epoch)

            # model visualization
            if i % 10 == 0:

                # visualize the inputs, maps and outputs
                show_inputs = make_grid(inputs)
                show_maps = make_grid(maps)
                show_outputs = make_grid(outputs)
                writer.add_image('Input_group', show_inputs)
                writer.add_image('Map_group', show_maps)
                writer.add_image('Out_group', show_outputs)
                # scheduler.step()

                # visualize the important feature maps (512*30*40) between the drn and convLSTM
                # x_show = inputs
                # for net_part_name, net_part_layer in net._modules.items():
                #     if net_part_name == 'dcn':
                #         x_show = net_part_layer(x_show)
                #     else:
                #         break
                # x_show_fb = x_show[0]
                # x_show_fb.unsqueeze_(1)
                # show_feature = make_grid(x_show_fb)  # just show the features of the first batch
                # writer.add_image('Feature_group', show_feature)

        # for name, layer in net.named_parameters():
        #     writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        #     writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        if epoch % 1 == 0:
            net.eval()
            for i, data in enumerate(valid_loader):

                if i > 0:
                    break

                images, maps, fixs, txt = data
                images, maps, fixs, txt = Variable(images), Variable(maps), Variable(fixs), Variable(txt)

                # forward
                outputs, txt = net(images, txt)
                # outputs.detach()

                # compute loss ("2" is an experiential default)
                loss = scal_KLD * criterion_KLD(outputs, maps) + scal_CC * criterion_CC(outputs, maps) \
                       + scal_NSS * criterion_NSS(outputs, fixs) + scal_MSE * criterion_MSE(outputs, maps) + scal_Sim * criterion_Sim(outputs, maps) + scal_contra * loss_contra(outputs, inputs, txt_feature) + loss_bias

                metric_cc = criterion_CC(outputs, maps)
                metric_nss = criterion_NSS(outputs, fixs)
                metric_auc = criterion_AUC(outputs, maps)
                metric_sauc = criterion_sAUC(outputs, maps)

                # compute metric
                correct_auc_val += metric_auc.item()
                correct_sauc_val += metric_sauc.item()
                correct_cc_val += metric_cc.item()
                correct_nss_val += metric_nss.item()

                loss_val += loss.item()
                total_val += 1

                loss_avg_val = loss_val
                loss_val = 0.0

                print(
                    "Validation: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} AUC:{:.4f} sAUC: {:.4f} CC: {:.4f} NSS: {:.4f}".format(
                        epoch + 1, nb_epoch, i + 1, len(valid_loader), loss_avg_val, correct_auc / total,
                        correct_sauc / total, correct_cc / total, correct_nss / total), flush=True)
                # record validation loss
                writer.add_scalar('Loss_group/valid_loss', loss_avg_val, epoch)
                # record validation accuracy
                writer.add_scalar('Metrics/valid_auc', correct_auc_val / total, epoch)
                writer.add_scalar('Metrics/valid_sauc', correct_sauc_val / total, epoch)
                writer.add_scalar('Metrics/valid_cc', correct_cc_val / total, epoch)
                writer.add_scalar('Metrics/valid_nss', correct_nss_val / total, epoch)

        if epoch % 10 == 0:
            file_name = 'net_params_' + dataset
            if dataset != "salicon":
                file_name += "_" + img_type
                if finetune:
                    file_name += "_" + "finetuned1"
            file_name += "_" + str(epoch) + ".pkl"
            net_save_path = os.path.join(log_dir_result, file_name)
            torch.save(net.state_dict(), net_save_path)


    print('finished training !', flush=True)
    writer.close()

    # ------------------------------------ step 5 : model saving ------------------------------------------------
    file_name = 'net_params_' + dataset
    if dataset != "salicon":
        file_name += "_" + img_type
        if finetune:
            file_name += "_" + "finetuned"
    file_name += ".pkl"
    net_save_path = os.path.join(log_dir_result, file_name)
    torch.save(net.state_dict(), net_save_path)

    # the end
    print('job done !', flush=True)
