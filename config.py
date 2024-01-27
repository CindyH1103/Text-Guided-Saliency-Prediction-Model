#########################################################################
# MODEL PARAMETERS														#
#########################################################################

import os

# batch size
b_s = 4  # be exactly divided by the total number of training as well validation image
# number of rows of model outputs
shape_r_out = 288
# number of cols of model outputs
shape_c_out = 384
# number of rows of learned features
shape_r_f = 36
# number of cols of learned features
shape_c_f = 48
# number of epochs
nb_epoch = 40
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16
# the height of the inputting images
img_H = 288
# the width of the inputting images
img_W = 384
# path of pretrained model_1
pth_pm_1 = 'resnet50-19c8e357.pth'
# path of saved parameters
log_dir = os.path.join('Results')
log_dir_result = os.path.join('model')

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training images
imgs_train_path = "/home/huangyixin/homework2/salicon/images/train/"
# path of training maps
maps_train_path = "/home/huangyixin/homework2/salicon/maps/train/"
# path of training fixation maps
fixs_train_path = "/home/huangyixin/homework2/salicon/fixations_map/train/"
# number of training images
# nb_imgs_train = 0
# path of validation images
imgs_val_path = "/home/huangyixin/homework2/salicon/images/val/"
# path of validation maps
maps_val_path = "/home/huangyixin/homework2/salicon/maps/val/"
# path of validation fixation maps
fixs_val_path = "/home/huangyixin/homework2/salicon/fixations_map/val/"
split_data = 0
# how many images are needed to compute the mean and std
CNum = 160
# computed mean of the training images
NormMean_imgs = [0.50893384, 0.4930997, 0.46955067]
# computed std of the training images
NormStd_imgs = [0.2652982, 0.26586023, 0.27988392]
# need to compute the mean and std ?
compute_ms = 0
# the coefficient of KL-DIV
scal_KLD = 10
# the coefficient of CC
scal_CC = -2
# set default of the epsilon for DC
epsilon_DC = 0.001
# the coefficient of NSS
scal_NSS = -1
# the coefficient of MSE loss
scal_MSE = 3
scal_Sim = -1
scal_contra = 1
# initialize the learning rate
lr_init = 1e-6
# learning rate updating
step_size = 5
# loss bias used in the loss
loss_bias = 0
# feature_channel = 512
# upsampling factor = 240/30 = 8
