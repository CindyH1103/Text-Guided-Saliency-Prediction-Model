from __future__ import division
import cv2
import numpy as np
import scipy.io
import scipy.ndimage
from torch.utils.data import Dataset
from PIL import Image
import torch
from scipy.io import loadmat
from config import shape_r_out, shape_c_out, b_s
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from BLIP.blip_pretrain import BLIP_Pretrain
from config import *
import torch.nn.functional as F
import pandas as pd
import datetime
import sys


# complement the class of Mydataset
class Dataset_salicon(Dataset):
    def __init__(self, imgs_path, maps_path, fixs_path,
                 transform_img=None, transform_map=None, transform_fix=None):
        imgs = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.imgs = imgs
        self.maps = maps
        self.fixs = fixs
        self.transform_img = transform_img
        self.transform_map = transform_map
        self.transform_fix = transform_fix
        self.blip = BLIP_Pretrain(image_size=(img_H, img_W), med_config="/home/huangyixin/homework2/config/med_config.json").cuda()

    def __getitem__(self, index):
        img_path_cur = self.imgs[index]
        map_path_cur = self.maps[index]
        fix_path_cur = self.fixs[index]
        img = Image.open(img_path_cur).convert('RGB')
        map = Image.open(map_path_cur)
        fix = Image.open(fix_path_cur)

        if self.transform_img is not None:
            img = self.transform_img(img).to("cuda")
        if self.transform_map is not None:
            map = self.transform_map(map).to("cuda")
        if self.transform_fix is not None:
            fix = self.transform_fix(fix).to("cuda")

        text_input = self.blip.tokenizer("", padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to("cuda")
        text_output = self.blip.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                             mode='text')
        txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:, 0, :]))
        # img_embeds = self.blip.visual_encoder(img.unsqueeze(0))
        # # print(img_embeds.shape)
        # image_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to("cuda")
        # text_output = self.blip.text_encoder(text_input.input_ids,
        #                                         attention_mask = text_input.attention_mask,
        #                                         encoder_hidden_states = img_embeds,
        #                                         encoder_attention_mask = image_atts,
        #                                         return_dict = True,
        #                                     )
        # txt_feature = text_output.last_hidden_state[:,0,:].float()
        # print("txt: ", txt_feature.shape)

        return img, map, fix, txt_feature

    def __len__(self):
        return len(self.imgs)


# add temporal dimention
def format_attLSTM(x, nb_ts):
    y = []
    x.unsqueeze_(1)
    for i in range(nb_ts):
        y.append(x)
    feature = torch.cat(y, 1)

    return feature


# preprocessing the fixs dataset
def fixs_preprocessing(fixs):
    fixs_out = []
    for i in range(b_s):
        img = transforms.ToPILImage()(fixs[i])
        plt.imshow(img)
        plt.show()

    print('Done!')


def redirect_log_file(dataset, img_type, log_root="./log"):
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    t = str(datetime.datetime.now())
    filename = t[2:][:-7] + " || " + dataset
    if dataset != "salicon":
        filename += "_" + img_type
    filename += ".txt"
    out_file = os.path.join(log_root, filename)
    print("Redirect log to: ", out_file, flush=True)
    sys.stdout = open(out_file, 'a')
    sys.stderr = open(out_file, 'a')
    print("Start time:", t, flush=True)
    print("hyperparameter: ", flush=True)
    print("scal_KLD:", scal_KLD, flush=True)
    print('scal_CC:', scal_CC, flush=True)
    print('scal_NSS: ', scal_NSS, flush=True)
    print('lr_init: ', lr_init, flush=True)
    print('loss_bias: ', loss_bias, flush=True)


class PureDataSet(Dataset):
    def __init__(self, img_table, img_file_prefix=None, map_prefix=None, fix_prefix=None, seed=None, transform_img=None, transform_map=None, transform_fix=None):
        self.img_table = img_table
        self.prefix = img_file_prefix
        self.map_prefix = map_prefix
        self.fix_prefix = fix_prefix
        self.seed = seed
        self.transform_img = transform_img
        self.transform_map = transform_map
        self.transform_fix = transform_fix
        self.blip = BLIP_Pretrain(image_size=(img_H, img_W), med_config="/home/huangyixin/homework2/config/med_config.json").cuda()

    def __getitem__(self, idx):
        row = self.img_table.iloc[idx]
        name = str(row["image_number"]) + "_0" + ".png"
        img_path = os.path.join(self.prefix, name)
        map_path = os.path.join(self.map_prefix, name)
        fix_path = os.path.join(self.fix_prefix, name)
        img = Image.open(img_path).convert('RGB')
        map = Image.open(map_path)
        fix = Image.open(fix_path).convert("L")
        # text_input = self.blip.tokenizer("", padding='max_length', truncation=True, max_length=35,
        #                                  return_tensors="pt").to("cuda")
        # text_output = self.blip.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
        #                                      mode='text')
        if self.transform_img is not None:
            img = self.transform_img(img).to("cuda")
        if self.transform_map is not None:
            map = self.transform_map(map).to("cuda")
        if self.transform_fix is not None:
            fix = self.transform_fix(fix).to("cuda")
        # txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:, 0, :]))
        text_input = self.blip.tokenizer("", padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to("cuda")
        text_output = self.blip.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                             mode='text')
        txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:, 0, :]))
        # img_embeds = self.blip.visual_encoder(img.unsqueeze(0))
        # # print(img_embeds.shape)
        # image_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to("cuda")
        # text_output = self.blip.text_encoder(text_input.input_ids,
        #                                         attention_mask = text_input.attention_mask,
        #                                         encoder_hidden_states = img_embeds,
        #                                         encoder_attention_mask = image_atts,
        #                                         return_dict = True,
        #                                     )
        # txt_feature = text_output.last_hidden_state[:,0,:].float()

        return img, map, fix, txt_feature
        # return prompt, img_path, align_score, quality_score

    def __len__(self):
        return len(self.img_table)


class Img_Text_DataSet(Dataset):
    def __init__(self, img_table, img_file_prefix=None, map_prefix=None, fix_prefix=None, data_type="all", transform_img=None, transform_map=None, transform_fix=None):
        self.img_table = img_table
        self.prefix = img_file_prefix
        self.map_prefix = map_prefix
        self.fix_prefix = fix_prefix
        self.data_type = data_type
        self.type = {
            "all": "_1",
            "non_salient": "_2",
            "salient": "_3"
        }
        self.transform_img = transform_img
        self.transform_map = transform_map
        self.transform_fix = transform_fix
        self.blip = BLIP_Pretrain(image_size=(img_H, img_W), med_config="/home/huangyixin/homework2/config/med_config.json").cuda()

    def __getitem__(self, idx):
        row = self.img_table.iloc[idx]
        name = (row["image_number"]) + self.type[self.data_type] + ".png"
        text = row["text"]
        img_path = os.path.join(self.prefix, name)
        map_path = os.path.join(self.map_prefix, name)
        fix_path = os.path.join(self.fix_prefix, name)
        img = Image.open(img_path).convert('RGB')
        map = Image.open(map_path)
        fix = Image.open(fix_path).convert("L")
        text_input = self.blip.tokenizer(text, padding='max_length', truncation=True, max_length=35,
                                         return_tensors="pt").to("cuda")
        text_output = self.blip.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                             mode='text')
        if self.transform_img is not None:
            img = self.transform_img(img).to("cuda")
        if self.transform_map is not None:
            map = self.transform_map(map).to("cuda")
        if self.transform_fix is not None:
            fix = self.transform_fix(fix).to("cuda")
        txt_feature = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:, 0, :]))
        # text_input = self.blip.tokenizer(text, padding='max_length', truncation=True, max_length=35,
        #                                  return_tensors="pt").to("cuda")
        # img_embeds = self.blip.visual_encoder(img.unsqueeze(0))
        # image_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to("cuda")
        # text_output = self.blip.text_encoder(text_input.input_ids,
        #                                         attention_mask = text_input.attention_mask,
        #                                         encoder_hidden_states = img_embeds,
        #                                         encoder_attention_mask = image_atts,
        #                                         return_dict = True,
        #                                     )
        # txt_feature = text_output.last_hidden_state[:,0,:].float()

        return img, map, fix, txt_feature
        # return prompt, img_path, align_score, quality_score

    def __len__(self):
        return len(self.img_table)


img_prefix = "/home/huangyixin/homework2/saliency/image"
map_prefix = "/home/huangyixin/homework2/saliency/map"
fix_prefix = "/home/huangyixin/homework2/saliency/fixation"

pure_df_path = "/home/huangyixin/homework2/saliency/pure.csv"
all_df_path = "/home/huangyixin/homework2/saliency/all.csv"
non_salient_df_path = "/home/huangyixin/homework2/saliency/non_salient.csv"
salient_df_path = "/home/huangyixin/homework2/saliency/salient.csv"

pure_df = pd.read_csv(pure_df_path, dtype=object)
# whole_df = pd.read_csv(whole_df_path, dtype=object)

all_df = pd.read_csv(all_df_path, dtype=object)
non_salient_df = pd.read_csv(non_salient_df_path, dtype=object)
salient_df = pd.read_csv(salient_df_path, dtype=object)

# split each df into train_df and test_df, with a ratio = 0.8
pure_train_df = pure_df.sample(frac=0.8)
pure_test_df = pure_df.drop(pure_train_df.index)

# whole_train_df = whole_df.sample(frac=0.8)
# whole_test_df = whole_df.drop(whole_train_df.index)

all_train_df = all_df.sample(frac=0.8)
all_test_df = all_df.drop(all_train_df.index)

non_salient_train_df = non_salient_df.sample(frac=0.8)
non_salient_test_df = non_salient_df.drop(non_salient_train_df.index)

salient_train_df = salient_df.sample(frac=0.8)
salient_test_df = salient_df.drop(salient_train_df.index)


class SaliencyDataSet:
    def __init__(self, transform_img=None, transform_map=None, transform_fix=None):
        self.transform_img = transform_img
        self.transform_map = transform_map
        self.transform_fix = transform_fix
        

    def get_loader(self, type="pure"):
        if type == "pure":
            pure_type_train_dataset = PureDataSet(pure_train_df, img_prefix, map_prefix, fix_prefix, transform_img=self.transform_img, transform_map=self.transform_map, transform_fix=self.transform_fix)
            pure_type_test_dataset = PureDataSet(pure_test_df, img_prefix, map_prefix, fix_prefix, transform_img=self.transform_img, transform_map=self.transform_map, transform_fix=self.transform_fix)
            return torch.utils.data.DataLoader(pure_type_train_dataset, batch_size=b_s,
                                               shuffle=True), torch.utils.data.DataLoader(pure_type_test_dataset,
                                                                                          batch_size=b_s,
                                                                                          shuffle=True)
        elif type == "all":
            all_type_train_dataset = Img_Text_DataSet(all_train_df, img_prefix, map_prefix, fix_prefix, data_type="all", transform_img=self.transform_img, transform_map=self.transform_map, transform_fix=self.transform_fix)
            all_type_test_dataset = Img_Text_DataSet(all_test_df, img_prefix, map_prefix, fix_prefix, data_type="all", transform_img=self.transform_img, transform_map=self.transform_map, transform_fix=self.transform_fix)
            return torch.utils.data.DataLoader(all_type_train_dataset, batch_size=b_s,
                                               shuffle=True), torch.utils.data.DataLoader(all_type_test_dataset,
                                                                                          batch_size=b_s,
                                                                                          shuffle=True)
        elif type == "non_salient":
            non_salient_type_train_dataset = Img_Text_DataSet(non_salient_train_df, img_prefix, map_prefix, fix_prefix, data_type="non_salient", transform_img=self.transform_img, transform_map=self.transform_map, transform_fix=self.transform_fix)
            non_salient_type_test_dataset = Img_Text_DataSet(non_salient_test_df, img_prefix, map_prefix, fix_prefix, data_type="non_salient", transform_img=self.transform_img, transform_map=self.transform_map, transform_fix=self.transform_fix)
            return torch.utils.data.DataLoader(non_salient_type_train_dataset, batch_size=b_s,
                                               shuffle=True), torch.utils.data.DataLoader(
                non_salient_type_test_dataset, batch_size=b_s, shuffle=True)
        elif type == "salient":
            salient_type_train_dataset = Img_Text_DataSet(salient_train_df, img_prefix, map_prefix, fix_prefix, data_type="salient", transform_img=self.transform_img, transform_map=self.transform_map, transform_fix=self.transform_fix)
            salient_type_test_dataset = Img_Text_DataSet(salient_test_df, img_prefix, map_prefix, fix_prefix, data_type="salient", transform_img=self.transform_img, transform_map=self.transform_map, transform_fix=self.transform_fix)
            return torch.utils.data.DataLoader(salient_type_train_dataset, batch_size=b_s,
                                               shuffle=True), torch.utils.data.DataLoader(
                salient_type_test_dataset, batch_size=b_s, shuffle=True)