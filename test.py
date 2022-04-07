import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader

from networks import getresnet18
from lmdbdataset import lmdbDataset
from utils import AverageMeter, accuracy, getbasenamewoext
import os
import shortuuid
from datetime import datetime




#
# def load_ckpt(model):
#   print("Load ckpt from {}".format(args.ckptpath))
#   checkpoint = torch.load(args.ckptpath)
#   model.load_state_dict(checkpoint['model_state_dict'])
#   # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#   epoch = checkpoint['epoch']
#   print ("Loaded epoch {}".format(epoch))

def testmodel(epoch, model, testdbpath, strckptpath):
  """
  """
  print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  the_file = open(strscorepath, "w")
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])# 0 to 1
  testdataset = lmdbDataset(testdbpath, transforms)

  print(testdataset)
  testloader = DataLoader(testdataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()
  probsm = nn.Softmax(dim=1)

  for index, (images, labels, imgpath) in enumerate(testloader):
    images, labels = images.cuda(), labels.cuda()
    logit = model(images)
    prob = probsm(logit)
    acc = accuracy(logit, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      the_file.write("{} {} {}\n".format(prob[idx][0], prob[idx][1], imgpathitem))
  the_file.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='anti-spoofing testing')
  # parser.add_argument('--ckptpath', type=str,
  #                     default='/home/user/model_2022', help='ckpt path')
  # parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
  # parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
  # parser.add_argument('--works', type=int, default=4, help='works')
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU)  # Set the GPU to use

  print(args)

  # trainmodel()
  testmodel(1, None, "/home/user/work_db/v220401_01/test_LDRGB_LD3007_1by1_260x260.db", "/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260_220407_G7gw5DGP2Z9oppqtU3DP9a")