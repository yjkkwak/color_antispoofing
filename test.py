import glob

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
from utils import AverageMeter, accuracy, getbasenamewoext, genfarfrreer, gentprwonlylive
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
  # print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  the_file = open(strscorepath, "w")
  if "260x260" in testdbpath:
    transforms = T.Compose([T.CenterCrop((256, 256)),
                            T.ToTensor()])  # 0 to 1
  elif "244x324" in testdbpath:
    transforms = T.Compose([T.CenterCrop((320, 240)),
                            T.ToTensor()])  # 0 to 1
  testdataset = lmdbDataset(testdbpath, transforms)

  # print(testdataset)
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
      the_file.write("{:.5f} {:.5f} {}\n".format(float(prob[idx][0]), float(prob[idx][1]), imgpathitem))
  the_file.close()

  # gen performace
  if "Emotion" in testdbpath:
    gentprwonlylive(strscorepath)
  else:
    genfarfrreer(strscorepath)

def testwckpt(model, strckptfilepath, testdbpath, strckptpath):
  """
  """
  # print ("test db {} based on {}".format(testdbpath, strckptpath))
  averagemetermap = {}
  averagemetermap["acc_am"] = AverageMeter()

  if model is None:
    model = getresnet18()
    model = model.cuda()

  checkpoint = torch.load(strckptfilepath)
  model.load_state_dict(checkpoint['model_state_dict'])
  epoch = checkpoint['epoch']

  strscorebasepath = os.path.join(strckptpath, getbasenamewoext(os.path.basename(testdbpath)))
  if os.path.exists(strscorebasepath) == False:
    os.makedirs(strscorebasepath)
  strscorepath = "{}/{:02d}.score".format(strscorebasepath, epoch)
  if os.path.exists(strscorepath):
    print ("exists {}".format(strscorepath))
    return
  the_file = open(strscorepath, "w")
  if "260x260" in testdbpath:
    transforms = T.Compose([T.CenterCrop((256, 256)),
                            T.ToTensor()])# 0 to 1
  elif "244x324" in testdbpath:
    transforms = T.Compose([T.CenterCrop((320, 240)),
                            T.ToTensor()])# 0 to 1
  testdataset = lmdbDataset(testdbpath, transforms)

  # print(testdataset)
  testloader = DataLoader(testdataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

  model.eval()
  probsm = nn.Softmax(dim=1)

  for index, (images, labels, imgpath) in enumerate(testloader):
    images, labels = images.cuda(), labels.cuda()
    logit = model(images)
    prob = probsm(logit)
    acc = accuracy(logit, labels)
    averagemetermap["acc_am"].update(acc[0].item())
    for idx, imgpathitem in enumerate(imgpath):
      the_file.write("{:.5f} {:.5f} {}\n".format(float(prob[idx][0]), float(prob[idx][1]), imgpathitem))
  the_file.close()

  # gen performace
  if "Emotion" in testdbpath:
    gentprwonlylive(strscorepath)
  else:
    genfarfrreer(strscorepath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='anti-spoofing testing')
  # parser.add_argument('--ckptpath', type=str,
  #                     default='/home/user/model_2022', help='ckpt path')
  # parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
  parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
  # parser.add_argument('--works', type=int, default=4, help='works')
  args = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU)  # Set the GPU to use

  print(args)
  #testmodel(1, None, "/home/user/work_db/v220401_01/test_LDRGB_LD3007_1by1_260x260.db", "/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260_220407_G7gw5DGP2Z9oppqtU3DP9a")


  # testdblist = ["/home/user/work_db/v220401_01/Test_v220401_01_SiW_1by1_260x260.db",
  #               "/home/user/work_db/v220401_01/Test_v220401_01_CelebA_1by1_260x260.db",
  #               "/home/user/work_db/v220401_01/Test_v220401_01_LD3007_1by1_260x260.db",
  #               "/home/user/work_db/v220401_01/Test_v220401_01_LDRGB_1by1_260x260.db"]

  testdblist = ["/home/user/work_db/v220401_01/Test_v220401_01_Emotion_1by1_260x260.db"]

  # basepathlist = ["/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_4by3_244x324_220415_4RnkqoXCLC7kjswpQ7jU5j_lr0.01_gamma_0.92_epochs_80_meta_163264/",
  #             "/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_4by3_244x324_220415_9JK2EGmzAk4hgnseEPZ8Ck_lr0.01_gamma_0.92_epochs_80_meta_163264/",
  #             "/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_4by3_244x324_220415_3FyDQrrgwjuRnD29YTPL4A_lr0.01_gamma_0.92_epochs_80_meta_163264/",
  #             "/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_4by3_244x324_220415_deXTJvMBCriiGtKuGkuA7Q_lr0.01_gamma_0.92_epochs_80_meta_163264/",]

  # basepathlist = ["/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_4by3_244x324_220412_eMedm6ND2sMEnU9wexbjCq_lr0.01_gamma_0.92/",
  #             "/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_4by3_244x324_220412_krSNtfNzMMAqdMSeM4kmn4_lr0.01_gamma_0.92/",
  #             "/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_4by3_244x324_220412_YEYEhPmvRPogm2fnjvJvHR_lr0.01_gamma_0.92/",
  #             "/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_4by3_244x324_220411_ARtFWpc23jBnAaSMLSc55h_lr0.01_gamma_0.92/",]

  # basepathlist = ["/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260_220408_nB8VRKkTsBdxgUVeKQyqcD_lr0.01_gamma_0.92/",
  #             "/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_1by1_260x260_220408_eSCEpkTvFk8aYoTn6RvgKt_lr0.01_gamma_0.92/",
  #             "/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_1by1_260x260_220408_6rAWNVCK3S72YS9svYtUrC_lr0.01_gamma_0.92/",
  #             "/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_1by1_260x260_220408_6ihoYoTHFkRdiLQcin3CNL_lr0.01_gamma_0.92/",]

  basepathlist = [
    "/home/user/model_2022/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260_220414_Rc3qebxDSvyocY4uCgcEKb_lr0.01_gamma_0.92_epochs_80_meta_163264/",
    "/home/user/model_2022/Train_v220401_01_CelebA_SiW_LD3007_1by1_260x260_220414_gXyb3RBNDU7pPVQjwt7KUw_lr0.01_gamma_0.92_epochs_80_meta_163264/",
    "/home/user/model_2022/Train_v220401_01_CelebA_SiW_LDRGB_1by1_260x260_220414_dnNzDP4dF5PX9gfrfGg3dq_lr0.01_gamma_0.92_epochs_80_meta_163264/",
    "/home/user/model_2022/Train_v220401_01_SiW_LDRGB_LD3007_1by1_260x260_220414_HDZCuMsB2eriabbcwYkRC5_lr0.01_gamma_0.92_epochs_80_meta_163264/", ]

  for basepath in basepathlist:
    ckptlist = glob.glob("{}/**/*.ckpt".format(basepath), recursive=True)
    for ckptpath in ckptlist:
      for testdbpath in testdblist:
        ffff = getbasenamewoext(ckptpath)
        if int(ffff[-2:]) > 60:
          print (ffff)
          testwckpt(None,
                    ckptpath,
                    testdbpath,
                    basepath)
