import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader

from networks import getbaseresnet18
from lmdbdataset import lmdbDataset
from utils import AverageMeter, accuracy, Timer, getbasenamewoext, Logger
import os
import shortuuid
from datetime import datetime
from test import testmodel
from shutil import copyfile

random_seed = 20220406
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

parser = argparse.ArgumentParser(description='anti-spoofing training')
parser.add_argument('--lmdbpath', type=str,
                    default='/home/user/work_db/v220419_01/Train_v220419_01_CelebA_LDRGB_LD3007_1by1_260x260.db', help='db path')
parser.add_argument('--ckptpath', type=str,
                    default='/home/user/model_2022/v220419_01', help='ckpt path')
parser.add_argument('--epochs', type=int, default=80, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
parser.add_argument('--works', type=int, default=4, help='works')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.97, help='gamma for scheduler')
parser.add_argument('--meta', type=str, default='meta', help='meta')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU)  # Set the GPU 2 to use
struuid = "{}_{}_{}_lr{}_gamma_{}_epochs_{}_meta_{}".format(getbasenamewoext(os.path.basename(args.lmdbpath)),
                                                            datetime.now().strftime("%y%m%d"),
                                                            shortuuid.uuid(),
                                                            args.lr,
                                                            args.gamma,
                                                            args.epochs,
                                                            args.meta)
strckptpath = os.path.join(args.ckptpath, struuid)
strlogpath = "/home/user/work_2022/logworkspace/baseline_resnet18pretrain_{}.log".format(struuid)
logger = Logger(strlogpath)
logger.print(args)


dbprefix = "/home/user/work_db/v220419_01"
if "260x260" in args.lmdbpath:
  testdblist = [os.path.join(dbprefix, "Test_v220419_01_CelebA_1by1_260x260.db"),
                os.path.join(dbprefix, "Test_v220419_01_LD3007_1by1_260x260.db"),
                os.path.join(dbprefix, "Test_v220419_01_LDRGB_1by1_260x260.db"),
                os.path.join(dbprefix, "Test_v220419_01_SiW_1by1_260x260.db"),
                os.path.join(dbprefix, "Test_v220419_01_Emotion_1by1_260x260.db")]
elif "244x324" in args.lmdbpath:
  testdblist = [os.path.join(dbprefix, "Test_v220419_01_CelebA_4by3_244x324.db"),
                os.path.join(dbprefix, "Test_v220419_01_LD3007_4by3_244x324.db"),
                os.path.join(dbprefix, "Test_v220419_01_LDRGB_4by3_244x324.db"),
                os.path.join(dbprefix, "Test_v220419_01_SiW_4by3_244x324.db"),
                os.path.join(dbprefix, "Test_v220419_01_Emotion_4by3_244x324.db")]

def save_ckpt(epoch, net, optimizer):
  if os.path.exists(strckptpath) == False:
    os.makedirs(strckptpath)
  strpath = "{}/epoch_{:02d}.ckpt".format(strckptpath, epoch)
  logger.print ("Save ckpt to {}".format(strpath))
  torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
  }, strpath)

def trainepoch(epoch, trainloader, model, criterion, optimizer, averagemetermap):
  #_t['forward_pass'].tic()
  fbtimer = Timer()
  totaliter = len(trainloader)
  # probsm = nn.Softmax(dim=1)
  for index, (images, labels, imgpath) in enumerate(trainloader):
    fbtimer.tic()
    images, labels = images.cuda(), labels.cuda()
    optimizer.zero_grad()
    logit = model(images)
    # prob = probsm(logit)
    loss = criterion(logit, labels)
    acc = accuracy(logit, labels)
    loss.backward()
    optimizer.step()
    averagemetermap["loss_am"].update(loss.item())
    averagemetermap["acc_am"].update(acc[0].item())
    if index % 10 == 0:
      fbtimer.toc()
      strprint = "  {}/{} at {}/{} loss:{:.5f} acc:{:.5f} lr:{:.5f} time:{:.5f}".format(index,
                                                                                     totaliter,
                                                                                     epoch,
                                                                                     args.epochs,
                                                                                     averagemetermap["loss_am"].avg,
                                                                                     averagemetermap["acc_am"].avg,
                                                                                     optimizer.param_groups[0]['lr'],
                                                                                     fbtimer.average_time)

      logger.print (strprint)


def trainmodel():
  """
  """
  averagemetermap = {}
  averagemetermap["loss_am"] = AverageMeter()
  averagemetermap["acc_am"] = AverageMeter()
  epochtimer = Timer()

  mynet = getbaseresnet18()
  mynet = mynet.cuda()

  if "260x260" in args.lmdbpath:
    transforms = T.Compose([T.RandomCrop((256, 256)),
                            T.RandomVerticalFlip(),
                            T.ToTensor()])  # 0 to 1
  elif "244x324" in args.lmdbpath:
    transforms = T.Compose([T.RandomCrop((320, 240)),
                            T.RandomVerticalFlip(),
                            T.ToTensor()])  # 0 to 1


  traindataset = lmdbDataset(args.lmdbpath, transforms)


  logger.print(mynet)
  logger.print(traindataset)
  trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.works, pin_memory=True)
  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = optim.Adam(mynet.parameters(), lr=args.lr, weight_decay=1e-4)
  # https://gaussian37.github.io/dl-pytorch-lr_scheduler/
  # https://sanghyu.tistory.com/113
  # ExponentialLR, LamdaLR same iof gamma is simple
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

  for epoch in range(args.epochs):
    mynet.train()
    epochtimer.tic()
    trainepoch(epoch, trainloader, mynet, criterion, optimizer, averagemetermap)
    epochtimer.toc()
    strprint = "{}/{} loss:{:.5f} acc:{:.5f} lr:{:.5f} time:{:.5f}".format(epoch, args.epochs, averagemetermap["loss_am"].avg, averagemetermap["acc_am"].avg, optimizer.param_groups[0]['lr'], epochtimer.average_time)
    logger.print (strprint)
    scheduler.step()
    save_ckpt(epoch, mynet, optimizer)

    if epoch > 58:
      for testdbpath in testdblist:
        testmodel(epoch, mynet, testdbpath, strckptpath)


if __name__ == '__main__':
  trainmodel()
  copyfile(strlogpath, "{}/trainlog.txt".format(strckptpath))

  # epoch = 10
  # strpath = "{}/epoch_{:02d}.ckpt".format(strckptpath, epoch)
  # print (strpath)

