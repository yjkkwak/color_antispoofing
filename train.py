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

def initargments():
  parser = argparse.ArgumentParser(description='anti-spoofing training')
  parser.add_argument('--lmdbpath', type=str,
                      default='/home/user/work_db/v220922/Train_4C3_SiW_RECOD_AIHUBx2_MSU_OULU_REPLAY_1by1_260x260.db.sort',
                      help='db path')
  parser.add_argument('--ckptpath', type=str,
                      default='/home/user/model_2022/v220922', help='ckpt path')
  parser.add_argument('--epochs', type=int, default=80, help='num of epochs')
  parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
  parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
  parser.add_argument('--works', type=int, default=4, help='works')
  parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
  parser.add_argument('--gamma', type=float, default=0.97, help='gamma for scheduler')
  parser.add_argument('--opt', type=str, default='adam', help='sgd or adam')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum for scheduler')
  parser.add_argument('--meta', type=str, default='meta', help='meta')
  parser.add_argument('--resume', type=str, default='', help='resume path')
  parser.add_argument('--random_seed', type=int, default=20220406, help='random_seed')
  parser.add_argument('--lamda', type=float, default=1.0, help='gamma for scheduler')

  args = parser.parse_args()

  torch.manual_seed(args.random_seed)
  torch.cuda.manual_seed(args.random_seed)
  torch.cuda.manual_seed_all(args.random_seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(args.random_seed)
  random.seed(args.random_seed)


  os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU)  # Set the GPU 2 to use
  struuid = "{}_{}_{}_bsize{}_opt{}_lr{}_gamma_{}_epochs_{}_meta_{}_lamda_{}".format(
    getbasenamewoext(os.path.basename(args.lmdbpath)),
    datetime.now().strftime("%y%m%d"),
    shortuuid.uuid(),
    args.batch_size,
    args.opt,
    args.lr,
    args.gamma,
    args.epochs,
    args.meta, args.lamda)

  if args.resume != "":
    print ("resume !!!")
    resumedir = os.path.dirname(args.resume)
    struuid = os.path.basename(resumedir)

  strckptpath = os.path.join(args.ckptpath, struuid)
  strlogpath = "/home/user/work_2022/logworkspace/{}.log".format(struuid)
  logger = Logger(strlogpath)
  logger.print(args)

  dbprefix = "/home/user/work_db/v220922"
  if "260x260" in args.lmdbpath:
    testdblist = [
                  os.path.join(dbprefix, "Test_4C1_RECOD_1by1_260x260.db.sort"),
                  os.path.join(dbprefix, "Test_4C1_FASD_1by1_260x260.db.sort")]

  if "CASIA_MSU_OULU" in args.lmdbpath:
    testdblist.append(os.path.join(dbprefix, "Test_4C1_REPLAY_1by1_260x260.db.sort"))
  elif "CASIA_MSU_REPLAY" in args.lmdbpath:
    testdblist.append(os.path.join(dbprefix, "Test_4C1_OULU_1by1_260x260.db.sort"))
  elif "CASIA_OULU_REPLAY" in args.lmdbpath:
    testdblist.append(os.path.join(dbprefix, "Test_4C1_MSU_1by1_260x260.db.sort"))
  elif "MSU_OULU_REPLAY" in args.lmdbpath:
    testdblist.append(os.path.join(dbprefix, "Test_4C1_CASIA_1by1_260x260.db.sort"))

  print (testdblist)

  # set args with external variables.
  args.struuid = struuid
  args.testdblist = testdblist
  args.logger = logger
  args.strckptpath = strckptpath
  print (args)
  return args



def main():
  args = initargments()
  trainmodel(args)
  copyfile(args.logger.getlogpath(), "{}/trainlog.txt".format(args.strckptpath))


def save_ckpt(args, epoch, net, optimizer):
  if os.path.exists(args.strckptpath) == False:
    os.makedirs(args.strckptpath)
  strpath = "{}/epoch_{:02d}.ckpt".format(args.strckptpath, epoch)
  args.logger.print ("Save ckpt to {}".format(strpath))
  torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
  }, strpath)

def trainepoch(args, epoch, trainloader, model, criterion, optimizer, averagemetermap):
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

      args.logger.print (strprint)


def trainmodel(args):
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
                            T.RandomHorizontalFlip(),
                            T.ToTensor()])  # 0 to 1
  elif "244x324" in args.lmdbpath:
    transforms = T.Compose([T.RandomCrop((320, 240)),
                            T.RandomHorizontalFlip(),
                            T.ToTensor()])  # 0 to 1


  traindataset = lmdbDataset(args.lmdbpath, transforms)

  args.logger.print(mynet)
  args.logger.print(traindataset)
  trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.works, pin_memory=True)
  criterion = nn.CrossEntropyLoss().cuda()
  if args.opt.lower() == "adam":
    # works
    optimizer = optim.Adam(mynet.parameters(), lr=args.lr, weight_decay=5e-4)
  else:
    optimizer = optim.SGD(mynet.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=5e-4)
  print (optimizer)
  # https://gaussian37.github.io/dl-pytorch-lr_scheduler/
  # https://sanghyu.tistory.com/113
  # ExponentialLR, LamdaLR same iof gamma is simple
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

  startepoch = 0
  if args.resume != "":
    args.logger.print("Resume from {}".format(args.resume))
    checkpoint = torch.load(args.resume)
    mynet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    startepoch = checkpoint['epoch'] + 1

  besthter = 100.0
  for epoch in range(startepoch, args.epochs):
    mynet.train()
    epochtimer.tic()
    trainepoch(args, epoch, trainloader, mynet, criterion, optimizer, averagemetermap)
    epochtimer.toc()
    strprint = "{}/{} loss:{:.5f} acc:{:.5f} lr:{:.5f} time:{:.5f}".format(epoch, args.epochs, averagemetermap["loss_am"].avg, averagemetermap["acc_am"].avg, optimizer.param_groups[0]['lr'], epochtimer.average_time)
    args.logger.print (strprint)
    scheduler.step()
    if epoch > 20 and averagemetermap["acc_am"].avg > 99.0:
      sumhter = 0.0
      for testdbpath in args.testdblist:
        hter = testmodel(epoch, mynet, testdbpath, args.strckptpath)
        sumhter += hter
      sumhter = sumhter/3.0

      if besthter > sumhter:
        besthter = sumhter
        save_ckpt(args, epoch, mynet, optimizer)
        copyfile(args.strlogpath, "{}/trainlog.txt".format(args.strckptpath))



if __name__ == '__main__':
  main()


