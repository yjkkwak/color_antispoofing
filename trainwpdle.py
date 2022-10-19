import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader
from networks import getbaseresnet18wgrl
#from lmdbdataset import lmdbDataset
from lmdbpdledataset import lmdbDatasetwpdle
from utils import AverageMeter, accuracy, Timer, getbasenamewoext, Logger
import os
import shortuuid
from datetime import datetime
from test import testpdlemodel
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
      os.path.join(dbprefix, "Test_4C0_RECOD_1by1_260x260.db.sort")]
  elif "244x324" in args.lmdbpath:
    testdblist = [
      os.path.join(dbprefix, "Test_4C0_RECOD_4by3_244x324.db.sort")]

  print(testdblist)

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


def save_ckpt(args, epoch, net, optimizer, islast=False):
  if os.path.exists(args.strckptpath) == False:
    os.makedirs(args.strckptpath)
  if islast:
    strpath = "{}/epoch_last.ckpt".format(args.strckptpath, epoch)
  else:
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
  regrsteps = torch.linspace(0, 1.0, steps=11).cuda()
  probsm = nn.Softmax(dim=1)
  for index, (tmpimages, tmplabels, imgpath, rimg, rlab, tmpuid1, tmpuid2) in enumerate(trainloader):
    fbtimer.tic()
    rand_idx = torch.randperm(rimg.shape[0])
    images = torch.cat((tmpimages, rimg[rand_idx,]), dim=0)
    labels = torch.cat((tmplabels, rlab[rand_idx]), dim=0)
    uid1 = torch.cat((tmpuid1, tmpuid2[rand_idx]), dim=0)
    labels = labels.type(torch.FloatTensor)

    images, labels = images.cuda(), labels.cuda()
    uid1 = uid1.cuda()
    optimizer.zero_grad()
    logit, dislogit = model(images)
    prob = probsm(logit)
    expectprob = torch.sum(regrsteps * prob, dim=1)
    mseloss = criterion["mse"](expectprob, labels)
    advclsloss = criterion["cls"](dislogit, uid1)
    # loss = args.lamda*mseloss + (1.0-args.lamda)*advclsloss
    loss = mseloss + args.lamda * advclsloss
    tmplogit = torch.zeros(images.size(0), 2).cuda()
    tmplogit[:, 1] = expectprob
    tmplogit[:, 0] = 1.0 - tmplogit[:, 1]
    acc = accuracy(tmplogit[0:tmpimages.shape[0], :], labels[0:tmplabels.shape[0]])
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

  mynet = getbaseresnet18wgrl(11, 7)

  mynet = mynet.cuda()

  if "260x260" in args.lmdbpath:
    transforms = T.Compose([T.RandomCrop((256, 256)),
                            T.RandomHorizontalFlip(),
                            T.ToTensor()])  # 0 to 1
  elif "244x324" in args.lmdbpath:
    transforms = T.Compose([T.RandomCrop((320, 240)),
                            T.RandomHorizontalFlip(),
                            T.ToTensor()])  # 0 to 1


  traindataset = lmdbDatasetwpdle(args.lmdbpath, transforms, 11)

  args.logger.print(mynet)
  args.logger.print(traindataset)
  trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.works, pin_memory=True)
  criterion = {}
  criterion["cls"] = nn.CrossEntropyLoss().cuda()
  criterion["mse"] = nn.MSELoss().cuda()

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
    if epoch > 20:
      sumhter = 0.0
      for testdbpath in args.testdblist:
        hter = testpdlemodel(epoch, mynet, testdbpath, args.strckptpath)
        sumhter += hter
      sumhter = sumhter/len(args.testdblist)

      if besthter > sumhter:
        besthter = sumhter
        save_ckpt(args, epoch, mynet, optimizer)
        copyfile(args.logger.getlogpath(), "{}/trainlog.txt".format(args.strckptpath))
    # lastckpt
    save_ckpt(args, epoch, mynet, optimizer, True)

if __name__ == '__main__':
  main()


