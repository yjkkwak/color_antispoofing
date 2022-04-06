import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import DataLoader


from networks import getresnet18_
from lmdbdataset import lmdbDataset
import os

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
                    default='/home/user/work_db/v220401_01/Train_v220401_01_CelebA_SiW_LD3007_1by1_260x260.db', help='db path')
parser.add_argument('--epochs', type=int, default=80, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'],
                    help='if you want to train model on cpu, pass "cpu" param')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(args.GPU)  # Set the GPU 2 to use

print(args)


def trainmodel():
  """
  """
  mynet = getresnet18_()

  transforms = T.Compose([T.RandomCrop((256, 256)),
                          T.RandomVerticalFlip(),
                          T.ToTensor()])# 0 to 1
  traindataset = lmdbDataset(args.lmdbpath, transforms)

  print (mynet)
  print(traindataset)
  trainloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
  for item, label in trainloader:
    print (item.shape, label.shape)
    break


if __name__ == '__main__':
  trainmodel()

