from utils import Hook, nested_children
# from torchvision import models
from models.myresnet import myresnet18
import torch
import torch.nn as nn


def getresnet18():
  """
  """
  resnet18 = myresnet18(pretrained=False, num_classes=2)

  return resnet18

def debugmode():
  rinput = torch.randn((1, 3, 256, 256))
  mynet = getresnet18()
  print (mynet)
  forwardhook = []
  for l in nested_children(mynet):
    forwardhook.append(Hook(l))
  print (forwardhook)
  logit = mynet(rinput)

  for hook in forwardhook:
    print (hook.m, hook.input[0].shape, hook.output[0].shape)

if __name__ == '__main__':
  debugmode()

