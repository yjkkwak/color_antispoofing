from utils import Hook, nested_children
from torchvision import models
import torch
import torch.nn as nn


def getresnet18_():
  """
  """
  resnet18 = models.resnet18(pretrained=False, num_classes=2)
  resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=0,
                               bias=False)
  for m in resnet18.modules():
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  return resnet18

def debugmode():
  rinput = torch.randn((1, 3, 256, 256))
  mynet = getresnet18_()

  forwardhook = []
  for l in nested_children(mynet):
    forwardhook.append(Hook(l))
  # print (forwardhook)
  logit = mynet(rinput)

  for hook in forwardhook:
    print (hook.m, hook.input[0].shape, hook.output[0].shape)

if __name__ == '__main__':
  debugmode()

