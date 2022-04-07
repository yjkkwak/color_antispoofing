import torch
import time
import os

class Hook():
  def __init__(self, module, backward=False):
    if backward == False:
      self.hook = module.register_forward_hook(self.hook_fn)
    else:
      self.hook = module.register_backward_hook(self.hook_fn)

  def hook_fn(self, module, input, output):
    self.m = module
    self.input = input
    self.output = output

  def close(self):
    self.hook.remove()


def nested_children(model: torch.nn.Module):
  children = list(model.children())
  flatt_children = []
  if children == []:
    # if model has no children; model is last child! :O
    return model
  else:
    # look for children from children... to the last child!
    for child in children:
      try:
        flatt_children.extend(nested_children(child))
      except TypeError:
        flatt_children.append(nested_children(child))
  return flatt_children


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res




class Timer(object):
  """A simple timer."""

  def __init__(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.

  def tic(self):
    # using time.time instead of time.clock because time time.clock
    # does not normalize for multithreading
    self.start_time = time.time()

  def toc(self, average=True):
    self.diff = time.time() - self.start_time
    self.total_time += self.diff
    self.calls += 1
    self.average_time = self.total_time / self.calls
    if average:
      return self.average_time
    else:
      return self.diff

  def clear(self):
    self.total_time = 0.
    self.calls = 0
    self.start_time = 0.
    self.diff = 0.
    self.average_time = 0.

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname