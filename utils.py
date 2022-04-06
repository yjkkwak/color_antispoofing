import torch

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