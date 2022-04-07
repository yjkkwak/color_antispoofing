import torch
import os
import lmdb
import numpy as np
import torch.utils.data as tdata
from torch.utils.data import DataLoader
from PIL import Image
import mydata.mydatum_pb2 as mydatum_pb2
from torchvision import transforms as T

class lmdbDataset(tdata.Dataset):
  def __init__(self, db_path, transform=None):
    self.env = None
    self.txn = None
    self.transform = transform
    self.db_path = db_path
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()
    # debug
    # index = 23771
    # strid = "{:08}".format(index)
    # lmdb_data = self.txn.get(strid.encode("ascii"))
    # mydatum = mydatum_pb2.myDatum()
    #
    # mydatum.ParseFromString(lmdb_data)
    # print (mydatum.width, mydatum.height, mydatum.channels, mydatum.label)
    # print(mydatum.path)
    # dst = np.fromstring(mydatum.data, dtype=np.uint8)
    # dst = dst.reshape(mydatum.height, mydatum.width, mydatum.channels)
    # print (dst.shape)
    # imgpil = Image.fromarray(dst)
    # imgpil.show()

  def _init_db(self):
    self.env = lmdb.open(self.db_path,
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    self.txn = self.env.begin()

  def __len__(self):
    return self.env.stat()["entries"]

  def __getitem__(self, index):
    strid = "{:08}".format(index)
    lmdb_data = self.txn.get(strid.encode("ascii"))
    self.mydatum.ParseFromString(lmdb_data)
    dst = np.fromstring(self.mydatum.data, dtype=np.uint8)
    dst = dst.reshape(self.mydatum.height, self.mydatum.width, self.mydatum.channels)
    img = Image.fromarray(dst)
    label = self.mydatum.label
    imgpath = self.mydatum.path

    if self.transform is not None:
      img = self.transform(img)

    return img, label, imgpath

if __name__ == '__main__':
  transforms = T.Compose([
                          T.ToTensor()])  # 0 to 1

  mydataset = lmdbDataset("/home/user/work_db/v220401_01/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260.db", transforms)
  trainloader = DataLoader(mydataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
  for item, label, imgpath in trainloader:
    print (item.shape, label.shape, imgpath[0])
    break