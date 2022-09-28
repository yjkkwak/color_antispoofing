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

    self.factlen = self.env.stat()["entries"]
    self.len = self.factlen // 5


    # debug
    # index = 1100
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
    return self.len#self.env.stat()["entries"]

  def __getitem__(self, xindex):
    index = np.random.randint(0, self.factlen)
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


class lmdbVideoDataset(tdata.Dataset):
  def __init__(self, db_path, transform=None):
    self.env = None
    self.txn = None
    self.transform = transform
    self.db_path = db_path
    self.db_path_img = "{}{}".format(self.db_path, ".path")
    self.videopath = {}
    self.videokeys = []
    self.setsinglevideo()
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()

  def _init_db(self):
    self.env = lmdb.open(self.db_path,
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    self.txn = self.env.begin()

  def setkeys(self, strkey, strpath):
    if strkey in self.videopath.keys():
      self.videopath[strkey].append(strpath)
    else:
      self.videopath[strkey] = []
      self.videopath[strkey].append(strpath)

  def setsinglevideo(self):
    fpath = open(self.db_path_img, "r")
    strlines = fpath.readlines()
    for index, strline in enumerate(strlines):
      strline = strline.strip()
      if "MSU-MFSD" in strline or "REPLAY-ATTACK" in strline:
        if ".mov" in strline:
          strtokens = strline.split(".mov")
          strkey = "{}.mov".format(strtokens[0])
        else:
          strtokens = strline.split(".mp4")
          strkey = "{}.mp4".format(strtokens[0])
        self.setkeys(strkey, index)
      elif "OULU-NPU" in strline or "CASIA-MFSD" in strline:
        strkey = os.path.dirname(strline)
        self.setkeys(strkey, index)
    self.videokeys = list(self.videopath.keys())

  def __len__(self):
    return len(self.videokeys)

  def __getitem__(self, reindex):
    strkey = self.videokeys[reindex]
    listofindex = self.videopath[strkey]

    files_total = len(listofindex)
    interval = (files_total // 10) + 1
    index = listofindex[interval]

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
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  #mydataset = lmdbDataset("/home/user/work_db/v220401_01/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260.db", transforms)
  mydataset = lmdbDataset("/home/user/work_db/v220419_01/Dev_v220419_01_OULUNPU_1by1_260x260.db/",
                          transforms)
  trainloader = DataLoader(mydataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
  for item, label, imgpath in trainloader:
    print (item.shape, label.shape, imgpath[0])
    for iii, fff in enumerate(label):
      print (fff, imgpath[iii])
    break