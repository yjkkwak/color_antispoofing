import torch
import os
import lmdb
import numpy as np
import torch.utils.data as tdata
from torch.utils.data import DataLoader
from PIL import Image
import mydata.mydatum_pb2 as mydatum_pb2
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image

class lmdbDataset(tdata.Dataset):
  def __init__(self, db_path, transform=None):
    self.env = None
    self.txn = None
    self.transform = transform
    self.db_path = db_path
    self.db_path_img = "{}{}".format(self.db_path, ".path")
    print(self.db_path_img)
    self.allimgidxs = []
    self.setframes()
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()

    self.factlen = len(self.allimgidxs)#self.env.stat()["entries"]
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

  def setframes(self):
    fpath = open(self.db_path_img, "r")
    strlines = fpath.readlines()
    for index, strline in enumerate(strlines):
      strline = strline.strip()
      if "CW" in strline: continue
      self.allimgidxs.append(index)
    # print(self.allimgidxs)

  def __len__(self):
    return self.len#self.env.stat()["entries"]

  def __getitem__(self, xindex):
    reindex = np.random.randint(0, self.factlen)
    index = self.allimgidxs[reindex]
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

class lmdbDatasetTest(tdata.Dataset):
  def __init__(self, db_path, transform=None):
    self.env = None
    self.txn = None
    self.transform = transform
    self.db_path = db_path
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()

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
  # imgpath="/home/user/data2/s3/CW/Train/cw_0922/20220922_PA_BA/LIVE/PNG/285904_92126703_F_iphone_bottom/285904_F_Live_iphone_13_bottom_upper_front1.png"
  # pilimg = Image.open(imgpath).convert('RGB')
  # print(pilimg)
  #
  transforms = T.Compose([T.CenterCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  #mydataset = lmdbDataset("/home/user/work_db/v220401_01/Train_v220401_01_CelebA_LDRGB_LD3007_1by1_260x260.db", transforms)
  mydataset = lmdbDataset("/home/user/work_db/v220922/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db.sort",
                          transforms)
  trainloader = DataLoader(mydataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
  for item, label, imgpath in trainloader:
    if item.shape[1] == 4:
      print(item.shape, label.shape, imgpath[0])

      # to_pil_image(item[0]).show()
      # break
    # for iii, fff in enumerate(label):
    #   print (fff, imgpath[iii])
    # break
