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

class lmdbDatasetwpdle(tdata.Dataset):
  def __init__(self, db_path, transform=None, lk=11):
    # 11 -1 -> 10
    self.lk = lk-1
    self.env = None
    self.txn = None
    self.transform = transform
    self.db_path = db_path
    self.db_path_img = "{}{}".format(self.db_path, ".path")
    self.mydatum = mydatum_pb2.myDatum()
    self._init_db()
    self.uuid = {}
    self.factlen = self.env.stat()["entries"]
    self.len = self.factlen//int(5*2)

  def _init_db(self):
    self.env = lmdb.open(self.db_path,
                         readonly=True, lock=False,
                         readahead=False, meminit=False)
    self.txn = self.env.begin()

  def __len__(self):
    return self.len

  def rand_bbox(self, size, lam):
    # tensor
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

  def getitem(self, index):
    strid = "{:08}".format(index)
    lmdb_data = self.txn.get(strid.encode("ascii"))
    self.mydatum.ParseFromString(lmdb_data)
    dst = np.fromstring(self.mydatum.data, dtype=np.uint8)
    dst = dst.reshape(self.mydatum.height, self.mydatum.width, self.mydatum.channels)
    img = Image.fromarray(dst)
    label = self.mydatum.label
    imgpath = self.mydatum.path

    return img, label, imgpath

  def getpairitem(self, index, label):
    rindex = np.random.randint(0, self.len)

    while(index == rindex):
      rindex = np.random.randint(0, self.len)

    rimg, rlabel, rimgpath = self.getitem(rindex)

    while(label == rlabel or index == rindex):
      rindex = np.random.randint(0, self.len)
      rimg, rlabel, rimgpath = self.getitem(rindex)

    return rimg, rlabel, rimgpath

  def __getitem__(self, xindex):
    index = np.random.randint(0, self.factlen)
    img, label, imgpath = self.getitem(index)
    rimg, rlabel, rimgpath = self.getpairitem(index, label)
    strtoken = imgpath.split("/")
    strrtoken = rimgpath.split("/")

    # print(rimgpath)

    splidx = 5
    splridx = 5
    if "/data1/" in imgpath:
      splidx = 6
    if "/data1/" in rimgpath:
      splridx = 6
    if strtoken[splidx] not in self.uuid.keys():
      self.uuid[strtoken[splidx]] = len(self.uuid.keys())
    if strrtoken[splridx] not in self.uuid.keys():
      self.uuid[strrtoken[splridx]] = len(self.uuid.keys())
    # print (self.uuid)
    #print (strtoken[5], strrtoken[5], self.uuid[strtoken[5]], self.uuid[strrtoken[5]])
    if self.transform is not None:
      img = self.transform(img)
      rimg = self.transform(rimg)

    lam = np.random.randint(1, self.lk)
    lam /= self.lk
    bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), 1 - lam)
    lam = (bbx2 - bbx1) * (bby2 - bby1) / (img.size()[1] * img.size()[2])
    # print(lam, imgpath, rimgpath)
    rimg[:, bbx1:bbx2, bby1:bby2] = img[:, bbx1:bbx2, bby1:bby2]

    if rlabel == 1:
      lam = 1.0 - lam

    ### R2
    # if lam > 0.5:
    #   lam = 1.0
    # else:
    #   lam = 0.0
    ###
    return img, label, imgpath, rimg, lam, self.uuid[strtoken[splidx]], self.uuid[strrtoken[splridx]]


if __name__ == '__main__':

  transforms = T.Compose([T.RandomHorizontalFlip(),
                          #T.RandomRotation(180),
                          T.RandomCrop((256, 256)),
                          T.ToTensor()])  # 0 to 1

  mydataset = lmdbDatasetwpdle("/home/user/work_db/v220922/Train_4C3_SiW_RECOD_AIHUBx2_MSU_OULU_REPLAY_1by1_260x260.db.sort/",
                          transforms)
  print(len(mydataset))
  trainloader = DataLoader(mydataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
  for imgp1, label, imgpath, rimg, lam, uid1, uid2 in trainloader:
  # for imgmap in trainloader:
  #   imgs = imgmap["imgs"]
  #   for iii, fff in enumerate(label):
  #     print (fff, imgpath[iii])
  #   break
  #   for sid in range(imgs.shape[1]):
  #     print(sid, imgs[0,sid,:,:,:].shape)
  #     print(sid, imgs[0, sid, :, :, :].flatten()[0:10])
  #     print(sid, imgs[1, sid, :, :, :].flatten()[0:10])
  #     # print(subi, images_x[0, subi, :, :, :].flatten()[0:10])
  #     # print(subi, images_x[1, subi, :, :, :].flatten()[0:10])
  #     # to_pil_image(imgs[0,sid,:,:,:]).show()
  #   break
    # rand_idx = torch.randperm(rimg.shape[0])
    # vimgs = torch.cat((imgp1, rimg[rand_idx[0:rimg.shape[0]],]), dim=0)
    # vlabels = torch.cat((label, lam[rand_idx[0:rimg.shape[0]]]), dim=0)
    # print (label)
    # print(lam[rand_idx[0:rimg.shape[0]]])
    # print (vlabels)
    # vlabels = vlabels * 10
    # vlabels = vlabels.type(torch.LongTensor)
    # print(vlabels)
    #print (vimgs.shape, vlabels.shape)
    # imgp1 = item["imgp1"]
    # imgp2 = item["imgp2"]
    # imgp3 = item["imgp3"]
    # label = item["label"]
    # imgpath = item["imgpath"]
    #imgmix, mixlabel = cutmix_data(imgp1, label)
    # print(imgp1.shape)
    # print (rimg.shape)
    # print(label.shape)
    # print (lam.shape)
    print (uid1, uid2)
    to_pil_image(imgp1[0]).show()
    to_pil_image(rimg[0]).show()
    break


    # liveidx = torch.where(label == 1)
    # fakeidx = torch.where(label == 0)
    # print(label)
    # print(liveidx)
    # print(label[0], imgpath)
    # to_pil_image(imgp1[0]).show()
    # print(imgp1[0].shape)
    # break
    # #
    # # print (imgp1.shape, imgp2.shape, imgp3.shape, label.shape, imgpath[0])
    # for iii, fff in enumerate(label):
    #   print (fff, imgpath[iii], label[iii])
    #   # to_pil_image(imgp1[iii]).show()
    #   # break
    # # break