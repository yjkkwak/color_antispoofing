import glob
import tarfile
import zipfile
import os
from tqdm import tqdm
import multiprocessing as mp
import re
import cv2

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def renamefile(strpath):
  jpgitems = glob.glob("{}/*.jpg".format(strpath))
  jpegitems = glob.glob("{}/*.jpeg".format(strpath))
  print (len(jpgitems), len(jpegitems))
  allfiles = jpgitems + jpegitems
  for fpath in allfiles:
    fname = os.path.basename(fpath)
    dname = os.path.dirname(fpath)
    refpath = re.sub(r"[^a-zA-Z0-9-_.]", '', fname)
    if fpath == os.path.join(dname,refpath): continue
    os.rename(fpath, os.path.join(dname,refpath))

def checkwh(strpath):
  jpgitems = glob.glob("{}/*.jpg".format(strpath))
  jpegitems = glob.glob("{}/*.jpeg".format(strpath))
  print(len(jpgitems), len(jpegitems))
  allfiles = jpgitems + jpegitems
  print (len(allfiles))
  for fpath in allfiles:
    img = cv2.imread(fpath)
    h, w, c = img.shape
    if h > w:
      print (fpath, h, w, c)





def main():
  # renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Test/Val_Anger/")
  # renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Test/Val_Embarrass/")
  # renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Test/Val_Happy/")
  # renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Test/Val_Nutral/")
  # renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Test/Val_Sad/")
  # renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Test/Val_Unrest/")
  # renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Test/Val_Wound/")
  renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Train/Train_Anger_01")
  renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Train/Train_Embarrass_01")
  renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Train/Train_Happy_01")
  renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Train/Train_Nutral_01")
  renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Train/Train_Sad_01")
  renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Train/Train_Unrest_01")
  renamefile("/home/user/data1/DBs/antispoofing/koran_emotion_aihub/Train/Train_Wound_01")


if __name__ == '__main__':
    main()
