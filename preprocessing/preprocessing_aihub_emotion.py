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
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Validation/EMOIMG_Anger_VALID/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Validation/EMOIMG_Anxious_VALID/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Validation/EMOIMG_Embarrassed_VALID/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Validation/EMOIMG_Happy_VALID/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Validation/EMOIMG_Hurt_VALID/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Validation/EMOIMG_Neutral_VALID/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Validation/EMOIMG_Sad_VALID/")

  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Anger_TRAIN_01/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Anger_TRAIN_02/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Anger_TRAIN_03/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Anger_TRAIN_04/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Anxious_TRAIN_01/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Anxious_TRAIN_02/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Anxious_TRAIN_03/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Anxious_TRAIN_04/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Embarrassed_TRAIN_01/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Embarrassed_TRAIN_02/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Embarrassed_TRAIN_03/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Embarrassed_TRAIN_04/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Happy_TRAIN_01/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Happy_TRAIN_02/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Happy_TRAIN_03/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Happy_TRAIN_04/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Hurt_TRAIN_01/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Hurt_TRAIN_02/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Hurt_TRAIN_03/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Hurt_TRAIN_04/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Neutral_TRAIN_01/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Neutral_TRAIN_02/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Neutral_TRAIN_03/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Neutral_TRAIN_04/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Sad_TRAIN_01/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Sad_TRAIN_02/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Sad_TRAIN_03/")
  renamefile("/home/user/data1/DBs/AIHUB/Emotion/Training/EMOIMG_Sad_TRAIN_04/")


if __name__ == '__main__':
    main()
