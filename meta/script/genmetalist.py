import glob
import os
import numpy as np
import torch
import random
from itertools import combinations
random_seed = 20220321

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

datatypes = ["Train", "Test"]

data4C3paths = ["/home/user/work_db/PublicDB/CASIA-MFSD/",  # train_jpg / test_jpg  --> fd          --> real spoof
                "/home/user/work_db/PublicDB/MSU-MFSD/",  # train_jpg / test_jpg    --> fd          --> real attack
                "/home/user/work_db/PublicDB/OULU-NPU/",  # train_jpg / test_jpg / devel_jpg/       --> real spoof
                "/home/user/work_db/PublicDB/REPLAY-ATTACK/",  # train_jpg / test_jpg / devel_jpg   --> real attack
             ]

datapaths = ["/home/user/data1/DBs/antispoofing/SiW/SiW_jpg/",  # Train/Test
             "/home/user/data1/DBs/antispoofing/LivenessDetection_RGB",  # Train/Test
             "/home/user/data1/DBs/antispoofing/LivenessDetection_3007",  # Train/Test
             "/home/user/data2/s3/CW"
             ]

testonlydatapaths = ["/home/user/data1/DBs/antispoofing/RECOD-MPAD"]

liveitems = ["/live/", "real"]
spoofitems = ["/spoof/", "attack"]
datakeys = {}


def gentrainlist(strver, dbtypes, datacompipaths, datapaths):
  allimagelist = []

  # allcommon dataset add _SiW, _AIHUBRGB, _AIHUB3007
  for dbpaths in datapaths:
    jpgitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
    pngitems = glob.glob("{}/**/*.png.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
    imgitems = jpgitems + pngitems
    print (dbpaths, len(imgitems))

    liveimages = [item for item in imgitems if any(liveitem in item for liveitem in liveitems)]
    spoofimges = [item for item in imgitems if any(spoofitem in item for spoofitem in spoofitems)]
    numoflive = len(liveimages)
    numofspoof = len(spoofimges)
    print(numoflive, numofspoof, len(imgitems), int(len(imgitems)*0.15), len(imgitems[0:int(len(imgitems)*0.15)]))
    # DB Live Spoof All
    # SIW 22028 42371 64399
    # aihubRGB 437760 656640  1094400 --> 164160
    # aihub3007 192150 576000  768150 --> 115222
    random.shuffle(imgitems)
    # SIW all use
    # aihub RGB and 3007 use 0.15
    if "SiW" in dbpaths or "CW" in dbpaths:
      allimagelist.extend(imgitems)
    else:
      allimagelist.extend(imgitems[0:int(len(imgitems)*0.15)])

  dbnameconcat = "_SiW_CW_AIHUBx2"
  for dbidxpath in datacompipaths:
    dbpaths = dbidxpath[1]
    if "CASIA" in dbpaths:
      dbnameconcat = "{}_CASIA".format(dbnameconcat)
    if "MSU" in dbpaths:
      dbnameconcat = "{}_MSU".format(dbnameconcat)
    if "OULU" in dbpaths:
      dbnameconcat = "{}_OULU".format(dbnameconcat)
    if "REPLAY" in dbpaths:
      dbnameconcat = "{}_REPLAY".format(dbnameconcat)

    print (os.path.join(dbpaths, "{}_jpg".format(dbtypes.lower())))
    #jpg jpeg png
    jpgitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, "{}_jpg".format(dbtypes.lower()))), recursive=True)


    print(dbpaths, len(jpgitems))

    if dbpaths in datakeys.keys():
      allimagelist.extend(datakeys[dbpaths])
      continue
    else:
      datakeys[dbpaths] = []
      datakeys[dbpaths].extend(jpgitems)

    print(len(jpgitems), dbpaths, dbtypes)
    allimagelist.extend(jpgitems)

  print (len(allimagelist))

  strtrainlist = "./{}_{}{}.list".format(dbtypes, strver, dbnameconcat)
  with open(strtrainlist, "w") as the_file:
    for imgpath in allimagelist:
      the_file.write("{}\n".format(imgpath.replace(".fd","")))
    the_file.close()

def gentestlist(strver, dbtypes):
  for dbpaths in testonlydatapaths:
    if "RECOD" in dbpaths:
      dbnameconcat = "RECOD"

    allimagelist = []
    pngitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)

    allimagelist.extend(pngitems)

    strtrainlist = "./{}_{}_{}.list".format(dbtypes, strver, dbnameconcat)
    with open(strtrainlist, "w") as the_file:
      for imgpath in allimagelist:
        if "mask" in imgpath.lower(): continue
        the_file.write("{}\n".format(imgpath.replace(".fd", "")))
      the_file.close()

def gentestsubset():
  fd_path = "./../v220922/Test_4C0_RECOD.list"
  with open(fd_path, "r") as the_file:
    strlines = the_file.readlines()
    the_file.close()
  with open("./../v220922/TestSubSet_4C0_RECOD.list", "w") as the_file:
    for strline in strlines:
      # print(strline)
      if "user_4" in strline.strip():
        print (strline)
        the_file.write("{}".format(strline))
    the_file.close()


def main():
  print ("HI")
  # for datacombi in list(combinations(enumerate(data4C3paths), 4)):
  #   gentrainlist("4C4", datatypes[0], datacombi, datapaths)
  # gentestlist("4C0", datatypes[1])
  gentestsubset()

if __name__ == '__main__':
  main()
