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
datapaths = ["/home/user/data1/DBs/antispoofing/CelebA_spoofing/CelebA_Spoof/Data",  # Train/Test
             "/home/user/data1/DBs/antispoofing/SiW/SiW_jpg/",  # Train/Test
             "/home/user/data1/DBs/antispoofing/LivenessDetection_RGB",  # Train/Test
             "/home/user/data1/DBs/antispoofing/LivenessDetection_3007"  # Train/Test
             ]
liveitems = ["/live/", "real"]
spoofitems = ["/spoof/", "attack"]
datakeys = {}

onlylivedatapath = ["/home/user/data1/DBs/antispoofing/koran_emotion_aihub"]


def getStatistics():
  for dbpaths in datapaths:
    for dbtypes in datatypes:
      print (dbpaths, dbtypes)
      pngitems = glob.glob("{}/**/*.png".format(os.path.join(dbpaths, dbtypes)), recursive=True)
      jpgitems = glob.glob("{}/**/*.jpg".format(os.path.join(dbpaths, dbtypes)), recursive=True)
      imgitmes = pngitems+jpgitems
      liveimages = [item for item in imgitmes if any(liveitem in item for liveitem in liveitems)]
      spoofimges = [item for item in imgitmes if any(spoofitem in item for spoofitem in spoofitems)]
      numoflive = len(liveimages)
      numofspoof = len(spoofimges)
      print (numoflive, numofspoof, len(imgitmes))

def getStatiscticwOnlyLivedata():
  for dbpaths in onlylivedatapath:
    for dbtypes in datatypes:
      print (dbpaths, dbtypes)
      jpgimages = glob.glob("{}/**/*.jpg".format(os.path.join(dbpaths, dbtypes)), recursive=True)
      jpegimages = glob.glob("{}/**/*.jpeg".format(os.path.join(dbpaths, dbtypes)), recursive=True)
      liveimages = jpgimages + jpegimages


      print (len(jpgimages), len(jpegimages), len(liveimages))


def gentrainlist(strver, dbtypes, datacompipaths):
  allimagelist = []
  ###
  with open("./../v220419_01/Train_v220419_01_Emotion.list", "r") as the_file:
    abc = the_file.readlines()
    the_file.close()
  for iii in abc:
    allimagelist.append(iii.strip())
  ###
  dbnameconcat = ""
  for dbidxpath in datacompipaths:
    dbpaths = dbidxpath[1]
    if "CelebA" in dbpaths:
      dbnameconcat = "{}_CelebA".format(dbnameconcat)
    if "SiW" in dbpaths:
      dbnameconcat = "{}_SiW".format(dbnameconcat)
    if "LivenessDetection_RGB" in dbpaths:
      dbnameconcat = "{}_LDRGB".format(dbnameconcat)
    if "LivenessDetection_3007" in dbpaths:
      dbnameconcat = "{}_LD3007".format(dbnameconcat)
    pngitems = glob.glob("{}/**/*.png.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
    jpgitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
    imgitmes = pngitems+jpgitems
    liveimages = [item for item in imgitmes if any(liveitem in item for liveitem in liveitems)]
    spoofimges = [item for item in imgitmes if any(spoofitem in item for spoofitem in spoofitems)]

    if len(liveimages) > 100000:
      random.shuffle(liveimages)
      if "CelebA_Spoof" in dbpaths:
        liveimages = liveimages[0:60000]
      else:
        liveimages = liveimages[0:40000]
    if len(spoofimges) > 300000:
      random.shuffle(spoofimges)
      if "CelebA_Spoof" in dbpaths:
        spoofimges = spoofimges[0:90000]
      else:
        spoofimges = spoofimges[0:70000]

    
    if dbpaths in datakeys.keys():
      print(len(liveimages), len(spoofimges), dbpaths, dbtypes)
      allimagelist.extend(datakeys[dbpaths])
      continue
    else:
      datakeys[dbpaths] = []
      datakeys[dbpaths].extend(liveimages)
      datakeys[dbpaths].extend(spoofimges)
      
    print(len(liveimages), len(spoofimges), dbpaths, dbtypes)
    allimagelist.extend(liveimages)
    allimagelist.extend(spoofimges)
 

  random.shuffle(allimagelist)
  print (len(allimagelist), allimagelist[0:10])

  strtrainlist = "./../{}/{}_{}{}.list".format(strver, dbtypes, strver, dbnameconcat)
  with open(strtrainlist, "w") as the_file:
    for imgpath in allimagelist:
      the_file.write("{}\n".format(imgpath.replace(".fd","")))
    the_file.close()


def gentestlist(strver, dbtypes, datacompipaths):
  for dbpaths in datacompipaths:
    dbnameconcat = ""
    if "CelebA" in dbpaths:
      dbnameconcat = "{}_CelebA".format(dbnameconcat)
    if "SiW" in dbpaths:
      dbnameconcat = "{}_SiW".format(dbnameconcat)
    if "LivenessDetection_RGB" in dbpaths:
      dbnameconcat = "{}_LDRGB".format(dbnameconcat)
    if "LivenessDetection_3007" in dbpaths:
      dbnameconcat = "{}_LD3007".format(dbnameconcat)

    pngitems = glob.glob("{}/**/*.png.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
    jpgitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
    imgitmes = pngitems+jpgitems
    liveimages = [item for item in imgitmes if any(liveitem in item for liveitem in liveitems)]
    spoofimges = [item for item in imgitmes if any(spoofitem in item for spoofitem in spoofitems)]
    if len(liveimages) > 50000:
      random.shuffle(liveimages)
      liveimages = liveimages[0:16000]
    if len(spoofimges) > 80000:
      random.shuffle(spoofimges)
      spoofimges = spoofimges[0:39000]

    print(len(liveimages), len(spoofimges), dbpaths, dbtypes)
    allimagelist = []
    allimagelist.extend(liveimages)
    allimagelist.extend(spoofimges)

    strtrainlist = "./../{}/{}_{}{}.list".format(strver, dbtypes, strver, dbnameconcat)
    with open(strtrainlist, "w") as the_file:
      for imgpath in allimagelist:
        the_file.write("{}\n".format(imgpath.replace(".fd","")))
      the_file.close()

def genTraintestonlylivelist(strver, dbtypes, datacompipaths):
  for dbpaths in datacompipaths:
    dbnameconcat = ""
    if "emotion" in dbpaths:
      dbnameconcat = "{}_Emotion".format(dbnameconcat)

    jpgimages = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
    jpegimages = glob.glob("{}/**/*.jpeg.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
    liveimages = jpgimages + jpegimages

    print(len(liveimages), len(jpgimages), len(jpegimages), dbpaths, dbtypes)
    allimagelist = []
    allimagelist.extend(liveimages)

    if dbtypes == "Train":
      random.shuffle(allimagelist)
      allimagelist = allimagelist[0:40000]

    if dbtypes == "Test":
      random.shuffle(allimagelist)
      allimagelist = allimagelist[0:20000]

    strtrainlist = "./../{}/{}_{}{}.list".format(strver, dbtypes, strver, dbnameconcat)
    with open(strtrainlist, "w") as the_file:
      for imgpath in allimagelist:
        the_file.write("{}\n".format(imgpath.replace(".fd","")))
      the_file.close()
    return allimagelist

def main():
  print ("HI")
  genTraintestonlylivelist("v220419_01", datatypes[0], onlylivedatapath)
  genTraintestonlylivelist("v220419_01", datatypes[1], onlylivedatapath)

  for datacombi in list(combinations(enumerate(datapaths), 3)):
    gentrainlist("v220419_01", datatypes[0], datacombi)
  gentestlist("v220419_01", datatypes[1], datapaths)
  # getStatiscticwOnlyLivedata()

  for datacombi in list(combinations(enumerate(datapaths), 4)):
    print (datacombi)
    gentrainlist("v220419_01", datatypes[0], datacombi)



if __name__ == '__main__':
  main()
