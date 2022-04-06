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

def gentrainlist(strver, dbtypes, datacompipaths):
  allimagelist = []
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
      liveimages = liveimages[0:30000]
    if len(spoofimges) > 300000:
      random.shuffle(spoofimges)
      spoofimges = spoofimges[0:60000]

    
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
        the_file.write("{}\n".format(imgpath))
      the_file.close()

def main():
  print ("HI")
  for datacombi in list(combinations(enumerate(datapaths), 3)):
    gentrainlist("v220401_01", datatypes[0], datacombi)
  #gentestlist("v220401_01", datatypes[1], datapaths)

if __name__ == '__main__':
  main()
