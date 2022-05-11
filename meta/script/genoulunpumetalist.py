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

datatypes = ["Train", "Test", "Dev"]
datapaths = ["/home/user/data1/DBs/antispoofing/OULU-NPU/",  # Train/Test/Dev
             ]
liveitems = ["/real/"]
spoofitems = ["/spoof/"]
datakeys = {}

def getStatistics():
  for dbpaths in datapaths:
    for dbtypes in datatypes:
      print (dbpaths, dbtypes)
      jpgitems = glob.glob("{}/**/*.jpg".format(os.path.join(dbpaths, dbtypes)), recursive=True)
      imgitmes = jpgitems
      liveimages = [item for item in imgitmes if any(liveitem in item for liveitem in liveitems)]
      spoofimges = [item for item in imgitmes if any(spoofitem in item for spoofitem in spoofitems)]
      numoflive = len(liveimages)
      numofspoof = len(spoofimges)
      print (numoflive, numofspoof, len(imgitmes))

def gentestlist(strver, dbtypes, datacompipaths):
  for dbpaths in datacompipaths:
    dbnameconcat = ""
    if "OULU-NPU" in dbpaths:
      dbnameconcat = "{}_OULUNPU".format(dbnameconcat)

    jpgitems = glob.glob("{}/**/*.jpg.fd".format(os.path.join(dbpaths, dbtypes)), recursive=True)
    imgitmes = jpgitems
    liveimages = [item for item in imgitmes if any(liveitem in item for liveitem in liveitems)]
    spoofimges = [item for item in imgitmes if any(spoofitem in item for spoofitem in spoofitems)]

    # random.shuffle(spoofimges)
    # spoofimges = spoofimges[0:102960]
    print(len(liveimages), len(spoofimges), dbpaths, dbtypes)
    allimagelist = []
    allimagelist.extend(liveimages)
    allimagelist.extend(spoofimges)

    strtrainlist = "./../{}/{}_{}{}.list".format(strver, dbtypes, strver, dbnameconcat)
    with open(strtrainlist, "w") as the_file:
      for imgpath in allimagelist:
        the_file.write("{}\n".format(imgpath.replace(".fd","")))
      the_file.close()

def main():
  print ("HI")
  gentestlist("v220419_01", datatypes[0], datapaths)
  # gentestlist("v220419_01", datatypes[2], datapaths)

if __name__ == '__main__':
  main()
