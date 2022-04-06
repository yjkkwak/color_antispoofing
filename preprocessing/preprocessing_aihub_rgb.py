import glob
import tarfile
import zipfile
import os
from tqdm import tqdm
import multiprocessing as mp

def mydelcmd(strcmd):
  print (strcmd)
  os.system(strcmd)

def getzipfiles(strpath):
  strsrcpath = "{}/*.zip".format(strpath)
  flist = glob.glob(strsrcpath)
  print (flist)
  return flist

def getbasenamewoext(srcfile):
  pathname, extension = os.path.splitext(srcfile)
  return pathname

def unzipandcleandata(strsrcpath):
  fziplist = getzipfiles(strsrcpath)
  for fzippath in fziplist:
    print(fzippath)
    unzipfile(fzippath)

def unzipfile(srcfile):
  strwheretounzip = os.path.dirname(srcfile)
  print (srcfile, strwheretounzip)
  with zipfile.ZipFile(srcfile, "r") as zip_fp:
    for zitem in tqdm(zip_fp.namelist(), desc='Extracting '):
      zip_fp.extract(member=zitem, path=strwheretounzip)
  print (os.path.join(strwheretounzip, os.path.basename(srcfile)))
  fanout_untar(getbasenamewoext(srcfile))

def unziptar(fullpath):
  """worker unzips one file"""
  print ("extracting... {}".format(fullpath))
  tar = tarfile.open(fullpath, 'r:*')
  tar.extractall(os.path.dirname(fullpath))
  tar.close()
  removemateral(fullpath)

  strdelcmd = 'rm -rf {}'.format(fullpath)
  mydelcmd(strdelcmd)


def removemateral(untarpath):
  # keep item
  # base/camera/condition/real or fake/color or depth or ir
  funtarpath = getbasenamewoext(untarpath)

  # remove space and left phase
  dlsit = glob.glob("{}/*".format(funtarpath))
  for dsrcitem in dlsit:
    ddestitem = dsrcitem.replace(" ", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    os.rename(dsrcitem, ddestitem)

  """
  real_01_phone /
  real_01_tablet /
  attack_01_print_none_flat/
  attack_03_replay_phone/
  attack_04_replay_tablet/
  """
  keepmodality = ['real_01_phone', 'real_01_tablet','attack_01_print_none_flat','attack_03_replay_phone','attack_04_replay_tablet']


  dlsit = glob.glob("{}/*/*/*".format(funtarpath))
  for ditem in dlsit:
    if os.path.basename(ditem) not in keepmodality:
      strdelcmd = 'rm -rf {}'.format(ditem)
      mydelcmd(strdelcmd)

  dlsit = glob.glob("{}/*/*/*/*/crop".format(funtarpath))
  for ditem in dlsit:
    strdelcmd = 'rm -rf {}'.format(ditem)
    mydelcmd(strdelcmd)

def fanout_untar(tarpath):
  #test
  # tarpath = "/home/user/data1/DBs/antispoofing/testdb/1888"
  ftarlist = glob.glob("{}/*.tar".format(tarpath))

  # for taritem in ftarlist:
  #   print (taritem)
  #   unziptar(taritem)
  #   break

  pool = mp.Pool(min(mp.cpu_count(), len(ftarlist)))  # number of workers
  pool.map(unziptar, ftarlist, chunksize=1)
  pool.close()


def main():
  print ("abc")
  strsrcpath = "/home/user/data1/DBs/antispoofing/LivenessDetection_RGB/Validation"
  unzipandcleandata(strsrcpath)

  strsrcpath = "/home/user/data1/DBs/antispoofing/LivenessDetection_RGB/Training"
  unzipandcleandata(strsrcpath)




if __name__ == '__main__':
    main()
